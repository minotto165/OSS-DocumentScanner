package com.akylas.documentscanner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer

/**
 * Neural page-flattening (vFlat-style dewarp) for scanned crops.
 *
 * Model: dewarp_flattener.onnx (Flattener bundle 1.1.0, Apache-2.0, 31.6 MB),
 * a uvdocnet model: input (1,3,712,488) RGB/255, output (1,2,45,31) grid in
 * flattener convention (+-1 = outer source-pixel edges). The dense remap and
 * sampling live in C++ ([nativeDewarpRemap], mirroring flattener/ops.py and
 * flattener/dewarp/coords.py exactly).
 *
 * Failure policy: dewarp is a QUALITY stage, never a correctness gate. Any
 * failure returns the input bitmap unchanged.
 */
object Dewarp {
    private const val TAG = "Dewarp"
    private const val MODEL_ASSET = "dewarp_flattener.onnx"
    private const val IN_W = 488
    private const val IN_H = 712
    private const val GH = 45
    private const val GW = 31
    private const val GRID_LEN = 2 * GH * GW
    private const val TTA_GAP = 0.03f

    @Volatile
    private var session: OrtSession? = null

    @Synchronized
    private fun getSession(context: Context): OrtSession? {
        session?.let { return it }
        return try {
            val bytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions()
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // CPU only for v1 (deterministic). NNAPI/quantization is a follow-up.
            val s = env.createSession(bytes, opts)
            session = s
            Log.i(TAG, "dewarp session ready (${bytes.size} bytes)")
            s
        } catch (e: Exception) {
            Log.w(TAG, "dewarp session failed: $e")
            null
        }
    }

    /**
     * Returns a flattened copy of [src], or [src] itself when dewarp is
     * unavailable or rejected. Never throws.
     */
    fun dewarpBitmap(context: Context, src: Bitmap): Bitmap {
        try {
            val session = getSession(context) ?: return src
            val w = src.width
            val h = src.height
            if (w < 2 || h < 2) return src
            val inputName = session.inputNames.first()
            // Preprocess: bilinear resize to 488x712 (align_corners=False
            // equivalent), RGB / 255, NCHW. Matches flattener ops.resize.
            val small = Bitmap.createScaledBitmap(src, IN_W, IN_H, true)
            val px = IntArray(IN_W * IN_H)
            small.getPixels(px, 0, IN_W, 0, 0, IN_W, IN_H)
            small.recycle()
            val data = FloatArray(3 * IN_H * IN_W)
            for (y in 0 until IN_H) {
                for (x in 0 until IN_W) {
                    val c = px[y * IN_W + x]
                    val base = y * IN_W + x
                    data[base] = ((c shr 16) and 0xFF) / 255f
                    data[IN_H * IN_W + base] = ((c shr 8) and 0xFF) / 255f
                    data[2 * IN_H * IN_W + base] = (c and 0xFF) / 255f
                }
            }
            // TTA: run original + horizontal mirror, unmirror, average when
            // the median node distance / 2 <= 0.03 (flattener predict_input).
            val batch = arrayOf(data, hflip(data))
            val grids = batch.map { d ->
                val buf = FloatBuffer.wrap(d)
                val tensor = OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(), buf, longArrayOf(1, 3, IN_H.toLong(), IN_W.toLong())
                )
                tensor.use {
                    val t0 = System.currentTimeMillis()
                    val out = session.run(mapOf(inputName to it))
                    out.use {
                        @Suppress("UNCHECKED_CAST")
                        val g = ((out[0].value as Array<*>)[0] as Array<*>) // (2, GH, GW)
                        val ch0 = g[0] as Array<FloatArray>
                        val ch1 = g[1] as Array<FloatArray>
                        Log.i(TAG, "dewarp infer ${System.currentTimeMillis() - t0} ms")
                        Pair(ch0, ch1)
                    }
                }
            }
            val grid = mergeTta(grids[0], grids[1])
            val flat = FloatArray(GRID_LEN)
            for (c in 0..1) for (i in 0 until GH) for (j in 0 until GW) {
                flat[c * GH * GW + i * GW + j] = grid[c][i][j]
            }
            val out = src.copy(Bitmap.Config.ARGB_8888, true) ?: return src
            val ok = try {
                nativeDewarpRemap(out, flat)
            } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "dewarp native missing: $e")
                false
            }
            if (!ok) {
                out.recycle()
                return src
            }
            return out
        } catch (e: Exception) {
            Log.w(TAG, "dewarp failed, keeping crop: $e")
            return src
        }
    }

    private fun hflip(nchw: FloatArray): FloatArray {
        val out = FloatArray(nchw.size)
        val plane = IN_H * IN_W
        for (c in 0..2) {
            for (y in 0 until IN_H) {
                for (x in 0 until IN_W) {
                    out[c * plane + y * IN_W + x] = nchw[c * plane + y * IN_W + (IN_W - 1 - x)]
                }
            }
        }
        return out
    }

    /** Merged (2, GH, GW) grid: TTA average when the mirror agrees, else the plain grid. */
    private fun mergeTta(
        orig: Pair<Array<FloatArray>, Array<FloatArray>>,
        flip: Pair<Array<FloatArray>, Array<FloatArray>>,
    ): Array<Array<FloatArray>> {
        // Unmirror: reverse columns, negate x.
        val mx = Array(GH) { FloatArray(GW) }
        val my = Array(GH) { FloatArray(GW) }
        for (i in 0 until GH) for (j in 0 until GW) {
            mx[i][j] = -flip.first[i][GW - 1 - j]
            my[i][j] = flip.second[i][GW - 1 - j]
        }
        val dists = FloatArray(GH * GW)
        var k = 0
        for (i in 0 until GH) for (j in 0 until GW) {
            val dx = orig.first[i][j] - mx[i][j]
            val dy = orig.second[i][j] - my[i][j]
            dists[k++] = kotlin.math.sqrt(dx * dx + dy * dy)
        }
        dists.sort()
        val gap = dists[(dists.size - 1) / 2] / 2f
        Log.i(TAG, "dewarp tta gap $gap")
        val merged = Array(2) { Array(GH) { FloatArray(GW) } }
        for (i in 0 until GH) for (j in 0 until GW) {
            merged[0][i][j] = if (gap <= TTA_GAP) (orig.first[i][j] + mx[i][j]) / 2f else orig.first[i][j]
            merged[1][i][j] = if (gap <= TTA_GAP) (orig.second[i][j] + my[i][j]) / 2f else orig.second[i][j]
        }
        return merged
    }

    private external fun nativeDewarpRemap(bitmap: Bitmap, grid: FloatArray): Boolean
}
