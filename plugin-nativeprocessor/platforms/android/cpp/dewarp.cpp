// Neural dewarp remap (vFlat-style page flattening), JNI for Dewarp.kt.
//
// Exact port of Flattener's NumPy code:
//   - flattener/dewarp/coords.py :: grid_to_map (edge-aligned grid 45x31 ->
//     dense center-aligned map, extrapolate=false)
//   - flattener/ops.py :: grid_sample (bilinear, border padding,
//     align_corners=False) + quantize "round"
// Pixels outside the source (|v| > 1 or non-finite) are filled white,
// matching flattener's render_unchecked with fill_rgb=white.
//
// The math is separable (rows @ grid @ cols^T) and runs in 128-row blocks,
// so peak extra memory is ~3 MB even for 12 MP crops.

#include <jni.h>

#include <android/bitmap.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kGh = 45;
constexpr int kGw = 31;
constexpr int kBlock = 128;

// (2*i+1)/n - 1 for i in [0, n): flattener center positions.
inline float center_pos(int i, int n) {
    return (2.0f * i + 1.0f) / n - 1.0f;
}

// One row of 1-D linear weights at fractional node index t in [0, n-1],
// end value held outside (extrapolate=false). Mirrors interpolation_matrix.
inline void interp_weights(float t, int n, int &left, float &w) {
    float tc = t < 0 ? 0 : (t > n - 1 ? n - 1 : t);
    int l = (int)floorf(tc);
    if (l < 0) l = 0;
    if (l > n - 2) l = n - 2;
    left = l;
    w = tc - l;
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_akylas_documentscanner_Dewarp_nativeDewarpRemap(JNIEnv *env, jobject /*thiz*/, jobject bitmap,
                                                        jfloatArray grid_) {
    if (bitmap == nullptr || grid_ == nullptr) return JNI_FALSE;
    if (env->GetArrayLength(grid_) != 2 * kGh * kGw) return JNI_FALSE;

    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return JNI_FALSE;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || info.width < 2 || info.height < 2)
        return JNI_FALSE;
    const int W = info.width;
    const int H = info.height;

    jfloat *gridVals = env->GetFloatArrayElements(grid_, nullptr);
    if (gridVals == nullptr) return JNI_FALSE;
    // Copy grid to (2, Gh, Gw) doubles for the two matmuls (float64 like NumPy).
    double grid[2][kGh][kGw];
    for (int c = 0; c < 2; c++)
        for (int i = 0; i < kGh; i++)
            for (int j = 0; j < kGw; j++) grid[c][i][j] = gridVals[c * kGh * kGw + i * kGw + j];
    env->ReleaseFloatArrayElements(grid_, gridVals, JNI_ABORT);

    // cols: W x Gw — (center_positions(W)+1)/2 * (Gw-1).
    std::vector<int> colL(W);
    std::vector<float> colW(W);
    for (int w = 0; w < W; w++) {
        float t = (center_pos(w, W) + 1.0f) / 2.0f * (kGw - 1);
        interp_weights(t, kGw, colL[w], colW[w]);
    }
    // tmp[c][i][w] = sum_j grid[c][i][j] * cols[w][j].
    std::vector<double> tmp(size_t(2) * kGh * W);
    auto T = [&](int c, int i, int w) -> double & { return tmp[(size_t(c) * kGh + i) * W + w]; };
    for (int c = 0; c < 2; c++) {
        for (int i = 0; i < kGh; i++) {
            for (int w = 0; w < W; w++) {
                int l = colL[w];
                float wt = colW[w];
                T(c, i, w) = grid[c][i][l] * (1.0 - wt) + grid[c][i][l + 1] * wt;
            }
        }
    }
    // rows matrix per block: index = (center_positions(h)+1)/2 * (Gh-1).
    std::vector<int> rowL(kBlock);
    std::vector<float> rowW(kBlock);

    void *pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return JNI_FALSE;
    // Snapshot source (in-place rewrite needs the original for sampling).
    const size_t n = size_t(H) * W;
    std::vector<uint8_t> srcR(n), srcG(n), srcB(n);
    {
        uint8_t *p = static_cast<uint8_t *>(pixels);
        const size_t stride = info.stride;
        for (int y = 0; y < H; y++) {
            const uint32_t *row = reinterpret_cast<const uint32_t *>(p + size_t(y) * stride);
            for (int x = 0; x < W; x++) {
                uint32_t v = row[x];
                size_t k = size_t(y) * W + x;
                srcR[k] = v & 0xFF;
                srcG[k] = (v >> 8) & 0xFF;
                srcB[k] = (v >> 16) & 0xFF;
            }
        }
    }

    auto sample = [&](const std::vector<uint8_t> &ch, float fx, float fy) -> uint8_t {
        // align_corners=False: x = (gx+1)*W/2 - 0.5, border clamp.
        float x = fx;
        float y = fy;
        x = x < 0 ? 0 : (x > W - 1 ? W - 1 : x);
        y = y < 0 ? 0 : (y > H - 1 ? H - 1 : y);
        int x0 = (int)floorf(x);
        int y0 = (int)floorf(y);
        int x1 = x0 + 1 < W ? x0 + 1 : W - 1;
        int y1 = y0 + 1 < H ? y0 + 1 : H - 1;
        float ex = x - x0;
        float ey = y - y0;
        float v = ch[size_t(y0) * W + x0] * (1 - ex) * (1 - ey) +
                  ch[size_t(y0) * W + x1] * ex * (1 - ey) +
                  ch[size_t(y1) * W + x0] * (1 - ex) * ey + ch[size_t(y1) * W + x1] * ex * ey;
        // quantize round: floor(clip01(v/255)*255 + 0.5).
        float s = v / 255.0f;
        s = s < 0 ? 0 : (s > 1 ? 1 : s);
        return (uint8_t)floorf(s * 255.0f + 0.5f);
    };

    std::vector<double> mapx(kBlock * W), mapy(kBlock * W);
    bool ok = true;
    for (int y0 = 0; y0 < H && ok; y0 += kBlock) {
        int bh = std::min(kBlock, H - y0);
        for (int r = 0; r < bh; r++) {
            float t = (center_pos(y0 + r, H) + 1.0f) / 2.0f * (kGh - 1);
            interp_weights(t, kGh, rowL[r], rowW[r]);
        }
        // map = rows_block @ tmp per channel.
        for (int c = 0; c < 2; c++) {
            for (int r = 0; r < bh; r++) {
                int l = rowL[r];
                float wt = rowW[r];
                for (int w = 0; w < W; w++) {
                    double v = T(c, l, w) * (1.0 - wt) + T(c, l + 1, w) * wt;
                    (c == 0 ? mapx : mapy)[size_t(r) * W + w] = v;
                }
            }
        }
        uint8_t *p = static_cast<uint8_t *>(pixels);
        for (int r = 0; r < bh; r++) {
            uint32_t *row = reinterpret_cast<uint32_t *>(p + size_t(y0 + r) * info.stride);
            for (int w = 0; w < W; w++) {
                size_t k = size_t(r) * W + w;
                double gx = mapx[k];
                double gy = mapy[k];
                bool valid = std::isfinite(gx) && std::isfinite(gy) && fabs(gx) <= 1.0 && fabs(gy) <= 1.0;
                uint32_t out;
                if (!valid) {
                    out = 0xFFFFFFFFu;  // white fill
                } else {
                    // align_corners=False pixel coords.
                    float fx = float((gx + 1.0) * W / 2.0 - 0.5);
                    float fy = float((gy + 1.0) * H / 2.0 - 0.5);
                    uint8_t rr = sample(srcR, fx, fy);
                    uint8_t gg = sample(srcG, fx, fy);
                    uint8_t bb = sample(srcB, fx, fy);
                    out = 0xFF000000u | (uint32_t(bb) << 16) | (uint32_t(gg) << 8) | rr;
                }
                row[w] = out;
            }
        }
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    return ok ? JNI_TRUE : JNI_FALSE;
}
