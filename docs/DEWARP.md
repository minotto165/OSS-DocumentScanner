# Neural dewarp (vFlat-style page flattening) — `feat/uvdoc-dewarp`

## What it is

Optional neural page-flattening applied **after** the existing perspective
crop. Pipeline per quad in `cropDocumentSync`:

1. `nativeCrop` (unchanged OpenCV quad → rectangle)
2. **NEW:** `Dewarp.dewarpBitmap` → ONNX Runtime inference + C++ remap
3. Save / return the flattened bitmap

Dewarp is a **quality stage, never a correctness gate**: any failure
(missing model, ORT error, TTA disagreement is only a fallback) returns the
perspective crop unchanged. The JSON result carries `"dewarp": true/false`
per page.

## Model

`App_Resources/documentscanner/Android/src/main/assets/dewarp_flattener.onnx`
(31.6 MB, Apache-2.0, Flattener bundle 1.1.0, uvdocnet architecture):

- input `(1,3,712,488)` RGB/255, bilinear resize without antialias
  (Android `createScaledBitmap(..., FILTER_BILINEAR)` is the equivalent)
- output `(1,2,45,31)` grid, flattener convention
  (+-1 = outer source-pixel edges)
- TTA: original + horizontal mirror, unmirror (reverse cols, negate x),
  average when median node distance / 2 <= 0.03, else plain grid

Desktop reference (same weights, Flattener's own `coords.py` + `ops.py`):
preprocess ~0.02 s, inference ~0.2-0.3 s (CPU), remap ~0.2 s on a Linux
desktop for a 1.4 MP photo. Expect roughly 1-3 s per crop on a modern
phone (CPU, background thread — the live preview stays quad-only).

## Files

| File | Role |
|---|---|
| `plugin-nativeprocessor/platforms/android/java/.../Dewarp.kt` | ORT session (lazy singleton), preprocess, TTA, calls native remap |
| `plugin-nativeprocessor/platforms/android/cpp/dewarp.cpp` | `grid_to_map` + `grid_sample` (border) + round quantize + white fill, 128-row blocks (~3 MB peak). Auto-built via `aux_source_directory` |
| `.../CustomImageAnalysisCallback.kt` | `dewarp` flag in `cropDocumentFromFile` options JSON → `cropDocumentSync(..., context, dewarp)` |
| `plugin-nativeprocessor/index.d.ts` | `CropOptions.dewarp?: boolean` (`cropDocumentFromFile` already forwards options as JSON) |
| `App_Resources/documentscanner/Android/app.gradle` | `onnxruntime-android:1.20.0` dependency |

Usage from JS:

```ts
await cropDocumentFromFile(src, quads, { saveInFolder, dewarp: true });
```

## Limits (v1)

- CPU only (no NNAPI/GPU delegate yet — deterministic first).
- No page re-crop / rerun passes (Flattener's full pipeline adds those for
  heavy crumple; here the OSS quad detector is the locate stage).
- Landscape spreads run as-is (model is portrait-trained; accuracy lower).
- `cropDocument` (live preview path) has no `dewarp` param — file path only.

## Verification done

- `dewarp.cpp` core: desktop parity test (`/tmp/dewtest`, stubs only —
  not committed): identity grid → MSE 0.0; top-row grid → exact;
  bad grid length → false. **ALL PASS.**
- Math mirrors `flattener/ops.py::grid_sample` (align_corners=False,
  border) + `coords.py::grid_to_map` exactly (Python identity MSE 0.0).
- C++ syntax: `g++ -fsyntax-only -std=c++20` clean.
- Model runs: ORT 1.31 CPU, sane grids, flip-consistent TTA gaps.
- Real UVDoc sample (magazine page): single pass straightened outer
  margins into a rectangle; sharp center crease partially remains
  (expected without rerun passes).

## Still to do

- Full APK build (`ns` toolchain) + on-device test with real bent-page
  photos (the decisive test — synthetic warps are out-of-distribution
  for this model, see investigation notes).
- Consider NNAPI + FP16, rerun pass, orientation handling.
