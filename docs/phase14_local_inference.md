# Phase 14: Verified Local Phase 9 Inference

## Status

Local inference is verified for the implemented browser path on six representative, non-test PlantVillage photos: one image from each saved class mapping. Chrome decoded each original image to the exact RGB bytes stored by the Python reference fixture. All six browser ONNX predictions matched the saved Python Phase 9 predictions. This is an implementation-parity check, not an accuracy evaluation.

The browser test surfaced and fixed one real issue: ONNX Runtime Web selected its JSEP WASM loader, but the local runtime-asset route initially allowed only the base WASM pair. The route returned 404 before model inference. The JSEP module and WASM file are now explicitly allowlisted.

## Frozen artifact and ONNX interface

- Model: ml/models/phase9_domain_adaptation/phase9_exp_20261019_v1/selected_model.onnx
- SHA-256: 1d741fbe7ee6d9a81f0bba78e7a2e84e63f4636000bda7ff8bd3223ffa28f212
- Input: X, float32 tensor, shape [batch, 8132].
- Outputs: label, int64 [batch]; probabilities, float32 [batch, 6].
- The second output contains raw SVC decision scores despite the ONNX output name. The UI does not display them or call them probabilities/confidence.
- Saved class order: 0 Potato early blight; 1 Potato late blight; 2 Potato healthy; 3 Tomato early blight; 4 Tomato late blight; 5 Tomato healthy.
- Python control check: ONNX labels matched Python for 8/8 fixture vectors; maximum absolute difference in the raw score output was 1.812e-5. Those scores are not probabilities.

## Frozen feature/preprocessing contract

The browser supplies interleaved RGB uint8 source pixels. The Python reference used Pillow image decoding without EXIF orientation, then:

1. HOG branch: Pillow RGB-to-L uint8 grayscale conversion; resize to 128×128 using Pillow bilinear; scikit-image HOG, 9 orientations, 8×8 pixels/cell, 2×2 cells/block, L2-Hys normalization, no square-root transform. The result has 8,100 float32 values.
2. HSV branch: Pillow RGB resize to 128×128 bilinear; RGB converted to float32 [0,1]; scikit-image-compatible RGB-to-HSV; H, S and V histograms use 16, 8 and 8 bins respectively over [0,1], each channel divided by the number of pixels. The result has 32 float32 values.
3. Concatenate HOG first, HSV second: 8,132 float32 values. The model's saved StandardScaler is inside the ONNX graph; the browser sends unscaled features.

Automated tests of the TypeScript feature extractor used by the browser worker compare against Python-generated vectors with per-value tolerance 1e-6 + 1e-5 × |reference|. Across eight fixtures (two synthetic patterns and six real PlantVillage images; the headless Chrome run separately checked exact decoded source pixels and browser predictions), HOG max/mean absolute differences were 5.9605e-8 / 2.8907e-9; HSV differences were 0 / 0; combined-vector differences were 5.9605e-8 / 2.8793e-9. No values exceeded tolerance. HSV matched exactly on these fixtures; this does not establish exactness for every possible RGB value or browser/runtime.

Relative error is not a useful stand-alone parity measure near zero-valued HOG bins, so the check uses absolute plus reference-scaled tolerance and reports maximum and mean absolute differences.

## Browser behavior and safety

- Tomato and potato selections can use the Phase 9 model locally. A Web Worker extracts features so HOG/HSV loops do not run on the main UI thread.
- The model is fetched lazily from a same-origin route at /api/models/phase9; only the ONNX artifact is served. ONNX Runtime's two JSEP assets and two base WASM assets are served from an allowlist. No training data or Python code is bundled.
- The image remains in browser memory for local inference. A failed model load or prediction shows an error and does not switch to Gemini.
- Gemini is a separately labelled option. It runs only when the user selects it, explicitly consents, and submits; this path sends the image to the server assessment endpoint. No live Gemini call was made.
- No treatment endpoint is connected.
- The model always chooses among six known classes. There is no validated unknown-class rejection policy. Outputs are possible assessments, not diagnoses; unfamiliar crops/diseases may be misclassified.
- The ONNX file is about 310 MB and is fetched on first local use, so initial loading can take time and memory.

## Prediction parity samples

The headless Chrome check used original PlantVillage JPEGs from existing Phase 6.6 train/validation manifests, never the test split. Five fixtures were from validation; Potato healthy was from training because that class has no validation sample. Browser-decoded pixels matched corresponding Python fixture bytes exactly, and browser results matched the saved Python model predictions:

| Source image class | Python model result | Browser model result |
|---|---|---|
| Potato early blight | Potato early blight | Potato early blight |
| Potato late blight | Potato late blight | Potato late blight |
| Potato healthy | Potato healthy | Potato healthy |
| Tomato early blight | Tomato early blight | Tomato early blight |
| Tomato late blight | Tomato early blight | Tomato early blight |
| Tomato healthy | Tomato healthy | Tomato healthy |

The Tomato late blight fixture is a model error relative to its dataset label, but the browser correctly reproduced the frozen Python model's prediction. This phase verifies pipeline parity, not model correctness or disease accuracy.

The browser test confirmed /api/models/phase9 was requested and /api/assess-plant was not requested during local inference.

## Tests and commands

Run from web/:

- npm run test:hog — PASS, 3 tests. HOG reference max absolute difference 1.1921e-7; grayscale resize matched all 196,608 pixel values in its held-out reference fixture.
- npm run test:phase-c — PASS, 30 tests across Phase A, B and C treatment/API/UI coverage.
- npm run test:phase14 — PASS, 6 tests; regenerates ignored Python reference fixtures, verifies feature dimensions/metrics, inference mapping, model-load failure handling, invalid output shape, and supported image MIME handling.
- node tests/phase14-browser-check.mjs — PASS, six Chrome headless browser inferences matched Python model outputs, exact decoded fixture pixels, no image assessment API request.
- npm run typecheck — PASS.
- npm run build — PASS; Next.js production build includes the Phase 9 model and runtime asset routes.

Feature fixture generation used the project interpreter and existing Phase 9 model/features. No model was retrained and no test images were read.

## Phase D files

- web/app/real-inference/phase9-features.ts, pillow-resize.ts, phase9-feature.worker.ts, phase9-inference.ts, image.ts, metadata.ts: browser preprocessing, worker extraction, ONNX loading/inference, image decoding and class mapping.
- web/app/api/models/phase9/route.ts, web/app/api/runtime/ort/[asset]/route.ts: serve only the frozen Phase 9 model and allowlisted ONNX Runtime files. The runtime route includes JSEP assets required by the installed runtime.
- web/app/assessment/assessment-form.tsx, web/app/page.tsx, web/app/globals.css: explicit local vs consented external routing and cautious result display.
- web/tests/phase9-features.test.mjs, phase9-inference.test.mjs, phase14-browser-check.mjs, web/package.json, web/package-lock.json: parity, error-path, browser tests and test tooling.
- ml/scripts/generate_phase14_feature_fixtures.py: creates local ignored reference fixtures; generated data lives under ml/.dev_artifacts/phase14/.

Phase A/B/C files and other pre-existing uncommitted work were preserved. The repository already had many untracked and modified files before this phase; nothing was staged or committed.

## Remaining limitations

Six representative headless browser samples and eight TypeScript feature fixtures establish implementation parity on those inputs, not universal bitwise parity or generalization. PlantVillage's controlled backgrounds differ from farm photographs. The external test set was not accessed. The Phase 9 model has no validated uncertainty/rejection behavior; the UI must not present an unfamiliar input as reliably diagnosed. First-time model loading requires roughly 310 MB of model data.
