# Phase 4: real-image browser inference

## Phase 3 input contract

The saved `ml/models/phase3/metadata.json` defines the input: Pillow reads the image, ignores EXIF orientation, converts to grayscale with `convert("L")`, resizes to 128 x 128 with Pillow bilinear interpolation and returns uint8 pixels (0-255). scikit-image HOG uses 9 unsigned orientation bins, 8 x 8 pixels per cell, 2 x 2 cells per block, L2-Hys block normalization, no square-root image transform, and C-order flattening. There are 15 x 15 blocks x 2 x 2 cells x 9 bins = 8,100 float32 values.

The browser route uses Canvas to decode and resize, then extracts grayscale bytes and HOG. Browser Canvas resize is implementation-dependent and is not Pillow's specified resampler. Therefore HOG-on-reference-pixels can be directly compared, while end-to-end grayscale/preprocessing parity needs a browser run on the generated held-out fixtures. The optional fixture JSON input reports max and mean absolute and relative differences for pixels, HOG, and SVM scores. Relative errors use small denominator floors documented in the page. Those are diagnostics, not an assumed pass based on matching parameter names.

## ONNX contract and labels

The Phase 3 ONNX graph input is `X: tensor(float)[N,8100]`. Output `label` is `tensor(int64)[N]`. The other output is named `probabilities` in the ONNX graph but is not a probability output: the training SVC sets `probability=False`. Across 12 held-out vectors, the six values agreed with scikit-learn `decision_function` to a maximum absolute difference of 6.84e-6 and mean absolute difference of 1.29e-6. The route labels the largest value as an uncalibrated SVM decision value and shows the top-two gap, never a probability or confidence percentage.

The saved mapping is `0 Potato___Early_blight`, `1 Potato___Late_blight`, `2 Potato___healthy`, `3 Tomato___Early_blight`, `4 Tomato___Late_blight`, `5 Tomato___healthy`. It is copied into `web/app/real-inference/metadata.ts` and checked against model outputs at runtime.

## Developer check

Start the app from `web` using `npm run dev`, then open `/real-inference`. Select `ml/models/phase3/hog_svm.onnx` and an image from a supported browser-decoded format (JPEG, PNG, BMP, or WebP). The file is used only in the current browser tab. The app rejects images over 20 MiB or 8192 pixels on either side. WASM/CPU execution is selected explicitly.

To create reference data from two examples of every class in the held-out test split:

```powershell
cd D:\PlantDiseaseAI
.\.venv\Scripts\python.exe ml\scripts\generate_browser_fixtures.py
```

Fixtures are written to ignored `ml/.dev_artifacts/phase4/browser-fixtures/`, include copied source images, Python grayscale bytes, HOG vectors, labels, saved-model predictions, and decision values. They are for parity/debugging only; saved predictions are not new accuracy measurements. Run the TypeScript HOG comparison with `cd web; npm run test:hog` after generating them. That automated check feeds Python's grayscale bytes to browser HOG. It does not verify Canvas decoding/resizing.

For an end-to-end browser comparison, start the site, open `/real-inference`, choose the Phase 3 ONNX file, choose one fixture image from `images/`, and choose `fixtures.json`. The page matches the image by filename and prints measured pixel, HOG, and SVM decision-value max/mean absolute and relative errors alongside the browser and saved Python predictions. Repeat for all 12 fixture images (two in each of six classes). Since Canvas and Pillow resize implementations differ, record these numbers; the app deliberately does not declare preprocessing parity automatically.

## Interpretation limits

The classifier was trained on PlantVillage images with controlled backgrounds. Real farm photographs, lighting, devices, other diseases, and mixed symptoms are not validated here. A class output is not a verified diagnosis. SVM decision values and their gap are not calibrated probabilities or uncertainty estimates.
