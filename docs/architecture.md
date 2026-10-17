# AgriGuard AI architecture

## Phase 1 scope

The application is planned around local, classical image classification. The initial ONNX exercise uses synthetic numeric data only; it does not include PlantVillage, plant images, or disease names. No diagnosis, recommendation, or farmer-facing workflow is implemented in this phase.

## Where components run

| Location | Planned responsibility |
| --- | --- |
| Farmer's phone/browser | Select and preprocess a leaf image, calculate HOG features in TypeScript, and run the exported SVM through ONNX Runtime Web. The browser can make a local prediction without a third-party AI key. |
| Vercel | Host the Next.js website and, later, optional server routes. Server routes keep provider credentials private and can retrieve non-public treatment records. |
| Local development/training machine | Inspect and split PlantVillage, implement the Python image pipeline, train/evaluate the HOG + SVM pipeline, test conversion, and create versioned model metadata. |

The Phase 1 test harness is developer-only. It accepts an ONNX file and numeric test-vector JSON chosen from the local filesystem; it does not package a model into a public asset or show plant or disease content.

## Phase 3 model baseline

The first real classifier is a Pillow resize/grayscale stage, scikit-image HOG feature extractor, `StandardScaler`, and class-balanced RBF `SVC`. The selected six-class PlantVillage subset is split using upstream leaf-group IDs where available plus exact-file hashes. Python and ONNX Runtime predictions matched on the held-out test feature vectors. Model files and evaluation reports are stored under the Git-ignored `ml/models/phase3/`. The baseline is an experiment, not a validated plant diagnostic product; field-image performance is unknown.

## Primary ML and browser inference

The intended primary model is a scikit-learn SVM trained on handcrafted Histogram of Oriented Gradients (HOG) features. Python will initially use scikit-image for HOG extraction and Pillow for image loading. The fitted model pipeline can be converted with `skl2onnx`; ONNX Runtime will compare Python and ONNX outputs. The temporary feasibility test verifies only the SVM and numeric conversion path.

For real browser inference, TypeScript will perform image validation, deterministic resizing/grayscale conversion, and HOG extraction before passing a feature vector to the ONNX SVM. The TypeScript HOG output must match Python's feature count, ordering, scaling, and values. Differences in resize interpolation, color conversion, gradient boundaries, cell/block layout, orientation bins, or normalization can change the predicted class. Those settings must therefore be recorded with the model and checked on shared test images before browser predictions are trusted.

The ONNX feasibility check can establish whether the chosen scikit-learn SVM pipeline converts and whether ONNX Runtime can evaluate its feature vectors. It cannot prove image preprocessing parity, field-image accuracy, model quality, or suitability for agricultural advice.

## Treatment information

The first treatment knowledge base is planned as structured JSON maintained as reviewed source data, not as a database service. Authoritative records should be stored outside `web/public` and loaded only by server-side code. Retrieval must match crop, disease, and region; treatment records need source attribution and appropriate review. If evidence is absent or conflicting, the application should withhold unsupported treatment details. This JSON approach is adequate for a small curated collection and avoids operating a database before there is a need for one.

## Optional external AI assessment

An optional Next.js server route may call Gemini or OpenAI only when local-model uncertainty crosses a validated threshold and the user-facing workflow explicitly indicates the fallback. Provider credentials stay server-side. The external model's result remains a separate, unconfirmed assessment and must be validated against supported classes and uncertainty rules. It must not silently replace the local result or invent treatment details.

The browser sends an image outside the device only during an explicitly invoked external assessment. The normal local prediction sends no image to an AI provider. A future product must explain this transfer and handle consent, retention, provider terms, request timeouts, and errors before enabling the fallback. Server-side treatment retrieval may transmit query metadata to Vercel, but it should not expose private source records as public static files.

## Deployment

The repository contains separate `web/` and `ml/` workspaces. Configure the Vercel project's Root Directory as `web`; Vercel will install and build the Next.js application from there. Python training scripts are development tools and do not run as a separate backend service. The initial architecture does not require Django, FastAPI, Streamlit, authentication, or a database server.

## Risks, limits, and unknowns

- The first HOG + SVM baseline has been measured on a small, group-aware PlantVillage subset; field-image performance remains unknown until measured using external field images.
- HOG vectors can be large and client-side work may be slow or memory-intensive on low-end phones. Image dimensions, feature-vector size, model size, and inference time must be measured.
- SVM-to-ONNX support depends on the selected estimator options, converter version, opset, and browser runtime kernels. The synthetic check is intended to reveal compatibility issues early.
- Numerical parity for synthetic vectors does not establish parity between Python and TypeScript image preprocessing.
- Browser support, WebAssembly performance, model download size, offline behavior, and mobile memory limits remain to be measured on target devices.
- Confidence scores need validation and calibration; a raw SVM margin is not automatically a calibrated probability.
- Supported crop/disease classes, geographic scope, authoritative treatment sources, review ownership, and escalation policy remain to be decided.
- External assessment introduces privacy, cost, availability, latency, and provider-policy considerations. It must stay optional and separate from the local model.

## Phase 1 dependencies

- **Next.js, React, TypeScript:** website framework, App Router, and static type checks.
- **onnxruntime-web:** development harness for loading the synthetic ONNX artifact in a browser.
- **NumPy, scikit-learn:** deterministic numeric fixtures, scaling, and SVM training.
- **skl2onnx:** converts the fitted scikit-learn pipeline to ONNX.
- **onnxruntime:** runs the converted model in Python for output comparison.
- **scikit-image, Pillow, joblib:** listed for the planned image-feature/model pipeline; this feasibility test does not use image extraction or serialize Python estimators with joblib.

No packages are installed as part of repository setup. See `ml/requirements.txt` and `web/package.json` for declared dependencies.
