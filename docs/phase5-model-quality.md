# Phase 5: traditional ML quality experiments

## Data and baseline preservation

Experiments used the existing 1,860 train, 398 validation, and 394 test images. The CSV manifests retain the Phase 3 hashes: train `28c86858f9b350b8c009b6b03444e27fada5510618320d4a897d4ebf627bd36f`, validation `b9dab9aa02df04baa321360370a47eb9f9eef9124227656e5cbe0afbd7dedf00`, test `548cd235a35928ba35c56eb09f45497657b5159e89a3e1246f4ce0087b18daa2`. The runner checked each image path and content hash and checked that paths do not overlap across splits. Class IDs remain 0 Potato early blight, 1 Potato late blight, 2 Potato healthy, 3 Tomato early blight, 4 Tomato late blight, 5 Tomato healthy.

The Phase 3 `.joblib`, `.onnx`, metadata, metrics, parity report, and Markdown report were checksummed before experimentation and verified unchanged afterward. Its preprocessing and feature configuration were not modified.

Train counts show the imbalance: Potato healthy has 104 images, versus 350-352 in each other class. Validation and test each have 24 Potato healthy images, versus 72-76 per other class. The baseline handles that class poorly despite balanced class weights: its validation recall is 5/24 (0.2083) and test recall 3/24 (0.1250).

## Baseline error analysis

On the baseline test split, Potato healthy is predicted as Potato late blight 16 times and Potato early blight 4 times; only 3 of 24 are correct. Tomato early blight is confused with Tomato late blight 25 times (37/72 correct). Potato early and late blight are confused with each other in both directions (12 early-to-late and 7 late-to-early). Some cross-crop errors also occur, although the largest baseline confusions are within the same crop. These counts describe this split only.

## Validation candidates

All models use the same training and validation rows, the same Phase 3 HOG features where applicable, and a `StandardScaler` fitted only on train data. All SVC candidates use an RBF kernel, `gamma="scale"`, and seed 2026. Candidate selection used validation macro F1 and per-class results only. No test metrics were read during that selection.

| Candidate | Feature set / change | Accuracy | Macro precision | Macro recall | Macro F1 | Potato healthy recall | SVC fit (s) | Validation predict (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_hog_C1_balanced` | HOG, C=1, balanced | 0.7714 | 0.8189 | 0.7079 | 0.7124 | 0.2083 | 14.75 | 3.17 |
| `hog_hsv_C1_balanced` | HOG + HSV histogram, C=1, balanced | **0.7940** | **0.8364** | **0.7279** | **0.7321** | 0.2083 | 14.74 | 3.01 |
| `hog_C2_balanced` | HOG, C=2, balanced | 0.7889 | 0.8325 | 0.7234 | 0.7280 | 0.2083 | 14.61 | 2.77 |
| `hog_C4_balanced` | HOG, C=4, balanced | 0.7889 | 0.8340 | 0.7234 | 0.7282 | 0.2083 | 15.60 | 3.87 |
| `hog_C1_unweighted` | HOG, C=1, no class weights | 0.7688 | 0.8202 | 0.6915 | 0.6830 | 0.0833 | 14.11 | 2.93 |
| `hog_C1_healthy_2x` | HOG, C=1, balanced weights with Potato healthy doubled | 0.7714 | 0.8189 | 0.7079 | 0.7124 | 0.2083 | 13.87 | 3.20 |

Train and validation features were extracted once and cached; all candidates share that cost: HOG 7.91s, HSV histograms 6.40s, total 14.31s. The HSV branch adds 32 values: 16 H, 8 S, 8 V bins from a Pillow RGB bilinear 128 x 128 image, with each channel histogram divided by pixel count. HOG remains 8,100 values, so the combined vector is 8,132 float32 values. The color candidate raised validation macro F1 by 0.0197 and improved validation precision, recall, and F1 for Potato early, Potato late, Tomato early, Tomato late, and Tomato healthy; Potato healthy stayed unchanged. This is evidence that color adds useful information for this dataset split, but not proof of field generalization.

Full precision/recall/F1 and confusion matrices for every validation candidate are preserved in `ml/.dev_artifacts/phase5/run1/validation_results.json` and copied into the selected model directory as `candidate_validation_results.json`.

## Frozen candidate and held-out test

`hog_hsv_C1_balanced` had the best validation macro F1 and was frozen before final test feature extraction. The held-out result is:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Phase 3 HOG baseline | 0.7538 | 0.7700 | 0.6811 | 0.6757 |
| Phase 5 HOG + HSV | **0.7766** | **0.7899** | **0.7019** | **0.6968** |

That is +2.28 percentage points accuracy and +2.11 points macro F1 on this test split. Per-class test results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Potato early blight | 0.8378 | 0.8611 | 0.8493 | 72 |
| Potato late blight | 0.6628 | 0.7500 | 0.7037 | 76 |
| Potato healthy | 0.7500 | 0.1250 | 0.2143 | 24 |
| Tomato early blight | 0.8864 | 0.5417 | 0.6724 | 72 |
| Tomato late blight | 0.6667 | 0.9600 | 0.7869 | 75 |
| Tomato healthy | 0.9359 | 0.9733 | 0.9542 | 75 |

Rows below are actual classes; columns are predicted classes, ordered as the class list above.

| Actual \\ Predicted | Potato early | Potato late | Potato healthy | Tomato early | Tomato late | Tomato healthy |
|---|---:|---:|---:|---:|---:|---:|
| Potato early | 62 | 7 | 1 | 1 | 1 | 0 |
| Potato late | 6 | 57 | 0 | 2 | 9 | 2 |
| Potato healthy | 3 | 18 | 3 | 0 | 0 | 0 |
| Tomato early | 3 | 3 | 0 | 39 | 25 | 2 |
| Tomato late | 0 | 1 | 0 | 1 | 72 | 1 |
| Tomato healthy | 0 | 0 | 0 | 1 | 1 | 73 |

Compared with baseline, Potato early recall rose from 0.7778 to 0.8611, Potato late from 0.7368 to 0.7500, Tomato early from 0.5139 to 0.5417, and Tomato healthy stayed 0.9733. Potato healthy recall remains 0.1250 (3/24), and Tomato late remains 0.9600. Eighteen Potato healthy images are still classified as Potato late blight. The change improves aggregate scores without solving the most important class-specific failure.

## Artifacts and ONNX

New model artifacts are under `ml/models/phase5/hog_svm_candidate_v1/`: `model.joblib`, `model.onnx`, `metadata.json`, `metrics.json`, `report.md`, and the full `candidate_validation_results.json`. The artifact embeds the HOG + HSV feature specification, model parameters, class mapping, split hashes, and validation/test metrics. Phase 3 remains intact and remains the browser model.

ONNX CPU inference matched Python class IDs on all 394 real held-out vectors (0 mismatches). Maximum absolute decision-score difference was `1.60e-5`. The graph input is `X: float32[N,8132]`. The test vector contains 8,100 HOG values followed by 32 HSV histogram values.

The selected pipeline is not yet browser-compatible. A TypeScript implementation of the 32 HSV histogram values must be added and parity-tested before the Phase 5 ONNX artifact can be used by the browser; the current TypeScript browser HOG route continues to use the Phase 3 8,100-feature model.

## Verification and limits

`python -m unittest discover -s tests -v`: 17 tests passed. `python -m pytest -q` was attempted but pytest is not installed in the project venv; no package was installed for this phase. The experiment runner itself verified manifest content hashes and disjoint paths, baseline checksums, feature dimensions, and selected-candidate ONNX parity.

All results come from a single fixed PlantVillage split with controlled backgrounds. They do not establish performance on farm photographs, different devices or lighting, mixed infections, or unseen cultivars. The selected candidate is a measured improvement on this split, not a guarantee of generalization.
