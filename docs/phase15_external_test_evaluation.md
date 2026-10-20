# Phase 15 — Frozen External Test Evaluation

**Status:** Complete. The reserved external test split was verified against the frozen Phase 9 candidate and evaluated once on 2026-10-19. This report records final test results; the test set must not guide further tuning.

## Protocol audit and freeze

The evaluated model is the Phase 9 selected candidate `hog_hsv_svm` from `ml/models/phase9_domain_adaptation/phase9_exp_20261019_v1/`. Its ONNX SHA-256 is `1d741fbe7ee6d9a81f0bba78e7a2e84e63f4636000bda7ff8bd3223ffa28f212`, matching the requested hash. The joblib pipeline used by the established evaluator has SHA-256 `b63b714ad460c73b3b9b2fddf485101d32ef84be5dbe90b8b954dd3f57eb12c6`.

The reserved manifest is `ml/data/splits/phase8_external/phase8_external_20261019_v2/test.csv`, SHA-256 `db422c68b20205a35581b7c93a8c65876b591838b9ef2690699fd3c155a096ec`. Its 8,636 paths and labels match the pre-evaluation freeze record. The Phase 8 summary and Phase 10 freeze record contain the same manifest hash. The Phase 8 record reports `fresh_test_evaluated: false`; the Phase 9 and Phase 10 records state the test was not accessed for Phase 9 model selection. The freeze record also states test images were not opened and test metrics were not used before freeze. No previous Phase 10 evaluation marker or predictions existed before the single run.

The split is disjoint from external train, external validation, and Phase 9 selected external training by known component and exact-hash checks (zero overlap reported for each). The test set contains three potato labels only. Phase 8 used exact hashes and perceptual-hash candidate edges to form components; the source has no reliable plant, leaf, or capture identifiers. Undetected related captures cannot be ruled out.

Class order used by the model and confusion matrix:

1. `Potato___Early_blight`
2. `Potato___Late_blight`
3. `Potato___healthy`
4. `Tomato___Early_blight`
5. `Tomato___Late_blight`
6. `Tomato___healthy`

The model emits SVC decision scores. They are not calibrated probabilities and are not interpreted as diagnostic confidence.

## Evaluation procedure and data accounting

The existing guarded evaluator was run once:

```powershell
Set-Location D:\PlantDiseaseAI
.\.venv\Scripts\python.exe ml\scripts\run_phase10_external_test.py --evaluate-once
```

It rechecked the frozen model and manifest hashes, file hashes, labels, and image readability before extraction. It evaluated one representative per exact SHA-256 image: 5,767 unique images from 8,636 manifest paths. The other 2,869 paths were omitted from fitting and metric calculation as exact-content duplicates; no files were removed. Unique-hash supports were 932 early blight, 1,943 late blight, and 2,892 healthy. The manifest path counts were 2,530, 3,043, and 3,063 respectively.

No test path was excluded for corruption or unreadability during this run. The upstream Phase 8 audit had already excluded four unreadable files before manifest creation: `earlyblt/earlyblt838.jpg` and `earlyblt/earlyblt862.jpg` (unidentified image files), `healthy/healthy11725.jpg` (truncated at 16 bytes), and `healthy/healthy4766.jpg` (truncated at 40 bytes). None is in the reserved test manifest. The Phase 8 audit also quarantined exact-hash label conflicts and mixed-label perceptual-hash components according to its documented policy.

The evaluator completed at `2026-10-19T13:58:47.191987+00:00`; its start marker is `2026-10-19T13:49:29.137571+00:00`. Feature extraction took 154.07 seconds and prediction took 362.12 seconds. The immutable run record is `ml/models/phase10_external_test/evaluation_complete.json` (`test_evaluation_count: 1`). Raw Phase 10 outputs remain preserved.

## Results

Metrics below are computed on the 5,767 distinct exact-hash representatives. The primary macro metrics average the three potato classes with test support. Balanced accuracy equals macro recall here.

| Metric | Result |
|---|---:|
| Accuracy | 77.61% (4,476 / 5,767) |
| Balanced accuracy | 71.80% |
| Macro precision (three supported potato classes) | 72.79% |
| Macro recall (three supported potato classes) | 71.80% |
| Macro F1 (three supported potato classes) | 72.01% |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Potato early blight | 53.62% | 57.94% | 55.70% | 932 |
| Potato late blight | 78.87% | 65.11% | 71.33% | 1,943 |
| Potato healthy | 85.88% | 92.36% | 89.00% | 2,892 |
| Tomato early blight | 0.00%* | 0.00%* | 0.00%* | 0 |
| Tomato late blight | 0.00%* | 0.00%* | 0.00%* | 0 |
| Tomato healthy | 0.00%* | 0.00%* | 0.00%* | 0 |

`*` Tomato classes have no true examples in this external potato test set; these zero-division values are placeholders, not measured tomato performance. Six-label macro values with zero-division set to zero are precision 36.40%, recall 35.90%, and F1 36.01%, but the three-class supported metrics are the meaningful summary for this test.

Confusion matrix (true labels are rows; predicted labels are columns, in the six-class order above):

| True \\ Predicted | Potato early | Potato late | Potato healthy | Tomato early | Tomato late | Tomato healthy |
|---|---:|---:|---:|---:|---:|---:|
| Potato early blight | 540 | 192 | 194 | 1 | 5 | 0 |
| Potato late blight | 401 | 1,265 | 245 | 14 | 17 | 1 |
| Potato healthy | 66 | 147 | 2,671 | 3 | 4 | 1 |
| Tomato early blight | 0 | 0 | 0 | 0 | 0 | 0 |
| Tomato late blight | 0 | 0 | 0 | 0 | 0 | 0 |
| Tomato healthy | 0 | 0 | 0 | 0 | 0 | 0 |

Error counts:

- Healthy potato predicted as a disease class: **220** (66 early blight, 147 late blight, 3 tomato early blight, 4 tomato late blight).
- Healthy potato predicted as tomato healthy: **1**; this is a wrong class prediction, but not a disease prediction.
- Diseased potato predicted as healthy potato: **439** (194 early blight and 245 late blight).
- Potato images predicted as tomato disease classes: **44**; as any tomato class: **46**.

The Phase 10 raw `metrics.json` reports 221 healthy potatoes as diseased because its count included the one image predicted as `Tomato___healthy`. The confusion matrix and prediction records show the correct disease-only count is 220. The original Phase 10 artifact was preserved unchanged; this report documents the corrected interpretation.

### Confidence intervals

Supplementary intervals were calculated from the saved prediction records only (no images or model were reopened). The method was a stratified cluster bootstrap over the 4,856 known connected components, sampling components with replacement within each supported potato class and retaining all unique-hash representatives in a selected component. It used 2,000 replicates, NumPy `default_rng` seed 20261019, and percentile 95% intervals:

| Metric | 95% interval |
|---|---:|
| Accuracy | 76.31%–79.01% |
| Balanced accuracy / macro recall | 69.90%–73.60% |
| Macro precision | 70.99%–74.46% |
| Macro F1 | 70.23%–73.66% |

These intervals account for known grouping edges only. Missing plant/leaf/capture IDs, undetected near-duplicates, label noise, and dataset sampling bias are not resolved by a confidence interval. Machine-readable supplementary results are in `ml/models/phase15_external_test/supplementary_metrics.json`.

## Interpretation and limits

The external test result (77.61% accuracy; 72.01% supported-class macro F1) is broadly consistent with the previously used external validation result (77.34% accuracy; 71.52% macro F1). The validation set was used during Phase 9 analysis/model selection and is not an independent final test. The Phase 9 PlantVillage test result (79.17% accuracy; 79.54% macro F1) came from a different source and split; it should not be directly compared as if the datasets had equivalent conditions.

The external images were collected in Tanzania, predominantly from Shangi potatoes using Samsung Galaxy A03 Core phones. These results do not establish reliability in Kerala, across other cultivars, cameras, lighting, or field backgrounds. The weak early-blight recall (57.94%) and late-blight recall (65.11%), plus disease-to-healthy errors, matter despite the stronger healthy recall. This remains a preliminary model assessment, not a confirmed diagnosis or production-readiness claim.

This reserved test set has now been consumed for final evaluation. Any future model changes require a new independent test set for an unbiased final assessment. Do not use these test errors to tune this model within Phase 15.

## Verification and artifacts

- The ONNX model hash still matches the requested SHA-256; the scored joblib and test manifest hashes match the freeze record.
- `evaluation_complete.json` records exactly one completed evaluation and 5,767 unique images.
- `ml/models/phase10_external_test/metrics.json`, `predictions.csv`, `confusion_matrix.csv`, and duplicate-exclusion records were preserved unchanged.
- `ml/models/phase15_external_test/supplementary_metrics.json` contains the independently recomputed metrics and intervals.
- `ml/scripts/summarize_phase15_predictions.py` derives supplementary summaries from saved predictions and validates expected row count, unique hashes, class support, and error counts.
- The evaluator's integrity checks and the supplementary summary script both completed successfully. No browser full-dataset pass, model retraining, build, or unrelated test suite was run; none was needed for this frozen evaluation.

**Conclusion:** Project records support that the reserved external test was untouched before freeze; the frozen-model evaluation completed once. Results are documented above. Future tuning requires a newly collected or otherwise independent test set.
