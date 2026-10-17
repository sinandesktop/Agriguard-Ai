# PlantVillage dataset preparation

## Dataset status

The full PlantVillage repository was not cloned. A filtered, depth-1 sparse clone is stored under ignored `ml/data/plantvillage-source/`; it contains the six selected `raw/color` classes and upstream `leaf_grouping` metadata. The audit found 6,652 readable images: Tomato healthy 1,591, early blight 1,000, late blight 1,909; Potato healthy 152, early blight 1,000, late blight 1,000. All six expected folder names matched exactly. There were no empty classes, unsupported images, or unreadable images. Exact hashing found 14 duplicate pairs (14 extra copies).

## Placement and layout

Keep downloaded data outside Git. The default image root is:

```text
ml/data/plantvillage/color/
├── Apple___Apple_scab/
│   └── image_001.jpg
├── Potato___Early_blight/
│   └── image_002.jpg
└── Tomato___healthy/
    └── image_003.jpg
```

The configured root must contain one class directory per label, with images directly inside each class directory. Do not mix `color`, `grayscale`, and `segmented` variants in one root. For a source checkout whose files are under `raw/color`, point the configuration there. The script accepts JPG/JPEG, PNG, BMP, TIF/TIFF, and WebP; it reports common unsupported image formats such as GIF, HEIC/HEIF, AVIF, and JFIF. Other non-image files are ignored.

Set the location in PowerShell for the current session:

```powershell
$env:PLANTVILLAGE_ROOT = "D:\PlantDiseaseAI\ml\data\plantvillage\color"
```

Or pass `--root "D:\path\to\raw\color"` to the script. No automatic downloader is included. Obtain the dataset manually from the [PlantVillage dataset repository](https://github.com/spMohanty/PlantVillage-Dataset) or its linked [Hugging Face dataset](https://huggingface.co/datasets/mohanty/PlantVillage); inspect the chosen release and license/terms before use. The upstream paper reports 54,306 images in 38 crop-disease classes, and documents color, grayscale, segmented variants and grouped views of the same leaves ([Mohanty et al., 2016](https://doi.org/10.3389/fpls.2016.01419)). The repository describes the collection as open access, but the inspected repository listing did not expose a clear license file. Cite the dataset and paper, retain any attribution/terms supplied with the exact copy you obtain, and do not assume that access alone grants redistribution or commercial rights.

## Labels and first experiment

The recommended first six folders use the common upstream naming convention:

```text
Tomato___healthy
Tomato___Early_blight
Tomato___Late_blight
Potato___healthy
Potato___Early_blight
Potato___Late_blight
```

The paper confirms these crop-condition classes conceptually. The local copy is absent, so the exact directory names have **not** been confirmed against the user's data. The script uses the exact names above by default and stops with the missing names if they are different. Inspect the report first; pass the actual folder names using `--classes`, and keep the actual label spelling in the manifests. Document an explicit mapping before training if a source labels (for example) `Potato___Late_blight` differently. The inspection tool never silently renames a class.

For a manageable first run, use `--per-class-limit 500`; this is a cap on selected readable images, and may be exceeded slightly when needed to keep an identity group intact. Use the full selected class data for the later final experiment after confirming runtime and storage.

## Inspection and split commands

From the repository root:

```powershell
python ml/scripts/inspect_plantvillage.py
python ml/scripts/inspect_plantvillage.py --root "D:\path\to\raw\color"
python ml/scripts/inspect_plantvillage.py --root ml/data/plantvillage-source/raw/color --output ml/data/reports --make-splits --splits-output ml/data/splits --per-class-limit 500 --seed 2026 --upstream-leaf-map ml/data/plantvillage-source/leaf_grouping/leaf-map.json
```

The audit writes `ml/data/reports/dataset_summary.json` and `dataset_report.md`. A missing root is still reported, and the command exits with status 2. Split creation requires every selected class directory to exist and contain readable images. It writes `train.csv`, `validation.csv`, `test.csv`, `splits.json`, and the converted `leaf_group_manifest.csv` under `ml/data/splits/`. Each split row stores a path relative to the configured root, the literal folder label, a SHA-256 digest, and an identity group. Paths and data reports are local and should not be committed with the private dataset.

Splits target 70% train, 15% validation, and 15% test with fixed seed 2026 and deterministic group-aware allocation. Exact byte-identical files share a group, including identical files placed under different labels, and cannot cross splits. Validation is for later model/threshold selection; reserve test for final evaluation. The class counts may depart from target ratios where identity groups are large or class counts are small.

The upstream source describes mappings for multiple views of the same leaf. The cloned `leaf-map.json` maps 5,070 of the 6,652 selected images into 1,268 connected identity groups; the pipeline converts these IDs and combines them with exact hashes. An automated cross-split audit confirmed that neither matching SHA-256 hashes nor generated group IDs appear in more than one split. The 500-per-class cap selected 2,652 images (some classes may exceed 500 slightly to keep groups intact): 1,860 train, 398 validation, and 394 test. Test counts are reserved for final evaluation. This phase does not compute perceptual near-duplicate matches. The remaining 1,582 images without mapped group IDs may include related views; filename similarity is not treated as proof of identity. Keep color images only for this experiment to avoid mixing known related grayscale/segmented variants.

## HOG preprocessing contract proposal

Phase 1 documented the need for Python/TypeScript parity but did not set numeric HOG parameters. For the first classical baseline, record and use this candidate contract consistently:

| Stage | Proposed setting |
| --- | --- |
| Input | One RGB or grayscale still image, decoded without EXIF auto-rotation unless both implementations match it |
| Resize | Direct resize to 128 × 128 pixels (aspect ratio is not preserved), Pillow bilinear interpolation |
| Color | Convert RGB input to 8-bit grayscale using Pillow `convert("L")`; keep grayscale pixels in 0–255 |
| HOG orientations | 9 unsigned orientation bins |
| Pixels per cell | 8 × 8 |
| Cells per block | 2 × 2 |
| Block normalization | L2-Hys |
| Other HOG options | `transform_sqrt=False`, `feature_vector=True`, no visualization |
| Output | Flat row-major NumPy `float32` vector of length 8,100 |

The length is 15 × 15 blocks × 2 × 2 cells × 9 bins = 8,100. The HOG extractor and browser implementation are not created in this phase. Treat these numeric settings as the Phase 2 baseline proposal; verify Pillow resizing/grayscale and scikit-image HOG against TypeScript fixtures before browser inference. Any later parameter change requires a new preprocessing/model version and regenerated features.

## Data limitations

PlantVillage images are collected under controlled conditions; good performance on these images would not establish reliability on field photographs. Image-level stratification can leak visually related leaves when grouping metadata is unavailable. Exact hashing catches byte-identical copies only. This phase does not train, select thresholds, report classifier metrics, or produce model artifacts.
