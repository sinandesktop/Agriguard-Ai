# AgriGuard AI

AgriGuard AI is planned as a mobile-friendly plant disease assessment website. Its primary model will use traditional image features and classical machine learning. This phase establishes the project structure and checks whether a small scikit-learn SVM can be exported to ONNX and run in a browser. It does not train or validate a plant disease model.

## Repository layout

```text
AgriGuard AI/
├── docs/                  # Architecture, setup, and technical decisions
├── ml/                    # Python training and model-conversion workspace
│   ├── data/              # Future local dataset workspace (not committed)
│   ├── scripts/           # Developer commands, including the ONNX feasibility check
│   ├── src/agriguard_ml/  # Reusable Python ML code (added incrementally)
│   └── requirements.txt   # Python ML and ONNX dependencies
├── tests/                 # Python feasibility and conversion tests
└── web/                   # Next.js + TypeScript App Router website
    └── app/               # Routes and responsive styles
```

## Development setup

### Website

Install Node.js 20.9 or newer, then run from `web/`:

```bash
npm install
npm run dev
```

The Next.js app uses the App Router. In Vercel, import the repository and set **Root Directory** to `web`; Vercel then builds the website from that directory. See [Vercel monorepos](https://vercel.com/docs/monorepos).

### Python feasibility check

Use Python 3.11 or newer. From the repository root, create and activate a virtual environment, install the packages listed in `ml/requirements.txt`, then run:

```bash
python -m pip install -r ml/requirements.txt
python ml/scripts/run_onnx_feasibility.py
python -m unittest discover -s tests -v
```

The script writes a synthetic ONNX model and test vectors under `ml/.dev_artifacts/`. That folder is ignored by Git and is not a production asset. To check it in a browser, start the website and open `/onnx-feasibility`; select the generated `.onnx` file and `test_vectors.json` from that folder. The harness uses only generic numeric feature vectors and integer class IDs.

PlantVillage data, real disease labels, production image preprocessing, treatment data, and the optional vision API are not part of Phase 1. Phase 2 adds dataset inspection and leakage-aware split-manifest preparation; it still does not train the real classifier. See [dataset preparation](docs/dataset-preparation.md).

## Configuration

`.env.example` lists placeholder names for a future server-side AI fallback. Copy it to a private `.env` only when that feature is implemented. Local prediction will not require an API key, and credentials must never be exposed to browser code.

## Phase status

Phase 1 established architecture and verified the synthetic SVM-to-ONNX path. Phase 2 prepared the six-class PlantVillage subset and leakage-aware split manifests. Phase 3 trained and evaluated a real HOG + SVM baseline, saved ignored Python and ONNX artifacts, and verified Python/ONNX parity on the held-out test vectors. The model has not been validated on field photographs. The next phase is implementing TypeScript HOG and real-model browser parity.
