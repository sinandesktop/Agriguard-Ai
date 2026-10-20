# Phase 16 — Core Workflow Stabilization

**Status:** Code stabilization and mocked/automated verification complete. Gemini account connectivity and response quality still require the opt-in smoke test described below.

## Audit and preservation

Before editing, I reviewed the Phase A treatment audit and Phase 12–15 reports, current assessment and treatment routes, Gemini provider adapter, Phase 9 ONNX inference path, UI and existing tests. Git already contained unrelated uncommitted changes across the app, reports, model scripts, and tests; those changes were preserved. No Phase 15 test image or label was opened or used. The Phase 9 model, preprocessing, class labels, manifests, treatment catalogue records, historical results, and `.env.local` were not modified.

The Phase 9 saved class order was checked against its artifact and is the source of the single shared map:

| ID | Saved model label | Crop | Displayed result |
|---:|---|---|---|
| 0 | `Potato___Early_blight` | Potato | Early blight |
| 1 | `Potato___Late_blight` | Potato | Late blight |
| 2 | `Potato___healthy` | Potato | Healthy-class message |
| 3 | `Tomato___Early_blight` | Tomato | Early blight |
| 4 | `Tomato___Late_blight` | Tomato | Late blight |
| 5 | `Tomato___healthy` | Tomato | Healthy-class message |

## Task 1 — Gemini assessment

### Findings and changes

- The saved model identifier remains `gemini-3.6-flash`. Google's current documentation lists it as a stable image-capable model with structured-output support. The SDK's installed type declarations support `responseJsonSchema`; the existing JPEG inline image payload and `response.text` access match the SDK/API contract.
- The provider request included `temperature: 0.1`. Google's Gemini 3.6 migration notes say to remove `temperature`, `top_p`, and `top_k` for Gemini 3.6. I removed that deprecated field. This is a verified compatibility correction, but without a live provider response it cannot be claimed as the proven cause of earlier failures. [Gemini 3.6 API changes](https://ai.google.dev/gemini-api/docs/generate-content/whats-new-gemini-3.6), [model capabilities](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash), [structured output schema](https://ai.google.dev/gemini-api/docs/generate-content/structured-output).
- The request flow retains server-only `GEMINI_API_KEY`, explicit consent, JPEG normalization, supported MIME checks, 4 MiB upload limit, bounded in-memory processing, and no image persistence. The current model ID was not changed.
- Provider errors now map to fixed categories: missing configuration, credentials/permission, model unavailable, invalid request, quota, timeout, provider unavailable, empty response, and response-schema validation failure. The route sends only the category and friendly fixed message to the browser. Server diagnostics contain only category, numeric status when available, and error class name; response bodies, images, keys, and raw provider messages are excluded.
- The UI handles the new categories, preserves consent during a retry, and prevents concurrent duplicate submits. The result itself remains visible if the separate guidance request fails.
- The local `.env.local` exists and has a non-empty `GEMINI_API_KEY` setting. Its contents were not read, printed, or edited. The development server reports `.env.local` as loaded, but that does not verify that the key is valid or enabled for the account.

### Manual live smoke test

Automated tests use mocks only. To check actual connectivity, open PowerShell in `D:\PlantDiseaseAI\web` and run:

```powershell
npm.cmd run smoke:gemini
```

This opt-in script sends a small synthetic in-memory image through the running local `/api/assess-plant` route, validates the response schema, and prints only a safe status. It does not use an uploaded photo, persist an image, or print the provider response or key. The request may consume API quota. The development server must be running (default: `http://127.0.0.1:3000`); set `AGRIGUARD_SMOKE_BASE_URL` only if it uses another local address.

After Phase 16, one live synthetic request was sent through the local route. It returned HTTP 200 with a schema-valid assessment (`plant_match: uncertain`, `possible_disease: unknown`, `image_quality: poor`, `needs_expert_review: true`). This verifies that the configured server key, account/model access, request format, and structured response path worked for that synthetic request at that time. It does not establish that every real leaf photo will complete within the current 15-second provider abort limit or produce a useful disease assessment. No user photo was sent.

## Task 2 — Disease-result presentation

- Added `web/lib/disease-mapping.ts` as the authoritative class-ID-to-label/crop/disease/display/healthy mapping. Phase 9 inference and UI use that map; tests compare it against `class_labels.json`.
- The primary result now says `Possible result: Early blight` or `Late blight`, or explains that the model did not identify one of its modeled diseases for the selected crop and can still miss disease. The selected crop and source are shown separately.
- Crop mismatches remain labeled with the model's actual predicted crop and show a warning. They are never silently renamed to the selected crop, and guidance is withheld.
- Raw SVC scores are not displayed. The UI repeats that the Phase 9 model always chooses one of six classes and has no validated unknown-class rejection rule. All predictions are preliminary assessments, not confirmed diagnoses.

## Task 3 — Approved guidance

- The treatment route no longer accepts `diagnosis_certain`. Strict request schemas reject unknown fields, including client-injected medicine or dosage. The route derives eligibility from the assessment source and canonical class/assessment fields, then does exact plant/disease lookup in the existing catalogue.
- Local Phase 9 class IDs are mapped on the server. A class/crop mismatch is withheld. Gemini output is treated as untrusted: only an exact supported class label can map to a catalogue record, and mismatch, uncertain plant, poor/uncertain image quality, unknown class, or `needs_expert_review` suppresses disease-specific guidance.
- Healthy-class results can show the existing prevention text but never disease medicine. Disease results can show only the catalogue's existing source-backed non-chemical guidance and source links. The UI explicitly says no verified chemical product, pesticide, or dosage is available. `web/data/treatments.json` was not changed.
- The result card keeps the possible disease, source, limitations, and eligible general guidance together. The route returns only the record fields needed by that UI.
- **Trust boundary:** local inference runs in the browser, so a user can forge an API request containing a different class ID. The endpoint does not treat the client value as proof of diagnosis or certainty; it performs only deterministic mapping to public, general catalogue guidance and never accepts a certainty assertion. Cryptographically proving that an ID came from an unmodified browser model would require moving inference server-side or a separate trusted attestation design, neither of which is part of this phase.

## Files created or changed

- Added `web/lib/disease-mapping.ts` and `web/lib/treatment-guidance-schema.ts`.
- Updated `web/app/real-inference/metadata.ts`, `web/lib/gemini-provider.ts`, `web/lib/plant-assessment.ts`, `web/lib/treatment-catalog.ts`, and `web/app/api/treatment/route.ts`.
- Updated `web/app/assessment/assessment-form.tsx`, `web/app/globals.css`, and `web/package.json`.
- Added `web/tests/treatment-route.test.mjs`; updated `web/tests/assessment-ui.test.tsx`, `web/tests/treatment-catalog.test.mjs`, `web/tests/assess-plant.test.mjs`, and `web/tests/phase14-browser-check.mjs`.
- Added `web/scripts/gemini-smoke-test.mjs` and the `smoke:gemini` npm script.
- Added this report. Existing unrelated working-tree changes were preserved; nothing was committed.

## Verification performed

Executed from `web/`:

| Command | Actual result |
|---|---|
| `npm.cmd run test:phase-c` | **PASS** — 20 UI tests and 19 mocked API/catalogue tests; includes Phase A/B/C regressions and the new treatment route tests. |
| `npm.cmd run test:phase14` | **PASS** — 6 feature parity, ONNX interface, local inference, error, and input-format tests. |
| `npm.cmd run test:hog` | **PASS** — 3 existing HOG regressions. |
| `npm.cmd run typecheck` | **PASS**. |
| `node --check scripts/gemini-smoke-test.mjs` | **PASS** — syntax check only; no network request. |
| `npm.cmd run build` | **PASS** — production build includes `/`, `/api/assess-plant`, `/api/treatment`, and the local model/runtime routes. |
| `npm.cmd run test:phase14:browser` | **PASS** — Chrome checked all six existing Phase 14 non-test PlantVillage fixtures. Decoded source pixels matched the saved Python fixture pixels, and all six browser class outputs matched the saved Python reference predictions. This verifies implementation parity only, not disease accuracy. |

The browser test driver had two existing harness problems: it used a direct property assignment that React did not recognize as a user crop change, and it used `127.0.0.1`, which caused Next dev-server cross-origin HMR warnings against `localhost`. The test now uses the native select setter and defaults to `localhost`. No dataset or model files were changed by the browser check.

## Workflow checklist

- [x] Local tomato and potato predictions render the actual mapped disease name.
- [x] All six mappings, including both healthy cases, have tests.
- [x] Local and external result cards label selected crop and assessment source.
- [x] Healthy results do not return disease medicine.
- [x] Crop mismatch, Gemini uncertainty, poor image quality, unknown results, and expert-review flags suppress disease guidance.
- [x] Eligible local and Gemini examples display exact catalogue guidance and its sources.
- [x] Consent is required for Gemini; retry retains the user's explicit consent and a busy guard prevents overlapping submits.
- [x] Missing configuration and safe error categories have mocked tests.
- [x] Model loading/inference errors do not silently call Gemini.
- [ ] Live Gemini connectivity and usable response under the configured account — requires running the opt-in smoke test.

## Remaining limits

This phase does not establish that the configured Gemini key is valid, that the account has model access/quota, or that real responses are useful. Run the smoke test first, then retry Gemini in the app with a photo only after reviewing the consent notice. The local model remains limited to its six classes and can misclassify unfamiliar or poor-quality photos. Catalogue guidance is general non-chemical management information, not a prescription; no pesticide/product/dosage was added.
