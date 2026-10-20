# Phase A — Treatment catalogue audit and implementation

Audit date: 2026-10-19

## Repository audit

- No treatment or medicine database, disease-to-treatment mapping, treatment lookup, treatment API route, or farmer-facing treatment display existed before this phase.
- The web application was a developer-only browser ONNX harness. It did not provide a production diagnosis workflow.
- The current Phase 9 model maps IDs as follows: `0 Potato___Early_blight`, `1 Potato___Late_blight`, `2 Potato___healthy`, `3 Tomato___Early_blight`, `4 Tomato___Late_blight`, `5 Tomato___healthy`.
- The Phase 9 model is a six-label tomato/potato classifier, not a universal plant identifier. Its earlier model metadata reports severe external-domain limitations; this catalogue does not turn the classifier output into a confirmed diagnosis.
- No existing treatment recommendations or source attributions were found to verify. The existing architecture document only describes a future structured JSON knowledge base.

## Implementation plan and decisions

1. Store reviewed source data in `web/data/treatments.json`, outside `web/public`.
2. Match only stable plant IDs (`tomato`, `potato`) and disease IDs (`healthy`, `early_blight`, `late_blight`). Keep the model class label on each record to make the mapping auditable.
3. Add a server-side lookup utility and a small Next.js POST route. The caller supplies IDs and an explicit `diagnosis_certain` boolean; it cannot submit medicine names or dosages. Uncertain diagnoses return no record.
4. Return no disease medicine for healthy classifications. For disease records, show only the sourced non-chemical practices in the catalogue.
5. Withhold chemical recommendations. KAU web pages contain pesticide tables, but this phase did not independently verify current Indian registration status and applicable product labels for each intended crop/disease and use. Product names and dosages are therefore not copied into the catalogue.
6. Add focused tests for all six mappings, healthy cases, unknowns, uncertainty, and invalid plant/disease combinations.

## Catalogue content and source verification

Four disease entries contain source-backed non-chemical guidance and symptom descriptions. Two healthy entries explicitly state that no disease medicine is indicated and offer no chemical recommendation. Symptoms are informational and not diagnostic proof. All entries have `chemical_status: no_verified_recommendation`; `chemical_options` is empty.

| Model label | Evidence used |
|---|---|
| Tomato early blight | KAU identifies *Alternaria solani*, typical concentric lesions, and non-chemical practices including disease-free seed and crop-debris removal. [KAU e-Crop Doctor](https://celkau.in/ecropdoctor/vegetables/tomato/earlyblight.php) |
| Tomato late blight | KAU describes the symptoms, disease-free seed, debris removal, and rotation. [KAU Agri-Infotech](https://www.celkau.in/Crops/Vegetables/tomato/lateblight.php) |
| Potato early blight | KAU describes *Alternaria solani*, typical symptoms, disease-free tubers, debris removal, and avoiding waterlogging. [KAU e-Crop Doctor](https://celkau.in/ecropdoctor/vegetables/potato/earlybight.php) |
| Potato late blight | KAU describes symptoms and sanitation, seed-tuber, ridge, and waterlogging practices. ICAR's review also supports healthy seed tubers, removing cull potatoes/volunteers, resistant cultivars, scouting, and forecasting. [KAU](https://celkau.in/crops/vegetables/potato/latelbight.php), [ICAR Indian Farming](https://epubs.icar.org.in/index.php/IndFarm/article/view/48821) |
| Healthy tomato | KAU's tomato crop guidance supports disease-free planting material and rotation as general prevention, not as treatment. [KAU tomato crop guidance](https://old.celkau.in/Crops/Vegetables/Tomato/tomato.aspx) |
| Healthy potato | KAU's potato late-blight guidance supports disease-free seed potatoes and sanitation as general prevention, not as treatment. [KAU potato guidance](https://celkau.in/crops/vegetables/potato/latelbight.php) |

The KAU pages do not display a publication date in the reviewed content; source dates are `null`. Sources were checked on 2026-10-19. The `verified` status here means the listed text is supported by the cited source; it is not an endorsement by a plant pathologist, a product-approval check, or a guarantee of suitability for an individual farm. Region is recorded as India, with local expert confirmation advised.

## What remains unavailable

- No pesticide product or exact dosage is offered. Applicable current labels, registration, crop/disease use, and local restrictions need separate confirmation before chemical recommendations can be added.
- Image symptoms overlap. Late blight may progress quickly; suspected cases should be escalated to local agricultural experts.
- A healthy leaf-class prediction does not establish that the whole plant is healthy.
- The catalogue lookup is server-side data access only; this phase does not connect it to a public farmer workflow.
- The Phase 9 HOG+HSV browser inference is not established as working end-to-end. The current browser prototype is for the older Phase 3 HOG-only model; its use must not be implied for Phase 9.

## Verification performed

Automated tests check all current labels against catalogue records, healthy behavior, uncertainty suppression, unknown IDs, invalid crop/disease combinations, missing records, and fail-closed chemical fields.

- `node --experimental-strip-types --test tests/treatment-catalog.test.mjs` (from `web/`): **PASS**, 5 tests, 0 failures.
- `npm.cmd run typecheck` (from `web/`): **PASS**.
- `npm.cmd run build` (from `web/`): **PASS**; `/api/treatment` compiled as a dynamic route.
- `git diff --check`: **PASS**; Git emitted only a line-ending normalization warning for the pre-existing `web/package.json` modification.
