# Phase 17 — Farmer Results and Unified Treatment Guidance

## Summary

The assessment page now uses concise farmer-facing wording. Local results continue to use the frozen Phase 9 model and its saved class order. Gemini and Groq share one validated assessment schema, one system prompt, and the same server-side treatment catalogue policy. No model, model artifact, dataset manifest, or evaluation result was changed.

## Diagnosis mapping and crop mismatch

The saved Phase 9 `class_labels.json` remains the authority for class IDs. The shared mapping in `web/lib/disease-mapping.ts` maps all six IDs to the exact model label, crop, disease ID, display name, and healthy status; tests compare that mapping with the saved artifact.

The selected crop and predicted crop are kept separate. Results display the mapped predicted crop and disease. For example, if Tomato is selected and the model maps to `Potato___Early_blight`, the result remains **Potato - Early blight**, followed by a warning that Tomato was selected and the plant should be confirmed before acting. Any catalogue guidance in that case is explicitly conditional on Potato being the correct crop. The app does not relabel the class or display raw SVM scores.

Healthy classes are phrased as a possible healthy appearance. They are not described as proof of health, and the UI states that the assessment may miss disease. There is no unknown-class rejection rule.

## Unified treatment guidance

All three assessment sources use `POST /api/treatment` and `resolveAssessmentGuidance`:

- Local inference submits the selected crop and mapped Phase 9 class ID.
- Gemini and Groq submit the validated exact model label and review/image status.
- The server rejects unsupported crop/class/label values and strict schemas reject client-supplied certainty, medicine, dosage, or other extra fields.
- Only exact `verified` records in `web/data/treatments.json` can be returned. Uncertain but mapped outcomes can receive only the existing non-chemical guidance; unknown outcomes have no disease-specific record. Crop-mismatched advice is marked conditional.
- Healthy results return no disease medicine.

The records now expose a non-chemical management category (`cultural` and/or `sanitation`) and have a typed, detailed structure available for a future chemical record. All current `chemical_options` arrays remain empty and the API response schema rejects chemical options. The UI says: **“A verified chemical recommendation is not available yet.”** No product, active ingredient, formulation, dosage, interval, or pesticide safety instruction was added.

The existing non-chemical catalogue advice and attributions were retained. Its disease-management sources include Kerala Agricultural University guidance for [tomato early blight](https://celkau.in/ecropdoctor/vegetables/tomato/earlyblight.php), [tomato late blight](https://www.celkau.in/Crops/Vegetables/tomato/lateblight.php), [potato early blight](https://celkau.in/ecropdoctor/vegetables/potato/earlybight.php), and [potato late blight](https://celkau.in/crops/vegetables/potato/latelbight.php). The [ICAR Indian Farming article](https://epubs.icar.org.in/index.php/IndFarm/article/view/48821) remains attributed for potato late-blight management.

No verified chemical product and dosage is currently available for tomato early blight, tomato late blight, potato early blight, or potato late blight. Healthy-class records also have no disease medicine.

## Gemini and Groq behavior

Both provider adapters use the same system prompt and Zod assessment schema. The schema requires a separate predicted crop, a short farmer-facing explanation, a supported exact class label or `unknown`, image quality, uncertainty, visible evidence, and expert-review status. It rejects a predicted crop that conflicts with the crop embedded in an exact class label. The prompt receives only a bounded JSON summary of verified, non-chemical catalogue records for the selected supported crop. It explicitly tells the provider not to generate treatment instructions; server-side catalogue resolution remains authoritative.

The generated explanation is included in the server-side safety scan. Provider failures continue to map to safe categories and user messages; image bytes, raw provider responses, and secrets are not logged or returned.

**Live provider status:** no live Gemini or Groq request was made during Phase 17; provider behavior was tested with mocks. A Groq synthetic smoke test was reported as successful by the user before these Phase 17 prompt/schema changes. That earlier result verifies only the then-current route and synthetic response; it does not establish that the updated prompt works live or that either provider gives reliable diagnoses. The earlier Gemini `response_validation_failed` issue therefore remains unconfirmed as resolved. `.env.local` was not read or modified.

## Files changed

- `web/app/assessment/assessment-form.tsx` — simpler before/after workflow, one **Analyze leaf** button, diagnosis-first results, selected crop/source display, mismatch warning, conditional guidance, and concise next steps.
- `web/app/globals.css` — diagnosis hierarchy and source styling.
- `web/lib/assessment-schema.ts`, `web/lib/gemini-provider.ts`, `web/lib/groq-provider.ts`, `web/lib/plant-assessment.ts` — shared explanation schema/prompt, bounded catalogue context, and explanation safety scan.
- `web/lib/treatment-catalog.ts`, `web/lib/treatment-guidance-schema.ts`, `web/app/api/treatment/route.ts` — shared provider policy, mismatch support, categories, and strict request validation.
- `web/data/treatments.json` — added category metadata only; existing advice and source records were retained and chemical options remain empty.
- `web/tests/assessment-ui.test.tsx`, `web/tests/assess-plant.test.mjs`, `web/tests/treatment-catalog.test.mjs`, `web/tests/treatment-route.test.mjs` — regression coverage for the revised UI, mapping, mismatch, safe treatment resolution, prompts, consent, and errors.
- `docs/phase17_farmer_ui_and_treatments.md` — this report.

## Verification

The focused treatment/provider check passed **23/23 tests**.

The final Phase C regression command passed **46/46 tests** across the UI and server tests:

```powershell
cd D:\PlantDiseaseAI\web
npm.cmd run test:phase-c
npm.cmd run typecheck
npm.cmd run build
```

TypeScript checking passed. The production build compiled and generated the expected routes, including `/api/assess-plant` and `/api/treatment`.

Phase 14 browser inference, HOG parity, model evaluation, and the Phase 15 external test set were not run or accessed. No live provider call was made.

## Remaining limitations

- Gemini live behavior after the new response schema/prompt is still unverified. Groq live behavior after these changes is also unverified.
- Treatment guidance is limited to the source-backed non-chemical entries already in the catalogue. The UI must not be interpreted as providing a pesticide prescription.
- A selected class is a possible result, not a confirmed diagnosis. Crop mismatch warnings require the user to confirm the plant before relying on conditional guidance.
- Provider-generated explanations are checked for obvious medicine/product/dose instructions, but this filter cannot establish the truth of a visual diagnosis; provider assessments remain preliminary.
