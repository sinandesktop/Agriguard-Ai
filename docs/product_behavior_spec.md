# AgriGuard AI — Product Behavior Specification

Audit date: 2026-10-20  
Scope: read-only inspection of the current Next.js application, related tests, and prior phase reports. This is a specification, not an implementation change.

## 1. Executive summary

The current home page is a single assessment form. A farmer selects a plant, uploads a JPEG/PNG/WebP image up to 4 MiB, and chooses either the in-browser Phase 9 model (Tomato or Potato only) or the consent-gated Groq service. The app displays local model results separately from external-provider results. External assessments can name crops outside the six local classes, including Banana, and can return disease, healthy, uncertain, or cannot-assess outcomes.

The major verified capabilities are the six-class Phase 9 ONNX mapping and browser feature path, client/server image checks, explicit external-provider consent, safe response parsing, source-backed non-chemical catalogue guidance, and a set of mocked UI/API tests. Phase 14's report records browser/Python feature and prediction parity on six representative non-test PlantVillage images. That is implementation-parity evidence, not diagnostic-accuracy evidence.

Important limitations and defects:

- The ordinary UI offers Groq and local assessment. Gemini has been removed from the active application, provider route, dependencies, and configuration template. Older phase reports retain historical references.
- Local inference always selects one of six classes. It has no validated unknown-crop/disease rejection or image-quality assessment.
- The form controls remain changeable while a request is running. A user can replace the image or change the selected plant after submission; the old request can then populate results beside the new form values. There is no request cancellation or submission identity check.
- External AI can produce different disease names on repeated requests. Current Groq code has a fixed seed and temperature 0, but provider documentation calls seeded determinism best-effort; this cannot guarantee repeatable image interpretation.
- The public `/api/treatment` route accepts only a local class ID. External guidance is resolved inside `/api/assess-plant`, so the routes do not provide one uniform public treatment API contract.
- There are six tomato/potato catalogue records. Selected Banana/Mango/Chilli labels can resolve to curated general references, but those are not verified prescriptions; unmatched labels and custom crops have no disease-specific record. No chemical product or dosage is verified.
- Provider-generated `ai_suggestions` are separate, explicitly unverified advice. A keyword/amount safety filter exists, but it is not a complete semantic safety system.

No live external AI call, model evaluation, build, or test suite was run for this documentation task. No `.env.local` contents were read. Git already contains extensive pre-existing changes; this audit did not modify them. The only created file is this document.

### Evidence labels

- **Source evidence** means the current code directly implements the behavior.
- **Test evidence** means an automated test has a named assertion for the behavior. Tests were inspected, not rerun in this task.
- **Report evidence** means a prior project report records a completed check; it is historical and may not describe today's current form.
- **User-reported evidence** means a terminal result was supplied in this conversation, not reproduced during this read-only audit.
- **Unverified** means source inspection or existing tests do not establish the real-world or browser behavior.

## 2. Current architecture and data flow

### 2.1 Main page and local path

`web/app/page.tsx` mounts `AssessmentForm` from `web/app/assessment/assessment-form.tsx`.

1. The form stores selected plant, optional custom name, selected method, chosen `File`, preview URL, external consent, pending state, result state, and safe errors in React state.
2. For local assessment, only Tomato and Potato show a local radio option. `runPhase9Inference` (`web/app/real-inference/phase9-inference.ts`) decodes the file in the browser, asks a Web Worker to extract the 8,132 HOG+HSV features, loads `/api/models/phase9`, and runs ONNX Runtime Web.
3. ONNX label output is mapped by class ID using `web/lib/disease-mapping.ts`. Class IDs are 0 Potato early blight, 1 Potato late blight, 2 Potato healthy, 3 Tomato early blight, 4 Tomato late blight, 5 Tomato healthy. The second ONNX output is raw SVC decision scores despite its name; the farmer UI does not display it.
4. After local inference, the browser posts `{source:"local_phase9", selected_plant_id, class_id}` to `/api/treatment`. That route validates the strict schema and resolves only a fixed class-to-catalogue mapping. The image is not sent to this route or to the external assessment route for local inference.

### 2.2 External Groq path

1. The form shows Groq for any selected plant. It is the only external provider in the current application.
2. The UI requires the user to check a consent box and deliberately submit. The multipart request contains `plant_id`, optional `custom_plant_name`, `provider`, literal `external_consent=true`, and the image.
3. `createAssessmentHandler` in `web/lib/plant-assessment.ts` validates POST, request size, consent, provider, plant/custom name, file MIME and actual decoded format. Sharp applies orientation, checks dimensions/pixels, scales to 1280px maximum, and converts the image to JPEG in memory.
4. The server reads only `GROQ_API_KEY` from server process environment; this audit did not read `.env.local` or print secret values. Provider calls time out after 30 seconds. The in-memory limiter allows five requests per client key per 60 seconds per handler instance.
5. Provider text is JSON-parsed and checked by `AssessmentSchema` (`web/lib/assessment-schema.ts`), which allows bounded free-text crop and disease names and four result states. It is not constrained to the local Phase 9 six-class enum. A keyword/amount/action-pattern filter rejects some chemical guidance. Server-computed expert review is forced for mismatch, poor/uncertain quality, uncertain, or cannot-assess results.
6. The server resolves guidance using the shared `resolveAssessmentGuidance` function and returns assessment plus guidance. External provider text cannot choose a treatment record. The client parses the response again before rendering.

### 2.3 Treatment data and output

- `web/data/treatments.json` contains exactly six records: healthy, early blight, late blight for Tomato and Potato. These records have source-backed non-chemical/prevention text, empty `chemical_options`, and `chemical_status: no_verified_recommendation`.
- `web/data/external_guidance.json` adds exact crop-scoped aliases for selected Banana, Mango, and Chilli conditions. Its `general_reference` results are cautious non-chemical reference steps and links. Broad leaf-spot labels link to differential pages; they do not establish the cause of the uploaded leaf.
- `web/lib/treatment-catalog.ts` performs exact mappings. Nearby strings are not guessed. Uncertain results, poor image quality, ambiguous plant match, missing records, or provider review requirements can withhold disease-specific guidance.
- `/api/treatment` accepts only `local_phase9` plus Tomato/Potato and class ID 0–5. It rejects external diagnosis fields and unknown injected properties. External resolution is part of `/api/assess-plant`, not this route.
- `TreatmentGuidanceSchema` rejects chemical option entries (`z.never()` with max zero). The UI renders non-chemical catalogue text/source links, a “verified chemical recommendation is not available yet” note, general references, or a “Verified guidance unavailable” result. Groq suggestions are a separate section labelled **AI suggestions — not verified**.

### 2.4 Operational detail

- Local model bytes are streamed from `web/app/api/models/phase9/route.ts` from the Phase 9 model path. That route checks the expected file size and sends a fixed ETag containing the known model hash; the handler does not recompute SHA-256 on each request.
- ONNX Runtime assets have an allowlisted route under `web/app/api/runtime/ort/[asset]/route.ts`.
- The form uses `URL.createObjectURL` for its preview and revokes the URL on image change/unmount.
- No page navigation, persistent result history, or dedicated “start over” action exists in the current assessment form.

## 3. Existing features and verified capabilities

| Capability | Current implementation | Evidence and confidence |
|---|---|---|
| Single assessment page | Home page mounts one form; no navigation or landing-page content sections exist. | Source: `web/app/page.tsx`, `web/app/assessment/assessment-form.tsx`. |
| Plant options | Tomato, Potato, Mango, Banana, Chilli, Other/specify. Other requires 1–80 character letters/numbers/spaces/apostrophe/period/hyphen. | Source: `PLANTS`, `parsePlant`; tests include custom name. |
| Local model coverage | Only Tomato/Potato; exact six Phase 9 class mapping. | Source: `disease-mapping.ts`; UI and mapping tests. Historical browser parity: Phase 14 report, six examples. |
| Local unknown rejection | None. Every valid inference selects one of six classes. | Source: ONNX `label` output and range validation; no rejection threshold. No local confidence is shown. |
| Image input | Browser accepts JPEG, PNG, WebP up to 4 MiB; `capture="environment"`; preview, replace, remove. | Source and named UI tests for type, size, replace/remove. |
| Server image acceptance | MIME and decoded format must match; 4 MiB file and bounded multipart request; up to 20 MP; normalized in-memory JPEG. | Source: `plant-assessment.ts`; mocked endpoint tests. |
| Consent | Unchecked by default; method/image/plant changes clear consent; server requires literal true before provider call. | Source and tests `submission requires explicit consent...`, `changing plant context clears prior consent`, endpoint consent test. Mid-request cancellation is not implemented. |
| Groq | UI available; current Qwen vision adapter uses structured JSON. Current request has temperature 0, fixed seed 1701, and one-name/uncertainty prompt. | Source: `groq-provider.ts`, `groq-request.ts`; test `Groq request uses low-variance sampling...`. User previously supplied a successful synthetic smoke-test transcript; this audit did not reproduce it. |
| Gemini | Removed from the active application. | Historical phase reports may still describe earlier implementation states. |
| External plant breadth | External schema permits free text bounded to 80 char plant / 100 char disease; not limited by local classes. | Source: `assessment-schema.ts`; Banana provider/UI/catalog tests. It does not validate botanical identity. |
| Treatment lookup | Exact server mapping and catalogue records only; no client treatment text/certainty accepted by treatment endpoint. | Source: route/catalog schemas and injection tests. Browser-supplied class ID itself is not cryptographic proof of model output. |
| External references | Selected exact Banana/Mango/Chilli labels can show general references; arbitrary custom plant/label can remain unavailable. | Source: `external_guidance.json`, `treatment-catalog.ts`, tests for exact aliases and rejection of near matches. |
| Provider errors | Fixed friendly UI messages for known categories; server logs sanitized provider/category/status/error name. | Source and mocked timeout/quota/auth/schema tests. No live provider operation was tested here. |
| Cross-request rate limit | Five calls/60s keyed by `x-real-ip` or first `x-forwarded-for`, in memory per handler instance. | Source. It is not global across serverless instances and depends on trusted forwarding headers. |
| Model accuracy | Historical Phase 15 external Potato test: 77.61% accuracy, 72.01% three-supported-class macro F1, early blight recall 57.94%, late blight recall 65.11%, healthy recall 92.36%. | `docs/phase15_external_test_evaluation.md`; consumed test set. Not shown in UI and not a guarantee for other locations/crops. Do not retune from that test. |

## 4. Complete user-action and condition matrix — existing behavior

Status values: **Verified working** means direct automated or recorded evidence exists; **Partially verified** means implementation exists but one or more requested dimensions are not tested or only mocked; **Broken/inconsistent** means a concrete defect or contract gap is visible; **Unverified** means no evidence for actual operation. Test references point to names in `web/tests/` and prior reports. Tests were inspected, not executed during this task.

### A. Initial page and form

| # | Starting condition → user action | Current system behavior and visible output | Next action | Status | Evidence |
|---:|---|---|---|---|---|
| 1 | Open `/`. | Shows “AgriGuard AI”, “Check your plant”, plant select, image input guidance, and assessment-source controls; source controls explain local/external privacy when applicable. | Select a plant. | Verified working (render test). | `page.tsx`; UI test “initial form uses farmer wording…” |
| 2 | No crop selected; submit attempt. | Button is disabled because `canSubmit` is false. If submit is triggered programmatically, inline alert says “Select a plant to continue.” | Choose one of six options. | Partially verified. | `assessment-form.tsx`; “plant selection is required…” |
| 3 | Select Tomato or Potato. | Local and Groq options appear; “Local assessment available / Your photo stays in this browser.” | Upload, then choose method. | Partially verified; option rendering is explicit, not an all-crops interaction test. | `PLANTS`, JSX conditional; UI tests local Tomato/Potato results. |
| 4 | Select Mango, Banana, Chilli, or Other. | Local option is omitted; page says “Online assessment required / Local assessment supports tomato and potato.” Groq appears. Other additionally shows custom-name input. | Upload, choose Groq, consent, submit. | Partially verified. | Form JSX; Banana UI test and custom-name test. |
| 5 | Crop selected; no image; submit. | Button stays disabled; guarded submit would show “Choose a leaf photo to continue.” | Select a photo. | Partially verified; no named no-image submit test found. | `canSubmit`, `submit`. |
| 6 | Choose a valid JPEG/PNG/WebP within 4 MiB. | Stores File, creates preview URL, clears old result/error and resets consent. | Choose method and submit. | Verified working for supported format behavior; image selection tested using mock file. | `chooseImage`; UI tests. |
| 7 | Choose unsupported format. | Image cleared, preview absent, input alert: “Choose a JPEG, PNG, or WebP image.” No provider request. | Choose supported image. | Verified working. | UI test “unsupported image types…” |
| 8 | Choose file >4 MiB. | Image cleared; alert: “The image must be 4 MiB or smaller.” | Choose smaller file. | Verified working client-side; server limit separately tested. | UI test “images larger than 4 MiB…”; API tests. |
| 9 | Valid image selected; remove or select replacement. | Preview displays file name; remove clears image/consent/result; replacement creates new preview and revokes prior URL. | Upload replacement or proceed. | Verified working. | UI test “preview can be removed and replaced…”. |
| 10 | Observe Analyze leaf button with varying inputs. | Enabled only with valid plant, image, method, required consent, and `!loading`. Invalid optional custom plant disables it. | Complete missing fields. | Partially verified; consent and loading are asserted, not every combination. | `canSubmit`; UI tests consent/loading. |

### B. Privacy and method selection

| # | Starting condition → user action | Current system behavior and visible output | Next action | Status | Evidence |
|---:|---|---|---|---|---|
| 11 | Tomato/Potato selected; choose Local. | Local call runs browser decode/feature extraction/ONNX; image not posted to assessment API. | Wait for model/result; retry or explicitly select Groq on error. | Verified via Phase 14 parity report and source. | `runPhase9Inference`; Phase 14 report; UI local tests. |
| 12 | Choose Gemini. | No such option exists; the server accepts only Groq requests. | Farmer cannot select it or invoke it through the API. | Removed by product decision. | Form provider controls; `/api/assess-plant` Groq-only handler. |
| 13 | Choose Groq. | Groq radio appears for any selected crop. Its notice says photo goes to Groq only after consent and analyze. | Check consent and submit. | Verified mocked flow. Live path user-reported smoke success before latest change; not rerun in this audit. | UI test “Groq is an explicit consent-gated provider…”; supplied smoke transcript. |
| 14 | Select Groq, leave consent unchecked, attempt submit. | Button disabled. Server independently rejects missing `external_consent=true` with 403 `consent_required`. | Check consent. | Verified by UI/API tests. | UI “submission requires explicit consent…”; API “rejects invalid plants and requires explicit…” |
| 15 | Check consent and submit. | Sends image, selected plant, provider, and consent flag to `/api/assess-plant`; server normalizes and sends to chosen provider. | Wait for result/error. | Verified with mocked requests; provider data sharing is a real external transmission only when this action occurs. | UI consent test; `plant-assessment.ts`. |
| 16 | Revoke consent, change plant, or replace image before submit. | Consent is cleared by plant/image/method handlers; submit remains disabled until checked again. | Re-consent if Groq still selected. | Verified for clearing consent on plant context; image/method reset is source-visible, not separately covered in all cases. | `choosePlant`, `changeCustomPlant`, `chooseImage`, radio handlers; UI test changing plant context. |
| 17 | While assessment is in progress, change consent/method/crop/image. | Fields remain interactive. Changing them clears displayed result and consent as applicable, but does not abort an already-sent provider request. That old request can later set assessment state. | Wait may produce a stale result; user can manually start another. | Broken/inconsistent: stale-result race and no cancellation. Revoking consent cannot retract already transmitted image. | Form controls lack `disabled={loading}`; no AbortController/request identity; handlers clear state only. |
| 18 | Submit when provider key is absent/invalid. | Missing key returns 503 safe message; invalid credentials are classified from provider status as `provider_credentials`. UI displays safe generic/provider-specific wording. | Add valid server-side config/restart or retry later. | Missing-key path mocked; live credential validity unverified. | `route.ts`, `classifyProviderFailure`; endpoint tests; `.env.local` not read. |

### C. Local model results

| # | Starting condition → user action | Current system behavior and visible output | Next action | Status | Evidence |
|---:|---|---|---|---|---|
| 19 | Local model returns Potato/Tomato healthy class. | “Possible diagnosis” identifies crop and says “possible healthy appearance”; guidance is prevention only and says no disease medicine; note says model can still miss disease. | Monitor for new symptoms; seek advice if needed. | Verified by mapping/UI/catalogue tests. | `disease-mapping.ts`; UI test healthy local; treatment tests. |
| 20 | Model returns one of four disease classes. | Shows predicted crop plus Early/Late blight; asks user to watch symptoms/seek local advice; shows exact available non-chemical guidance if eligible. | Review source links and seek expert if symptoms persist/worsen. | Verified by unit/UI tests; diagnostic correctness not established by parity. | mapping and UI tests; Phase 14 report. |
| 21 | Predicted local crop matches selected crop. | Result separately shows selected crop and actual model crop/disease, without raw scores or probability. | Read catalogue guidance. | Verified by UI tests. | UI test local Tomato/Potato results and all-six mapping test. |
| 22 | Model crop differs from selected Tomato/Potato. | Displays predicted crop and a mismatch warning; does not relabel result; catalogue output is marked conditional and warns against chemical use. | Confirm plant/disease before action. | Verified by UI test. | UI test “crop mismatch displays predicted crop…”. |
| 23 | Local input appears uncertain, blurry, or low quality. | Local pipeline has no image-quality or uncertainty branch; it still returns one of six labels if inference succeeds. | User may assess result cautiously; no automatic low-quality warning. | Broken capability gap. | Inference output only label/scores; no quality/rejection check; local result UI tests. |
| 24 | Image is outside training distribution or does not resemble supported classes. | The model still selects one of six classes; there is no unknown-class threshold. UI does not claim unknown detection. | Treat result cautiously; currently no specific local unknown action. | Unsupported capability (explicit limitation). | `phase9-inference.ts`; Phase 14 report. |
| 25 | Select Banana/Mango/Chilli/Other and try local. | Local radio is not displayed; if locally forced, submit shows “Local assessment is available only for tomato and potato.” | Choose Groq. | Verified source behavior; not a disease-recognition guarantee for online AI. | Form conditions; external-crop tests. |
| 26 | Local model file fails to load. | Safe local error: “Local assessment could not be completed. You can retry or explicitly choose the optional online assessment.” No silent Groq call. | Retry or deliberately choose Groq with consent. | Verified by mocked local error test; 310 MB network download failure not freshly reproduced. | UI “local model failure stays local…”; `phase9-inference.test.mjs`. |
| 27 | Decode, worker, feature extraction, ONNX execution, or output validation fails. | Caught and shown as same actionable local error; model response must have expected names/shapes/class ID. | Retry or explicitly choose Groq. | Partially verified by unit tests; broad browser/device errors unverified. | `runPhase9Inference`; inference tests; Phase 14. |
| 28 | Browser worker or model/runtime network unavailable. | Promise rejects and same generic local error appears. Model session promise resets after failed load, allowing retry. | Retry after connectivity is restored. | Partially verified; Phase 14 report establishes six Chrome samples, not every browser/network. | inference code; test model-load failure; Phase 14 report. |

### D. External AI results

| # | Starting condition → user action | Current system behavior and visible output | Next action | Status | Evidence |
|---:|---|---|---|---|---|
| 29 | External assessment using the removed provider. | No route or provider adapter exists for it. | Use Groq if available, or local assessment for Tomato/Potato. | Removed. | Current assessment route and provider tests. |
| 30 | Select Groq, consent, submit valid photo. | Server uses `qwen/qwen3.8-27b` and returns validated assessment/guidance; UI renders. | Follow diagnosis/guidance or retry safe error. | Mocked app flow verified; earlier user-reported synthetic smoke passed; current live diagnosis not reverified. | UI/API tests; user transcript. |
| 31 | Provider returns healthy. | Assessment schema requires known plant and null disease for healthy. UI says “[crop] - appears healthy”; healthy guidance never supplies disease medicine. | Monitor and seek advice if symptoms occur. | Source and mapping tested partly; external healthy UI scenario lacks a named dedicated test. | `AssessmentSchema`; renderer; catalogue behavior. |
| 32 | Provider returns a disease name. | Bounded free-text disease is displayed as `[predicted_crop] - [possible_disease]`; catalogue matched by exact aliases. Unmatched names keep diagnosis and show unavailable guidance plus Groq suggestions. | Review source links/suggestions cautiously. | Verified with disease, Banana, near-match and guidance tests; actual disease accuracy unverified. | schema, resolver, UI/catalogue tests. |
| 33 | Provider identifies Banana or another nonlocal plant. | Valid free-text crop is accepted; Banana result can render even though local model has no Banana class; exact curated labels may get general references. | Confirm species/diagnosis; no local model fallback. | Verified via mocked Banana API/UI tests; plant identity accuracy unverified. | external crop support tests. |
| 34 | Provider crop differs from selected crop. | UI shows its predicted crop and mismatch notice. Guidance may be conditional on predicted crop when catalogue mapping succeeds; custom aliases may create false mismatch by string comparison. | Confirm actual crop before taking action. | Partially verified; exact standard crop mismatch test exists; aliases not normalized in UI comparison. | `assessmentCropMismatch`; mismatch UI/catalogue tests. |
| 35 | Provider returns malformed, empty, oversized, incomplete, or invalid response. | Server returns fixed error code/message; client does not display provider raw text. Schema and keyword safety guard reject some unsafe text. | Retry or ask expert; no result is shown. | Verified using mocks. | `parseAssessment`; tests “empty, malformed JSON…” and UI malformed response. |
| 36 | Provider says uncertain/cannot assess. | Disease null is required; diagnosis says “disease uncertain” or “Could not assess this plant reliably”; caution/next step appears; disease-specific guidance is withheld. AI suggestions may advise retake/seek expert. | Retake clear photo or ask expert. | Verified by API/UI tests. | Schema, UI uncertainty tests, catalogue tests. |
| 37 | Key missing or rejected. | Missing key: safe configuration response and UI configuration message. Auth/permission failure: classified provider credentials and safe UI message. | Administrator configures server key, restarts; if rejected, check provider account access. | Missing mocked; invalid-key live status unverified. | route/error map; API tests. |
| 38 | Rate limit/quota, timeout, network, provider or invalid-request failure. | Server maps HTTP status/timeout to safe categories; UI has corresponding fixed messages, else generic fallback. Rate limiter is 5/minute per handler instance. | Wait/retry; check Groq usage if quota. | Mocked classification tested; live provider status unverified. | `classifyProviderFailure`; API tests; `SAFE_ERRORS`. |
| 39 | Error shown; user retries. | “Try again” submits same form if still eligible; busy ref blocks overlapping requests; retry clears old state at submit and uses currently selected fields/consent. | Retry once service/input is ready. | Verified by UI retry test. | UI test “server failure is shown safely and retry works.” |
| 40 | Error shown; user switches assessment method. | Current UI switch is Groq ↔ local where Tomato/Potato. Selecting local resets consent/results; selecting Groq resets consent/results. | Re-consent before another Groq request. | Partially verified; explicit local choice test exists. | source radio handlers and UI local failure test. |

### E. Diagnosis and treatment

| # | Starting condition → user action | Current system behavior and visible output | Next action | Status | Evidence |
|---:|---|---|---|---|---|
| 41 | Healthy classification. | Local returns healthy prevention text; external healthy can resolve a tomato/potato healthy record if image/plant/review gates pass. Neither returns disease medicine. | Monitor the plant. | Catalogue/server verified; external UI branch partly tested. | treatment catalogue tests; resolver. |
| 42 | Exact disease has one of six local catalogue records. | Local route returns source-backed non-chemical text and sources, with chemical unavailable note. External exact Tomato/Potato needs good quality, match, and no expert review. | Read sources; confirm diagnosis before action. | Verified via treatment route/catalogue/UI tests. | `treatment-route.test.mjs`, catalogue and UI tests. |
| 43 | Disease name has no exact catalogue record. | External diagnosis stays visible; guidance status is unavailable. Selected exact Banana/Mango/Chilli aliases may instead return curated `general_reference`, not a catalogue prescription. | Read linked references if provided; otherwise expert. | Verified for unmatched Banana and exact expansions. | treatment tests, `external_guidance.json`. |
| 44 | Unsupported crop/disease/custom name. | Diagnosis can remain visible if provider schema validates; guidance says verified unavailable; generic KAU/TNAU portals also appear in external result. | Seek relevant local expert/source. | Partially verified; no complete crop/disease coverage. | resolver, renderer, general portal constants. |
| 45 | Crop mismatch. | Local UI displays model's predicted crop and conditional catalogue guidance if its exact record exists. External UI shows selected and predicted crop and warning; server resolver may return conditional record or reference only if matching gates pass. | Confirm crop before disease-specific action; no chemical prescription. | Standard cases tested; normalization edge case is a risk. | mismatch UI/catalogue tests. |
| 46 | Uncertain/poor photo/review-required. | Uncertain/cannot-assess and exact Tomato/Potato catalogue paths withhold disease-specific guidance; source links may be given. However, the external `general_reference` branch checks image quality and `plant_match` only: it does not check `needs_expert_review` or compare selected crop with `predicted_crop`. A curated reference can therefore still appear when expert review is requested or the selected and predicted crops differ. Groq suggestions also appear separately as unverified. | Retake or ask expert; treat any mismatched reference as comparison-only. | Broken/inconsistent eligibility for general references; no chemical prescription is returned. | `externalCropReference` and `resolveAssessmentGuidance` in `treatment-catalog.ts`; Banana tests. |
| 47 | Local `/api/treatment` fails, malformed response, or network rejects. | `loadGuidance` catches failure; diagnosis remains and guidance becomes “Verified guidance could not be loaded. Ask a qualified agricultural expert.” External assessment does not call this route; its guidance is in assessment response. | Retry assessment or rely on expert/reference links. | Source-visible; no explicit UI route-failure test found. | `loadGuidance`, external assessment flow. |
| 48 | Unknown class ID/treatment identifiers or client-supplied treatment. | Strict treatment request rejects unknown source/class/fields (400). Resolver rejects class outside 0–5. It does not accept user medicine/dose. A caller may nevertheless submit any valid local class ID; API cannot prove the browser actually ran inference. | Correct request; no treatment record for unknown values. | Verified schema/API tests; provenance caveat remains. | route schema and injection tests; Phase 16 audit note. |
| 49 | Guidance has sources or no sources. | Available records/general references show source titles as external links. Unavailable results may show record sources only when resolver supplies them. Generic KAU/TNAU portals appear for every external result. | Open source in new tab. | Source-rendering tested for Banana and Tomato/Potato; link availability varies. | UI and catalogue tests; `agricultural-references.ts`. |
| 50 | Any result with chemical question. | No chemical product, formulation, dosage, concentration, interval, or label instruction is in current treatment output; UI says chemical recommendation unavailable. Groq AI suggestions are checked by a limited text filter and prompt prohibition, not made verified by that filter. | Consult qualified local expert/current label; no in-app chemical regimen. | Verified catalogue data; semantic safety of provider text only partially controlled. | treatment JSON, schema `z.never()`, prompt/parser. |

### F. Page state and recovery

| # | Starting condition → user action | Current system behavior and visible output | Next action | Status | Evidence |
|---:|---|---|---|---|---|
| 51 | Submit a valid assessment. | `loading=true`; button says “Analyzing leaf…”; live status says model loading/feature extraction or preparing provider request. | Wait. | Verified with UI loading test. | UI test “loading state is visible…” |
| 52 | Click/submit repeatedly while pending. | Button disabled; `submissionInFlight` ref returns before another request; unit test confirms one request for deliberate submit. | Wait for completion. | Verified in normal same-component event flow. | `submit`; UI consent/one-request test. |
| 53 | Valid response arrives. | Result is stored and inline result panel appears; displays diagnosis, selected crop, source, explanation, AI suggestions (external), guidance, next step, disclaimer. | Read result/source; user may change inputs. | Verified with mocked UI tests, not all states. | `assessment-form.tsx`, UI tests. |
| 54 | Request fails or response invalid. | Safe error panel appears, retry button shown where submit remains eligible; no raw provider text displayed. Local errors have a separate message. | Retry/change method/photo. | Verified by mocked tests; unanticipated browser crashes unverified. | UI error handler and tests. |
| 55 | Retry after error. | `submit` clears previous state and sends another request; consent remains if not changed. | Wait for result. | Verified by retry test. | UI test “server failure is shown safely and retry works.” |
| 56 | Start a new assessment. | No dedicated reset/start-new button. Change crop or image/remove image and old result is cleared. | Change current form inputs. | Partially implemented; explicit reset action absent. | form state handlers. |
| 57 | Replace image after a result. | `chooseImage` clears result/consent and validates replacement; current image preview updates. If while request active, no cancellation and race remains. | Re-select method/re-consent. | Verified when idle; broken race while pending. | UI preview test; form request handling. |
| 58 | Change crop after a result. | Clears method, consent and result; crop-specific choices update. | Choose method again. | Source-visible; plant-context clear test verifies consent; all result reset scenarios not individually tested. | `choosePlant`. |
| 59 | Switch assessment method after a result. | Clears result/guidance/error and consent; local option only appears for Tomato/Potato, Groq for all crops. | Re-consent if switching to Groq. | Source-visible; selected local fallback test. | radio handlers and UI test. |
| 60 | Change inputs while a request is still active. | No snapshot ID or abort exists. Old response may write assessment/guidance after user changed crop/photo/method. The display then uses some current UI state with old provider result. | User must notice and manually rerun; safe recovery is not guaranteed. | Broken, high severity. | `submit`, state handlers; no `AbortController`, request ID, or loading-time control lock. |

## 5. Diagnosis and treatment output rules — existing behavior

### Local model

- The model's selected crop is taken from the class mapping, never inferred from the farmer's selection. The UI keeps “Selected crop” separate from the predicted crop.
- Healthy classes show “possible healthy appearance”; they do not produce disease medicine.
- Disease classes are Early/Late blight for Tomato/Potato only.
- Raw SVC decision scores are not shown or presented as percentages.
- The model has no validated confidence, unknown-class, unsupported-crop, or low-quality rejection mechanism. It always selects one class when inference succeeds.

### External provider

- The shared schema accepts bounded free text; it does not require a disease to map to the local classes.
- The UI displays the provider's `predicted_crop` and `possible_disease` verbatim after schema validation. Uncertainty/cannot-assess are distinct outcomes.
- Model output is an assessment, not a confirmed diagnosis. Provider output is not a validated botanical/disease label.
- Any Groq `ai_suggestions` are unverified. The server's text filter catches certain named chemicals, dose patterns, and action-plus-product patterns, but cannot prove all generated care advice safe or appropriate.

### Treatment rules

- Local API: strict `source=local_phase9`, selected plant Tomato/Potato, integer class 0–5. The server maps the class to exact catalogue records. It rejects client disease IDs, certainty assertions, product names, and dosages.
- External API: validated provider result is passed to resolver inside assessment route; it does not use `/api/treatment`. Tomato/Potato can map to exact disease aliases; selected Banana/Mango/Chilli aliases can return curated references. Other valid diagnoses remain visible with unavailable guidance.
- Eligibility uses provider fields such as `plant_match`, `image_quality`, `needs_expert_review`; the browser cannot assert treatment certainty in the public treatment route. Nonetheless, external assessment itself relies on AI-reported quality/review fields and cannot independently establish diagnostic certainty.
- Six catalogue records, all non-chemical/prevention and source attributed. `chemical_options` must be empty. No medicine/dose/interval is offered.

## 6. Error and recovery behavior — existing behavior

| Condition | Current API/UI behavior | Recovery currently offered |
|---|---|---|
| Missing provider key | `provider_not_configured`, 503, fixed user message naming provider configuration variable but not value. | Set server env and restart; UI may retry. |
| Authentication/permission | `provider_credentials`, safe fixed message. | Check account/key outside app; retry. |
| Model unavailable (404) | `model_unavailable`. | Wait/configure provider model; admin may need to inspect. |
| Invalid request/schema/provider 400/422 | `provider_invalid_request`. | Retry or administrator checks provider config. |
| Rate/quota 429 | `provider_quota`; browser text says Groq returned rate/usage limit. | Wait and check provider usage. |
| Timeout | 30 second server `AbortSignal.timeout`; `provider_timeout`. | Retry later. No cancellation/retry backoff. |
| Provider 5xx/network | `provider_unavailable` or generic provider error. | Retry later. |
| Empty/malformed/schema invalid/safety-filter failure | `empty_provider_response` / `response_validation_failed`; raw provider content is not returned. | Retry or expert. |
| Client unsupported/malformed image | `invalid_image` (415); bad multipart `invalid_request`; too large 413. | Select readable JPEG/PNG/WebP up to 4 MiB. |
| Local inference/model failure | Local error panel; no automatic external fallback. | Retry or explicitly choose Groq. |
| Treatment API/network error | Diagnosis remains; guidance degrades to safe unavailable message. | Retry new assessment or consult expert; no dedicated guidance retry button. |
| Missing/malformed treatment result in external response | Client schema failure substitutes `guidance_withheld/unavailable` but keeps validated diagnosis. | Use expert/reference. |

`rateLimited` uses in-memory state per handler instance and trusts `x-real-ip` / `x-forwarded-for` values as client keys. It is not a global production limiter; IP header trust depends on deployment proxy configuration.

## 7. Known bugs and inconsistencies, prioritized

### High priority

1. **Stale response after in-flight input changes.** Crop, image, source, and consent controls remain active while `loading`. Handlers clear current results but do not abort the request or bind results to a request snapshot. The old response can then be rendered with the new selected crop or image. Lock controls or implement abort/request IDs before redesign launch.
2. **Provider choice is Groq-only.** Gemini was removed from the active application at the user's request. Historical Phase 12/13 reports describe earlier states and are retained as an audit trail.
3. **Local model cannot reject unsupported input.** Any image reaching ONNX produces one of six labels. Crop restrictions only hide local option for non-Tomato/Potato selections; an arbitrary tomato photo can still be forced into a known disease class.

### Medium priority

4. **External diagnosis variability remains possible.** Groq uses temperature 0 and fixed seed 1701, with prompt instruction to choose one canonical disease or uncertain. This is best-effort; repeated provider calls may differ and disease recognition itself is not validated. Avoid claiming repeatability.
5. **Crop mismatch compares strings rather than canonical crop identities in the UI.** `assessmentCropMismatch` lowercases raw names only. Synonyms/scientific names such as “Solanum tuberosum” vs “Potato” can produce a false mismatch even where server catalogue normalization accepts them.
6. **Treatment route contracts differ by source.** Local result requires a second `/api/treatment` call; external result gets guidance in `/api/assess-plant`. A common assessment-result service exists internally, but API route contract and failure behavior differ.
7. **Groq suggestion filter is not complete semantic validation.** Prompt prohibitions and regex reduce obvious drug/dose output; a compliant-looking but unsafe/unverified instruction could pass. UI appropriately labels suggestions unverified; redesigned app must preserve that distinction and never relabel suggestions as verified treatment.
8. **No dedicated reset/start-over action.** User can clear via crop change/remove image, but this is not discoverable as a single “Check another leaf” action.
9. **Local API class ID has no provenance proof.** `/api/treatment` deterministically maps any client-supplied valid ID; strict validation blocks injected treatment text but cannot prove the browser actually ran the model. It returns only general catalogue text, not chemical prescriptions.

**Additional eligibility inconsistency:** the external `general_reference` branch checks image quality and `plant_match`, but it does not check `needs_expert_review` or compare the selected crop with `predicted_crop`. Thus a curated condition-specific reference can still appear for an expert-review result or selected/predicted crop mismatch. The material is non-chemical and cautionary, but the next implementation should gate it or make it explicitly differential-only. Evidence: `externalCropReference` and `resolveAssessmentGuidance` in `web/lib/treatment-catalog.ts`; Banana tests.

### Lower priority / operational

10. **Model route returns static hash in ETag but only checks file size.** It does not recalculate artifact hash on serving. Deployment artifact integrity must be verified by release/build process.
11. **Rate limiter is per process, not global.** Forwarded IP headers must be configured by trusted infrastructure; otherwise one client may bypass limits or share a bucket.
12. **No live Groq verification in this audit.** An earlier user-run synthetic smoke test passed, but this audit did not make a live provider request.
13. **Provider “does not store” phrasing is only about app behavior.** The app processes in memory and does not intentionally persist photos, but cannot guarantee external provider retention or processing terms.

## 8. Proposed farmer-facing behavior (not implemented)

Every desired item below is a proposal. Example text is illustrative, not existing copy.

| Scenario | Proposed behavior and sample farmer-facing text | Action |
|---|---|---|
| First visit / no crop or image | One clear sequence: “Choose a plant”, “Add a clear leaf photo”, “Choose assessment”. Button disabled until complete. | Select plant/upload photo. |
| Supported Tomato/Potato | Offer Local (photo stays on device) and Groq (external; requires consent). State local model only covers these two crops. | Pick explicit method. |
| Banana/Mango/Chilli/Other | Hide local path; state “Online assessment is needed for this plant. Groq will receive the photo only after you agree.” | Choose Groq, review notice, consent. |
| Bad file / oversized file | Inline, specific message: “Use a JPEG, PNG, or WebP image under 4 MiB.” / “This photo is over 4 MiB. Choose a smaller one.” Keep previous valid image only if clearly indicated; do not silently keep a stale preview. | Choose another photo. |
| Groq not consented | Explain “Your photo goes to Groq for analysis.” Checkbox starts unchecked. Disable analyze; on attempted submit state “Please agree before sending your photo.” | Check consent deliberately. |
| Consent revoked before submit | Keep request local; disable submission. State “Your photo has not been sent.” | Re-consent or choose local. |
| Consent revoked / fields changed after submit begins | Freeze crop/image/method controls or cancel request and ignore its response. State “Assessment cancelled because the photo or crop changed.” Never show old result beside new input. Already-sent image cannot be recalled; say so if applicable. | Submit again only after reviewing consent. |
| Local loading | Show “Checking this photo on your device…” without exposing model size/technical terms. | Wait; prevent duplicate submit. |
| Local named disease | “Possible diagnosis” then “Potato — Late blight”. Show source as “On-device assessment” and only eligible non-chemical catalogue guidance. | Review guidance/source; monitor or ask expert. |
| Local healthy class | “The model did not identify one of its supported diseases. It can still miss disease.” Never show disease medicine. | Monitor; seek advice if symptoms persist. |
| Local selected/predicted crop mismatch | Keep model result: “You selected Tomato. The model predicted Potato — Late blight. Confirm the plant before using any crop-specific guidance.” Mark any reference conditional; no chemical advice. | Confirm species with expert. |
| Local unsupported/unknown-looking image | Since no rejection policy exists, avoid claiming known diagnosis reliability. Until a validated rejection mechanism exists, pair local result with concise limitation: “The model always chooses a tomato or potato class; it may be wrong for this photo.” | Use Groq only by explicit selection/consent, or expert. |
| Local model/network/worker error | “On-device check could not start. Check your connection and try again, or choose Groq if you want an online assessment.” | Retry or deliberately switch; never auto-upload. |
| External named disease | “Possible diagnosis: Banana — Black Sigatoka” plus source “Groq visual assessment”; show uncertainty and next step. Do not transform raw provider wording into stronger certainty. | Confirm with local extension/expert. |
| External healthy | “The plant appears healthy in this photo; this does not rule out disease.” No disease treatment. | Monitor. |
| External uncertain/cannot assess | “The photo does not show enough to identify the cause. Try a close, well-lit photo of both sides of the leaf or ask an expert.” Do not invent disease. | Retake or expert. |
| External crop mismatch | “You selected Mango; Groq identified a possible Banana leaf. Confirm the plant before using the linked guidance.” Preserve external predicted crop. | Confirm species first. |
| Verified exact catalogue record | Head “Source-backed general guidance for a possible [disease]”; keep evidence links close; note “No verified chemical product or dosage is available.” | Open source/ask expert. |
| Curated general reference (e.g. broad leaf spot) | Head “Disease not confirmed”; explain links compare known conditions only. Avoid “verified treatment” heading. | Use links for comparison and seek local diagnosis. |
| No matching guidance | Keep diagnosis and Groq suggestions; show “Verified crop-specific guidance is not available for this result.” Clearly label AI suggestions as unverified; no fabricated medicine/source. | Ask local agricultural expert. |
| Provider missing, invalid, quota, timeout, unavailable | Translate to concise cause-specific copy: “Groq is not configured on this site”; “Groq is busy or has reached a usage limit”; “The request timed out.” Avoid internal codes. | Retry after wait or contact site owner; never expose keys/raw errors. |
| Retry / switch provider | One retry button; preserve same image/crop but require consent still valid only if no input changed. Switching provider clears result and consent; switching to local must never send photo. | Choose one explicit route. |
| Start another check | Provide “Check another leaf” that clears input/result/consent and scrolls focus to plant selection. | Start clean flow. |

Principles for all result states: show predicted crop/disease before technical detail; selected crop remains separate; distinguish source; one concise AI-not-diagnosis note; expert advice when concerning or uncertain; no unsupported accuracy percentage; no treatment from provider text; no chemical product/dose unless independently verified against crop-specific authoritative source and current product label.

## 9. Proposed single-page website content inventory (not implemented)

| Section | Purpose and exact content | Always/conditional | Buttons/links and destination | Existing function to preserve |
|---|---|---|---|---|
| 1. Header/navigation | Brand “AgriGuard AI”; anchors “Check a plant”, “How it works”, “Limits & privacy”, “About”. | Always. Compact on mobile. | “Check a plant” scrolls/focuses form; other items scroll to their section. | Keep assessment as central task. |
| 2. Hero | Heading “Check a plant”; simple subtext “Upload a clear leaf photo for a preliminary assessment.” Avoid accuracy claims/testimonials. | Always. | Primary “Start assessment” scrolls to form. | No unsupported promises. |
| 3. Assessment form | Crop list (Tomato, Potato, Mango, Banana, Chilli, Other); custom name only for Other; JPEG/PNG/WebP and 4 MiB note; preview/remove/replace; Local only Tomato/Potato; Groq with unchecked consent and precise disclosure. | Always; custom field, local option and external notice conditional. | “Analyze leaf”; consent required before external send. | Preserve MIME/size checks, consent server validation, external image processing disclosure. |
| 4. Inline result | Possible crop and disease; source; selected crop; crop mismatch warning; healthy/uncertain/cannot-assess copy; brief limitation. | Only after success. Error/loading replaces or precedes it. | “Check another leaf”; source links if relevant. | Keep local/external mappings separate; keep valid outside-local-class diagnosis visible. |
| 5. Guidance and next steps | Exact source-backed non-chemical catalogue text, curated general references, or explicit unavailable state. Separate AI suggestions block only for external provider. Healthy gets no disease medicine. | Conditional on result, certainty gates, and exact mapping. | Source links open official source; “Ask expert” only if a real destination/contact is configured—otherwise plain next-step text. | Keep chemicals unavailable unless verified; provider text cannot select treatment records. |
| 6. How it works | Three short steps: choose plant, add photo, choose on-device or consented online assessment. State local crop restriction and online processor. | Always, compact. | “Try a check” returns to form. | Accurately explain two separate paths. |
| 7. Limits, accuracy and privacy | Local model chooses among six Tomato/Potato classes, no unknown rejection, can be wrong; Phase 15 external test is historical Tanzania/mostly Shangi results and not Kerala assurance; online photo is sent to Groq only after consent; app does not intentionally persist it but third-party processing terms apply. | Always, expandable/details. | Official privacy/provider docs if URLs have been reviewed; no fabricated privacy guarantees. | No raw score as confidence, no accuracy guarantees. |
| 8. About AgriGuard AI | Factual short description: educational plant-image assessment prototype using traditional ML locally and optional external vision service; guidance records cite sources and are incomplete. | Always. | No invented team, affiliation, awards, or testimonials. | Set realistic scope. |
| 9. Footer/final CTA | “AI assessment, not diagnosis”; links to privacy, source references, and “Check another plant” anchor. | Always. | CTA focuses form; links only to actual pages. | Repeat privacy/diagnostic caution once, not redundantly. |

## 10. Acceptance criteria for later UI implementation

1. Every crop/method combination is explained accurately; only Tomato/Potato expose Local.
2. External consent is unchecked by default, explicitly names Groq and photo sharing, is required in browser and server, and resets on crop/image/provider change.
3. During request, the app prevents stale input/result pairing by locking inputs or cancelling/ignoring stale responses; duplicate submission is blocked.
4. Unsupported MIME, malformed image, empty file, and >4 MiB file receive actionable messages before transmission.
5. Local results use the frozen six-class mapping, show predicted crop separate from selected crop, never show decision scores as confidence, and disclose no unknown rejection.
6. External results remain independent of local class enums, accept outside-class diagnoses when valid, and preserve healthy/uncertain/cannot-assess distinctions.
7. Error UI maps all known provider categories and never exposes provider raw text, stack, key, or URL with credentials. Retry never causes implicit provider switch or duplicate call.
8. Healthy result never returns disease medicine. Catalogue result only uses server-owned exact record; mismatched/uncertain/poor/review-required cases suppress inappropriate disease-specific guidance.
9. General references are visibly differentiated from verified treatment records. Groq suggestions remain marked unverified and cannot create a product/dosage/source link.
10. No chemical recommendations appear until crop-specific disease, product/formulation, current registration/label, dosage, interval, safety/PPE and applicable restrictions have authoritative verification. Current feature should say unavailable.
11. Every non-empty source link is valid, labelled, and adjacent to the claim it supports. Unmatched custom crops do not get falsely matched generic disease guidance.
12. A single “Check another leaf” action clears all form/result/consent/error state and moves focus accessibly.
13. Keyboard/screen-reader/mobile behavior is tested; success, error, and loading updates are announced without stealing focus unexpectedly.
14. UI says the app does not intentionally persist photos but does not promise a provider's retention behavior.
15. Future verification tests cover representative state transitions, including crop/image changes during delayed requests; mocked tests do not substitute for explicit live-provider verification.

## 11. Open questions not answerable from code

- Is the project intended for Kerala specifically, India broadly, or general educational use? Catalogue applicability and translations depend on this.
- Should local model remain selectable despite historical early/late-blight errors and absence of a rejection mechanism? UI must not imply safety for field use.
- Is Groq's current model/account access expected for production? Only mocked tests and an earlier user-supplied synthetic smoke pass exist; current live disease output and repeatability were not checked.
- Do users need a second external provider in the future? The current app intentionally offers Groq only.
- What locales/languages, literacy level, and offline requirements are needed for target farmers?
- Is any agronomist/plant-pathologist review available for the existing treatment records and new crop aliases? Code sources are not equivalent to expert approval.
- Which provider data-retention terms and user privacy notice should be presented? The application code alone cannot establish provider practices.
- Is the app deployed in a serverless or persistent Node environment? That affects the in-memory limiter and file-served ONNX model route.
- Should a model class ID submitted by the browser remain sufficient to retrieve general guidance, or should a future backend validate inference itself? Current route cannot prove inference origin.
- How should users contact a qualified local expert? No verified local contact/directory is configured.
- Should the generic KAU/TNAU portals be retained? They do not necessarily discuss the particular predicted condition and can feel irrelevant.

## 12. Source files, tests, and reports inspected

### Application files

- `web/app/page.tsx`
- `web/app/assessment/assessment-form.tsx`
- `web/app/api/assess-plant/route.ts`
- `web/app/api/treatment/route.ts`
- `web/app/api/models/phase9/route.ts`
- `web/app/api/runtime/ort/[asset]/route.ts` (listed in repo; bracket path was not separately read due PowerShell path resolution)
- `web/app/real-inference/phase9-inference.ts`
- `web/app/real-inference/image.ts`
- `web/app/real-inference/phase9-features.ts`
- `web/app/real-inference/metadata.ts`
- `web/lib/assessment-schema.ts`
- `web/lib/treatment-guidance-schema.ts`
- `web/lib/disease-mapping.ts`
- `web/lib/plant-assessment.ts`
- `web/lib/groq-provider.ts`
- `web/lib/groq-request.ts`
- Gemini provider and smoke-test files were removed from the active application during the follow-up request; historical docs are retained.
- `web/lib/treatment-catalog.ts`
- `web/lib/agricultural-references.ts`
- `web/data/treatments.json` (record inventory inspected; 6 records, empty chemical options)
- `web/data/external_guidance.json`
- `web/package.json`

### Tests inspected

- `web/tests/assessment-ui.test.tsx`: initial form, preview, invalid/oversize image, consent, Groq success, Banana, uncertain, mismatch, retry, malformed response, state reset, loading, local results, healthy, six class mappings, crop mismatch, explicit external path.
- `web/tests/assess-plant.test.mjs`: schema response, key missing, Groq own key/model, invalid provider, malformed/unsafe responses, image validation/limits, consent, Banana, uncertainty, provider errors, sanitized logs, rate limit.
- `web/tests/treatment-route.test.mjs`: local mapping, external fields rejected, injected certainty/medicine/dosage rejected, healthy no disease medicine.
- `web/tests/treatment-catalog.test.mjs`: six mappings, healthy, external resolution, unsupported IDs, Banana gap, exact aliases, uncertain suppression, source integrity, prompt context.
- `web/tests/phase9-inference.test.mjs`: input shape/class map, load failure, unexpected output shape, MIME decoder cases.
- `web/tests/phase9-features.test.mjs`: HOG/HSV/combined parity and malformed RGB input.
- `web/tests/phase14-browser-check.mjs`: browser parity harness exists; not run in this task.
- `web/tests/hog.test.mjs`, `web/tests/groq-provider.test.mjs`: listed; not run in this task.

### Prior reports inspected

- `docs/phase11_treatment_audit.md`
- `docs/phase12_gemini_integration.md`
- `docs/phase13_consent_ui.md`
- `docs/phase14_local_inference.md`
- `docs/phase15_external_test_evaluation.md`
- `docs/phase16_workflow_stabilization.md`
- `docs/phase16_provider_selection.md`
- `docs/phase17_farmer_ui_and_treatments.md`
- `docs/phase17_external_crop_support_fix.md`
- `docs/phase17_guidance_expansion.md`
- `docs/groq_diagnosis_consistency.md`

Some early reports describe earlier app states; use current source as authoritative. Phase 15 report says its external test was consumed once and must remain excluded from any future tuning. No test, build, live provider request, or model pipeline was executed for the original documentation audit; focused Groq-only regression checks are run for the provider removal.
