# Phase B — Secure Gemini image assessment

Implementation date: 2026-10-19

## What was added

- `POST /api/assess-plant`, a Node.js App Router handler.
- The Google GenAI JavaScript SDK (`@google/genai`), Zod for runtime validation, and Sharp for bounded image decoding/normalization.
- Mocked endpoint tests in `web/tests/assess-plant.test.mjs`.

The implementation uses the official SDK `GoogleGenAI` client and `models.generateContent` call. At implementation time Google's model page lists `gemini-3.6-flash` as stable, accepting image input and supporting structured output. The API account's entitlement and quota are account-specific and were not checked with a live request. [Google model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash), [official structured output guide](https://ai.google.dev/gemini-api/docs/generate-content/structured-output).

## Request contract

The endpoint accepts `multipart/form-data` with:

- `plant_id`: `tomato`, `potato`, `mango`, `banana`, `chilli`, or `other`.
- `custom_plant_name`: required only for `other`, 1–80 bounded letters, numbers, spaces, apostrophes, periods, or hyphens.
- `image`: a decodable JPEG, PNG, or WebP.
- `external_consent`: must equal the literal string `true`.

Limits are 4 MiB for the image and 4 MiB plus 64 KiB for the entire multipart body. The request body is bounded while streaming, and images are decoded with a 20-megapixel limit, orientation-corrected, resized to fit within 1280×1280, and normalized to JPEG in memory before sending to the provider. Unsupported MIME types, MIME/content mismatches, corrupt images, and malformed requests are rejected before provider calls.

Success returns only `{ status: "assessment_ready", assessment }`. The validated assessment fields are `plant_match`, `possible_plant`, `possible_disease`, `image_quality`, `visible_evidence`, `uncertainties`, and `needs_expert_review`. Disease values are the six current model class labels or `unknown`. Unknown results are accepted. Uncertain/mismatched plant, poor/uncertain image quality, or unknown disease forces `needs_expert_review: true`. Additional fields, invalid enums, oversized strings/lists, or text that appears to contain chemical instructions are rejected. The endpoint does not return provider text, model metadata, scores, or confidence percentages.

## Security, privacy, and operational limits

- The server reads only `GEMINI_API_KEY`; there is no `NEXT_PUBLIC_` key. `.env.local` is ignored by `.gitignore`. The existing `.env.example` keeps an empty `GEMINI_API_KEY=` placeholder; no `.env.local` was created or read.
- Configure locally by creating `D:\PlantDiseaseAI\.env.local` if it does not exist and adding `GEMINI_API_KEY=your-replacement-key`. Do not paste the key into source, a browser variable, or Git. Restart `npm run dev` after changing it.
- Explicit consent is required in the request. The future UI must show a clear external-processing notice before setting `external_consent=true`; this API flag alone cannot prove that the user saw the notice or freely consented.
- The uploaded file and normalized buffer are processed in memory only. The code writes no image to disk and logs neither image bytes nor provider errors. Images are sent to Google only after validation and consent.
- A per-process in-memory limiter allows five requests per IP per 60 seconds. It is a small abuse guard, not global protection: serverless instances do not share this state. Production deployment should use a trusted gateway or shared rate-limit service and configure trusted client-IP headers.
- The Gemini request has a 15-second abort timeout, one SDK attempt, and sanitized handling for timeout, invalid credentials, unavailable model, quota, and other provider errors. API responses contain fixed messages and never raw exception text or stack traces.
- This endpoint performs visual triage only; it does not verify species, diagnose disease, or prescribe treatment. It is not connected to treatment lookup and does not change the catalogue.

## Tests and limits

Automated tests use an injected provider mock, so they do not need a key and do not consume API quota. A live smoke test was intentionally not run. Account-specific model availability, provider billing/quota, deployed request limits, and real-world diagnostic quality remain unverified. Consent UI and farmer-facing routing are future work.

Completed checks:

- `node --experimental-strip-types --test tests/assess-plant.test.mjs tests/treatment-catalog.test.mjs` (from `web/`): **PASS**, 14 tests, 0 failures.
- `npm.cmd run typecheck` (from `web/`): **PASS**.
- `npm.cmd run build` (from `web/`): **PASS**; Next.js lists `/api/assess-plant` as a dynamic route.
- `.env.local` ignore check: **PASS** (`.gitignore` rule `.env.*` matched it); `.env.local` was absent during the check.
