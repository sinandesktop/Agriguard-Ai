# Phase C — Consent-aware assessment UI

Implementation date: 2026-10-19

## User flow and routing

The home page now presents a small mobile-first form. It accepts only plant IDs supported by `POST /api/assess-plant`: tomato, potato, mango, banana, chilli, and other. `other` requires a bounded custom name. The browser accepts JPEG, PNG, and WebP files up to 4 MiB, previews the selected file, and lets the user remove or replace it.

The Phase 9 HOG + HSV model is not connected to browser inference in this app. Tomato and potato selections therefore clearly show that local assessment is unavailable. The older Phase 3 HOG-only harness is not used as a fallback. Other plants are described as unsupported by the local model. For every selection, Gemini is a separate optional path; plant selection and upload alone make no network request.

Before submit, the user sees an external-processing notice and an unchecked consent checkbox. The submit control is disabled until plant, image, and consent are present. Changing the selected image or plant context clears consent. Only deliberate submit sends multipart data to `/api/assess-plant`, including the exact `external_consent=true` field. The server's own consent, validation, size, and rate-limit checks remain authoritative.

## Result handling and privacy

The browser validates response data with the same Zod schema used by the Phase B server. Unknown diseases remain unknown. Mismatch, uncertain plant match, uncertain/poor image quality, unknown disease, or expert-review flags are displayed as caution states. Fixed user-facing error text is used instead of provider messages. The UI has no connection to `/api/treatment`, displays no generated medicine/dosage, and explicitly says that the result is not a confirmed diagnosis.

No live Gemini request was made. Automated UI tests mock `fetch`. The user must still actively submit each image after seeing the notice and checking consent. This UI is not a guarantee about Google's retention or processing terms; users should avoid unnecessary personal information in images. The API's in-memory limiter remains per server process and is not a global production rate limit.

## Tests and commands

From `web/` in PowerShell:

```powershell
npm.cmd run test:phase-c
npm.cmd run test:hog
npm.cmd run typecheck
npm.cmd run build
```

`test:phase-c` runs interaction tests with a mocked assessment endpoint, followed by the existing Phase B endpoint tests and Phase A treatment-catalogue tests. No key or provider quota is needed.

## Files changed for Phase C

- `web/app/page.tsx` — mounts the assessment form as the main page.
- `web/app/layout.tsx` — sets assessment-focused metadata.
- `web/app/globals.css` — responsive styling, focus indicators, visible error/result states, and touch-sized controls; existing developer-harness styles are retained.
- `web/app/assessment/assessment-form.tsx` — form, client validation, image preview, explicit consent, server request, safe result display, and retry.
- `web/lib/assessment-schema.ts` — shared runtime response schema for server and browser.
- `web/lib/plant-assessment.ts` — reuses the shared schema without changing the Phase B validation rules.
- `web/tests/assessment-ui.test.tsx` — mocked UI interaction tests.
- `web/package.json` and `web/package-lock.json` — test command and UI-test tooling/types.
- `docs/phase13_consent_ui.md` — this phase record.

No `.env.local`, API key, model, manifest, treatment data, or historical evaluation was changed. Existing unrelated working-tree changes were preserved.

## Verification and incomplete items

Executed from `web/`:

- `npm.cmd run test:phase-c` — **PASS**, 28 tests total (14 UI interaction tests, 8 mocked Phase B endpoint tests, 6 Phase A catalogue tests), no live Gemini calls.
- `npm.cmd run test:hog` — **PASS**, 3 browser-HOG regression tests.
- `npm.cmd run typecheck` — **PASS**.
- `npm.cmd run build` — **PASS**; Next.js production build includes `/`, `/api/assess-plant`, and the existing developer inference routes.

Browser-based Phase 9 HOG + HSV inference remains unimplemented and unverified; the interface accurately presents local assessment as unavailable. A future deployment still needs a shared rate limiter and a real-browser accessibility/responsive review. Treatment authorization is intentionally out of scope and the treatment API is not called.
