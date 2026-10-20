# Online Assessment Provider Selection

## What changed

At the time of this phase, the assessment page offered separate consent-gated choices for Google Gemini and Groq. The browser sent only the selected provider ID (`gemini` or `groq`) to the same server route. Provider credentials are read on the server from `GEMINI_API_KEY` or `GROQ_API_KEY`; neither key is sent to the browser. Both keys were found configured in `web/.env.local`; their values were not read or displayed. The root `.env.example` contains empty placeholders only.

Current UI update: Gemini is no longer offered to farmers because the user reports that it is not working. Groq is the only online provider choice. The existing Gemini server adapter and mocked tests remain in the repository but are unreachable from the ordinary assessment UI; this preserves the integration for later repair without exposing a nonworking option.

## Provider requests and shared prompt

Both providers receive the same `SYSTEM_PROMPT` and the same selected-plant context produced by `buildPlantContextPrompt`. Provider-specific request formatting carries that same prompt and the normalized JPEG image. Gemini uses `gemini-3.6-flash`. Groq uses `qwen/qwen3.8-27b`, which Groq documents as a vision model with JSON Schema output support; Groq currently labels the model Preview. The Groq adapter uses strict JSON Schema output. Both responses pass through the same server-side Zod schema and safety checks before the browser receives them.

The app's treatment-guidance rules remain unchanged: generated assessment text cannot provide medicines, pesticides, product approvals, or dosages. Disease guidance still comes only from exact approved catalogue mappings and server-side eligibility checks.

## Validation error investigation

The prior `response_validation_failed` status means the provider response failed JSON/schema/length checks or the treatment-content guard. The old guard rejected common words such as “apply”, “recommendation”, or “chemical” wherever they appeared, even in non-actionable caution text. This was a confirmed false-positive risk, though the old response body was intentionally not retained, so the exact trigger for the user's specific image cannot be established. The guard is now narrower: it looks for named chemical products, dosage patterns, and action-plus-product phrasing. Sanitized server diagnostics include only a reason category (`oversized`, `malformed_json`, `schema_invalid`, or `unsafe_treatment_text`), never the response text.

The provider request timeout is now 30 seconds. No automatic provider fallback occurs; the user must choose a provider explicitly.

## Privacy and consent

The page names the selected external provider and requires consent each time the provider changes or the photo changes. The photo is processed in memory, normalized to JPEG, and sent only to the selected provider after submit. The app does not intentionally persist the upload. Provider processing is external; users should avoid photos with people or private information.

## Verification

- `npx.cmd tsx --test tests/assess-plant.test.mjs tests/assessment-ui.test.tsx` — PASS, 34 tests.
- `npm.cmd run typecheck` — PASS.
- No live Groq call was made in this follow-up. The Groq key presence was checked without reading its value; provider/account access remains unverified until an explicit Groq request is made.
- Phase 14, HOG parity, model artifacts, manifests, and evaluation reports were not run or changed.

To exercise Groq with a synthetic image after restarting the development server, set `AGRIGUARD_SMOKE_PROVIDER=groq` in the PowerShell session and run `npm.cmd run smoke:gemini` from `web/`. The opt-in smoke request can consume quota. It uses only a generated synthetic image and prints no assessment content or key.

## Official Groq references

- [Groq vision guide](https://console.groq.com/docs/vision)
- [Groq structured outputs](https://console.groq.com/docs/structured-outputs)
- [Groq Qwen 3.8 27B model page](https://console.groq.com/docs/model/qwen/qwen3.8-27b)
