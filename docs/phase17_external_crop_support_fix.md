# Phase 17 focused fix: external crop support

## Root cause

The external assessment schema and both provider response schemas reused the Phase 9 local classifier's six tomato/potato labels. The shared provider prompt also instructed Gemini and Groq to answer only with those classes. The result component then mapped the provider's disease through the local class-ID label map and described unmatched plants as outside the supported classes. The treatment route accepted only tomato or potato, and the UI skipped catalogue lookup for other selected crops.

## Changes

- External provider results now have a separate schema: a bounded plant name, one of `disease`, `healthy`, `uncertain`, or `cannot_assess`, and a nullable possible disease. Disease text is required only for the disease outcome. Unknown or uncertain conditions do not get a fabricated disease.
- The Gemini and Groq prompts and structured response schemas accept plant and disease names outside the local Phase 9 classes. Both still prohibit invented product and dosage recommendations. Their responses pass the same server-side validation.
- The local Phase 9 class mapping and local inference route are unchanged. External display uses the provider's validated plant/disease fields instead of looking them up as local model classes.
- The assessment server resolves treatment immediately after validating Gemini/Groq output and returns diagnosis plus guidance. This keeps eligibility decisions server-controlled. The public treatment route is limited to local class IDs; a caller cannot submit fabricated external certainty fields to request guidance.
- Only exact existing verified catalogue records are returned, currently limited to tomato and potato. A valid Banana diagnosis remains visible while treatment lookup reports that verified guidance is unavailable. Healthy outcomes can return prevention guidance only; uncertain, unassessable, poor-quality, or review-required outcomes do not return disease-specific guidance.
- The result UI distinguishes an unclear photo or uncertain identification from a named possible disease that the provider recommends an expert confirm. In the latter case, it keeps the disease visible and explains why disease-specific guidance is withheld.

### Follow-up: avoid labeling expert review as image uncertainty

The prior UI combined `needs_expert_review`, `image_quality`, and disease uncertainty into the same sentence. Therefore, a provider response could name a likely disease and still make the UI say the image was uncertain. The UI now uses separate wording for uncertain diagnosis, provider-marked photo quality, and expert confirmation. A named possible disease remains visible. Catalogue guidance remains withheld when the provider requests expert review, and the UI explains that reason. The provider's raw structured flags were not available from the user's screenshot, so the exact flag that triggered the earlier generic warning cannot be established from that report alone.

When an exact tomato/potato disease maps to an existing catalogue record but guidance is withheld for expert review, the response now includes that record's source links as further reading. The UI clearly states that these pages may contain chemical-control information and that AgriGuard has not verified current product approvals or label directions. This does not add products or doses to AgriGuard recommendations. The linked KAU tomato early-blight page describes the disease and non-chemical management and also contains a chemical-control table; this is a reference to the source, not verification of current product use. [KAU e-Crop Doctor: Tomato early blight](https://celkau.in/ecropdoctor/vegetables/tomato/earlyblight.php).

## Treatment catalogue status

No catalogue records were added or changed. Existing tomato and potato records remain the only available source-backed guidance. Banana, Mango, Chilli, custom crops, and diseases without an exact tomato/potato catalogue match have no verified treatment guidance. The catalogue has no verified chemical product, formulation, or dosage recommendations; the UI continues to state that chemical guidance is unavailable.

## Verification

Focused mocked tests cover both provider handlers returning a Banana disease, uncertain Banana output, Banana UI rendering when guidance is absent, all six unchanged Phase 9 class mappings, local healthy behavior, conditional crop-mismatch guidance, exact catalogue resolution, strict rejection of injected treatment fields, and safe provider errors/malformed output. External-photo consent tests remain in the provider and UI suites.

Commands run:

```powershell
cd D:\PlantDiseaseAI\web
npx.cmd tsx --test tests\assess-plant.test.mjs tests\treatment-catalog.test.mjs tests\treatment-route.test.mjs tests\assessment-ui.test.tsx
npm.cmd run typecheck
```

The combined focused suite passed (50/50) after the external-crop and source-link changes. The UI/treatment subset passed (32/32) after the warning wording change. TypeScript type checking passed. No live Gemini/Groq request, Phase 14/HOG parity run, model evaluation, or Phase 15 holdout access was performed.

## Limitations

Provider-generated diagnoses for crops outside the local model remain preliminary visual assessments and are not validated disease determinations. No Banana treatment records currently exist. Live provider behavior was not tested in this fix; automated provider responses were mocked. The configured external providers may still be unavailable or may return an assessment that fails validation.

### Follow-up: source-reviewed references for Banana, Mango, and Chilli

The server now has exact, crop-scoped aliases for selected Banana, Mango, and Chilli disease names, backed by an external guidance dataset. It covers Banana Sigatoka and broad `banana leaf spot` labels; Mango anthracnose, powdery mildew, grey blight and broad `mango leaf spot`; and Chilli anthracnose, bacterial wilt, powdery mildew and broad `chilli leaf spot`. Specific aliases receive source-backed non-chemical reference steps. Broad leaf-spot labels receive differential reference links and explicitly do not get treated as a confirmed disease. The cited Kerala Agricultural University and TNAU pages describe conditions and relevant management or comparison information. The browser still displays Groq's separate suggestions as unverified.

Black Sigatoka is identified by Australia's Department of Agriculture as *Pseudocercospora fijiensis*, so the app explicitly distinguishes KAU's *Cercospora musae* Sigatoka page when Groq returns `Black Sigatoka (Mycosphaerella musae)`. None of these additions establishes a verified chemical product, formulation, dosage, or schedule; none is shown.

This is deliberately not a promise to provide disease-specific guidance for every possible image. When the crop, condition, or diagnosis has no exact reviewed mapping, the app retains the possible diagnosis and Groq suggestions, then reports that verified guidance is unavailable. It must not invent a treatment to fill a catalogue gap. Custom plant names and unlisted conditions remain unsupported by this curated guidance file.

Sources: [Kerala Agricultural University: Banana Sigatoka leaf spot](https://www.celkau.in/Crops/fruits/banana/leafspot.php); [Australian Department of Agriculture: Black Sigatoka identification](https://www.agriculture.gov.au/biosecurity/australia/naqs/naqs-target-lists/black-sigatoka).

Focused follow-up verification: the assessment/UI/catalogue/route suite and TypeScript check passed before this expanded-catalogue change. Re-run results for the latest catalogue change are recorded in the current turn's completion report. No live provider call or model evaluation was run.

### Follow-up: provider suggestions and multiple reference portals

Gemini and Groq now return a required `ai_suggestions` field. The server validates it against the structured assessment schema and the existing no-chemical/no-dosage content checks. The UI labels these suggestions **AI suggestions — not verified**, displays them separately from server-resolved catalogue guidance, and keeps consent mandatory for sending the image. Catalogue recommendations and source links remain the source-backed section; suggestions from the provider do not select or alter a catalogue record.

The farmer-facing provider selector now offers Groq only. Gemini's server adapter remains dormant and is not selectable in the UI until its live path is repaired.

The result also links to the Kerala Agricultural University Agri Infotech Portal and TNAU Agritech Portal as broader official reference portals. They are described as portals that may not cover the exact crop or condition. Disease-specific links are shown only when an exact catalogue or known general-reference mapping provides them. Provider-created URLs are never accepted.

This lets the selected provider offer cautious, non-chemical suggestions even when an exact catalogue entry is absent, while making their unverified status visible. It still cannot safely guarantee a disease-specific solution for every image; uncertain or poorly supported results should favor photo retaking and local expert confirmation. No live provider call was made during verification.
