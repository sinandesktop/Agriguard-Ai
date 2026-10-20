# Phase 17 guidance expansion

## What changed

The verified tomato/potato catalogue remains unchanged. A separate curated reference set now covers exact aliases for selected Banana, Mango, and Chilli conditions. It returns source-backed, non-chemical information and disease-specific source links. Broad leaf-spot labels return comparison pages and explicitly state that the cause is not confirmed.

Covered external labels:

- **Banana:** Sigatoka leaf spot, Black Sigatoka (with pathogen caveat), and broad banana leaf spot.
- **Mango:** anthracnose, powdery mildew, grey blight, and broad mango leaf spot.
- **Chilli:** anthracnose, bacterial wilt, powdery mildew, and broad chilli leaf spot.

The entries are in [`external_guidance.json`](../web/data/external_guidance.json). The server uses exact crop-scoped alias matching in [`treatment-catalog.ts`](../web/lib/treatment-catalog.ts). A near-match does not select a record. Groq's suggestions remain visible separately as unverified AI output.

## Sources checked

- KAU Banana Sigatoka: https://www.celkau.in/Crops/fruits/banana/leafspot.php
- TNAU Banana yellow Sigatoka: https://agritech.tnau.ac.in/crop_protection/banana_diseases/crop_prot_crop%20diseases_fruits_banana_8.html
- KAU Mango anthracnose: https://www.celkau.in/ecropdoctor/fruits/Mango/anthracnose.php
- TNAU Diseases of Mango and their Management: https://agritech.tnau.ac.in/pdf/9.pdf
- KAU Mango powdery mildew: https://www.celkau.in/Crops/fruits/Mango/mildew.php
- TNAU Mango powdery mildew: https://agritech.tnau.ac.in/crop_protection/mango_2.html
- KAU Mango grey blight: https://www.celkau.in/Crops/fruits/Mango/blight.php
- KAU Chilli anthracnose: https://celkau.in/crops/vegetables/chilli/anthracnose.php
- TNAU Chilli bacterial leaf spot: https://agritech.tnau.ac.in/crop_protection/chilli_diseases_4.html
- TNAU Chilli Cercospora leaf spot: https://agritech.tnau.ac.in/crop_protection/chilli_diseases_5.html
- KAU Chilli bacterial wilt: https://www.celkau.in/ecropdoctor/vegetables/chilli/wilt.php
- KAU Chilli powdery mildew: https://celkau.in/crops/vegetables/chilli/mildew.php
- TNAU Chilli powdery mildew: https://www.agritech.tnau.ac.in/crop_protection/chilli_diseases_3.html

The pages can contain chemical-control tables. AgriGuard adopts none of those chemical instructions here. It shows only selected non-chemical information described in each record, with attribution. Chemical products, registration, dosage, intervals, and safety directions remain unavailable.

## Limits

This is an initial curated set, not coverage for every disease, plant variety, or custom plant name. An unfamiliar Groq disease label remains unmatched unless an explicit reviewed alias is added. For a broad label such as `banana leaf spot`, links are presented for comparison only; the app does not claim the result is Sigatoka. Poor image quality, uncertain plant identity, or an uncertain/cannot-assess result continues to suppress disease-specific guidance.

## Verification

Automated tests verify the screenshot's `banana leaf spot` path, exact Mango and Chilli mappings, non-chemical references, HTTPS sources, and rejection of similar-but-unmapped disease strings. No live Groq request was made.
