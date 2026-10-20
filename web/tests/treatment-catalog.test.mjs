import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { getVerifiedPromptTreatmentContext, resolveAssessmentGuidance } from "../lib/treatment-catalog.ts";
import { getPhase9ClassByLabel, getPhase9ClassResult, PHASE9_CLASS_RESULTS } from "../lib/disease-mapping.ts";

const classMap = {
  Tomato___healthy: ["tomato", "healthy"],
  Tomato___Early_blight: ["tomato", "early_blight"],
  Tomato___Late_blight: ["tomato", "late_blight"],
  Potato___healthy: ["potato", "healthy"],
  Potato___Early_blight: ["potato", "early_blight"],
  Potato___Late_blight: ["potato", "late_blight"],
};

test("all six saved Phase 9 class IDs map to exact catalogue records and readable disease results", async () => {
  const modelLabels = JSON.parse(await readFile(new URL("../../ml/models/phase9_domain_adaptation/phase9_exp_20261019_v1/class_labels.json", import.meta.url), "utf8")).class_id_to_label;
  const catalog = JSON.parse(await readFile(new URL("../data/treatments.json", import.meta.url), "utf8"));
  const catalogKeys = new Set(catalog.records.map((record) => `${record.model_label}:${record.plant_id}:${record.disease_id}`));
  for (const [classId, label] of Object.entries(modelLabels)) {
    const mapped = getPhase9ClassResult(Number(classId));
    assert.ok(mapped, `missing class ID ${classId}`);
    assert.equal(mapped.label, label);
    assert.ok(classMap[label], `unexpected model label: ${label}`);
    const [plant, disease] = classMap[label];
    assert.equal(mapped.plantId, plant);
    assert.equal(mapped.diseaseId, disease);
    assert.ok(catalogKeys.has(`${label}:${plant}:${disease}`), `missing ${label}`);
    assert.equal(resolveAssessmentGuidance({ source: "local_phase9", selected_plant_id: plant, class_id: Number(classId) }).status, "guidance_available");
  }
  assert.equal(Object.keys(modelLabels).length, 6);
  assert.equal(PHASE9_CLASS_RESULTS.length, 6);
  assert.equal(getPhase9ClassResult(100), undefined);
  assert.equal(getPhase9ClassResult(-1), undefined);
  assert.equal(getPhase9ClassResult(1.2), undefined);
  assert.equal(getPhase9ClassByLabel("similar but not exact early blight"), undefined);
});

test("healthy results return only the existing prevention record and never disease medicine", () => {
  for (const class_id of [2, 5]) {
    const result = resolveAssessmentGuidance({ source: "local_phase9", selected_plant_id: class_id === 2 ? "potato" : "tomato", class_id });
    assert.equal(result.status, "guidance_available");
    if (result.status === "guidance_available") {
      assert.equal(result.guidance_kind, "healthy_prevention");
      assert.deepEqual(result.record.chemical_options, []);
      assert.deepEqual(result.record.non_chemical_management, ["No disease medicine is indicated for a healthy classification."]);
    }
  }
});

test("external Groq uses the exact-record service, including conditional crop-mismatch guidance", () => {
  const base = { source: "groq", selected_plant_id: "potato", predicted_crop: "Potato", result_type: "disease", possible_disease: "Early blight", plant_match: "match", image_quality: "good", needs_expert_review: false };
  for (const changes of [
    { plant_match: "uncertain" },
    { image_quality: "poor" },
    { needs_expert_review: true },
    { result_type: "uncertain", possible_disease: null },
  ]) {
    const result = resolveAssessmentGuidance({ ...base, ...changes });
    assert.equal(result.status, "guidance_withheld");
    if (changes.needs_expert_review) {
      assert.equal(result.reason, "review_required");
      assert.match(result.message, /provider recommends expert confirmation/);
      assert.ok(result.sources?.length);
      assert.equal(result.sources?.[0]?.url, "https://celkau.in/ecropdoctor/vegetables/potato/earlybight.php");
    }
  }
  const mismatch = resolveAssessmentGuidance({ ...base, selected_plant_id: "tomato", plant_match: "mismatch" });
  assert.equal(mismatch.status, "guidance_available");
  if (mismatch.status === "guidance_available") {
    assert.equal(mismatch.conditional_on_predicted_crop, true);
    assert.equal(mismatch.record.model_label, "Potato___Early_blight");
    assert.deepEqual(mismatch.record.chemical_options, []);
  }
  assert.equal(resolveAssessmentGuidance({ source: "local_phase9", selected_plant_id: "tomato", class_id: 0 }).status, "guidance_available");
  assert.equal(resolveAssessmentGuidance({ ...base, source: "groq" }).status, "guidance_available");
});

test("unsupported class IDs and non-exact disease names cannot select guidance", () => {
  assert.equal(resolveAssessmentGuidance({ source: "local_phase9", selected_plant_id: "potato", class_id: 6 }).status, "invalid_input");
  assert.equal(resolveAssessmentGuidance({ source: "groq", selected_plant_id: "potato", predicted_crop: "Potato", result_type: "disease", possible_disease: "Potato___Late_blight maybe", plant_match: "match", image_quality: "good", needs_expert_review: false }).status, "guidance_withheld");
  assert.equal(resolveAssessmentGuidance({ source: "local_phase9", selected_plant_id: "mango", class_id: 0 }).status, "invalid_input");
});

test("unmatched Banana diseases retain diagnosis without invented treatment data", () => {
  const result = resolveAssessmentGuidance({ source: "groq", selected_plant_id: "banana", predicted_crop: "Banana", result_type: "disease", possible_disease: "Panama wilt", plant_match: "match", image_quality: "good", needs_expert_review: false });
  assert.equal(result.status, "guidance_withheld");
  if (result.status === "guidance_withheld") assert.match(result.message, /not available for this crop and possible disease/);
});

test("banana leaf-spot label shows comparison references without claiming a Sigatoka diagnosis", () => {
  const result = resolveAssessmentGuidance({ source: "groq", selected_plant_id: "banana", predicted_crop: "Banana", result_type: "disease", possible_disease: "banana leaf spot", plant_match: "match", image_quality: "good", needs_expert_review: true });
  assert.equal(result.status, "general_reference");
  if (result.status === "general_reference") {
    assert.match(result.message, /broad leaf-spot label/);
    assert.equal(result.sources.length, 2);
    assert.ok(result.sources.some((source) => source.title.includes("Kerala Agricultural University")));
    assert.ok(result.sources.some((source) => source.title.includes("TNAU Agritech")));
  }
});

test("source-reviewed mango and chilli disease aliases return non-chemical reference guidance", () => {
  const cases = [
    ["Mango", "mango leaf spot"],
    ["Mango", "Anthracnose (Colletotrichum gloeosporioides)"],
    ["Mango", "Powdery mildew"],
    ["Mango", "Grey blight"],
    ["Chilli", "chilli leaf spot"],
    ["Chilli", "Anthracnose"],
    ["Chilli", "Bacterial wilt (Ralstonia solanacearum)"],
    ["Chilli", "Powdery mildew"],
  ];
  for (const [crop, disease] of cases) {
    const result = resolveAssessmentGuidance({ source: "groq", selected_plant_id: crop.toLowerCase(), predicted_crop: crop, result_type: "disease", possible_disease: disease, plant_match: "match", image_quality: "good", needs_expert_review: true });
    assert.equal(result.status, "general_reference", `${crop} ${disease}`);
    if (result.status === "general_reference") {
      assert.ok(result.sources.length >= 1);
      assert.ok(result.sources.every((source) => source.url.startsWith("https://")));
        assert.doesNotMatch(JSON.stringify(result), /pesticide|dosage|active ingredient|spray \d/i);
    }
  }
});

test("nearby but unmapped external disease labels are not guessed", () => {
  const result = resolveAssessmentGuidance({ source: "groq", selected_plant_id: "mango", predicted_crop: "Mango", result_type: "disease", possible_disease: "mango leaf spot maybe anthracnose", plant_match: "match", image_quality: "good", needs_expert_review: false });
  assert.equal(result.status, "guidance_withheld");
});

test("Banana Sigatoka provides cautious non-chemical reference steps, not a prescription", () => {
  const result = resolveAssessmentGuidance({ source: "groq", selected_plant_id: "banana", predicted_crop: "Banana", result_type: "disease", possible_disease: "Black Sigatoka (Mycosphaerella musae)", plant_match: "match", image_quality: "good", needs_expert_review: true });
  assert.equal(result.status, "general_reference");
  if (result.status === "general_reference") {
    assert.match(result.message, /distinct disease/);
    assert.equal(result.steps.length, 3);
    assert.ok(result.sources.some((source) => source.url === "https://www.celkau.in/Crops/fruits/banana/leafspot.php"));
    assert.doesNotMatch(JSON.stringify(result), /dosage|spray \d|chemical product/i);
  }
});

test("uncertain Banana results never receive disease-specific guidance", () => {
  const result = resolveAssessmentGuidance({ source: "groq", selected_plant_id: "banana", predicted_crop: "Banana", result_type: "uncertain", possible_disease: null, plant_match: "match", image_quality: "good", needs_expert_review: true });
  assert.equal(result.status, "guidance_withheld");
});

test("catalogue content retains sources and contains no chemical recommendations", async () => {
  const catalogText = await readFile(new URL("../data/treatments.json", import.meta.url), "utf8");
  const catalog = JSON.parse(catalogText);
  assert.ok(catalog.records.every((record) => record.chemical_options.length === 0));
  assert.ok(catalog.records.every((record) => Array.isArray(record.treatment_categories)));
  assert.ok(catalog.records.every((record) => record.treatment_categories.every((category) => ["cultural", "sanitation", "biological", "chemical"].includes(category))));
  assert.ok(catalog.records.every((record) => record.chemical_status === "no_verified_recommendation"));
  assert.ok(catalog.records.every((record) => record.sources.length > 0));
  assert.ok(catalog.records.every((record) => record.sources.every((source) => /^https:\/\//.test(source.url))));
});

test("expanded external guidance data has unique crop-scoped aliases and direct HTTPS sources", async () => {
  const data = JSON.parse(await readFile(new URL("../data/external_guidance.json", import.meta.url), "utf8"));
  const aliases = new Set();
  for (const entry of data.entries) {
    assert.ok(["banana", "mango", "chilli"].includes(entry.crop));
    assert.ok(entry.aliases.length > 0);
    assert.ok(entry.steps.length > 0);
    assert.ok(entry.sources.length > 0);
    for (const alias of entry.aliases) {
      const key = `${entry.crop}:${alias}`;
      assert.equal(aliases.has(key), false, `duplicate alias ${key}`);
      aliases.add(key);
    }
    assert.ok(entry.sources.every((source) => /^https:\/\//.test(source.url)));
  }
});

test("provider prompt context is bounded, crop-specific, and contains no chemical prescription", () => {
  const potatoContext = getVerifiedPromptTreatmentContext("potato");
  assert.ok(potatoContext.length < 4_000);
  assert.match(potatoContext, /Potato___Early_blight/);
  assert.doesNotMatch(potatoContext, /Tomato___Early_blight/);
  assert.match(potatoContext, /no_verified_recommendation/);
  assert.doesNotMatch(potatoContext, /dosage|spray interval|active ingredient/i);
  assert.match(getVerifiedPromptTreatmentContext("mango"), /No matching verified catalogue records/);
});
