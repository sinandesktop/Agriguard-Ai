import assert from "node:assert/strict";
import { test } from "node:test";
import { POST } from "../app/api/treatment/route.ts";

async function post(body) {
  return POST(new Request("http://localhost/api/treatment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }));
}

test("route resolves local Phase 9 IDs to exact catalogue guidance", async () => {
  const response = await post({ source: "local_phase9", selected_plant_id: "tomato", class_id: 3 });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.status, "guidance_available");
  assert.equal(body.record.model_label, "Tomato___Early_blight");
  assert.equal(body.guidance_kind, "possible_disease");
  assert.deepEqual(body.record.chemical_options, []);
});

test("public treatment route cannot trust browser-submitted external certainty or diagnoses", async () => {
  const response = await post({ source: "groq", selected_plant_id: "potato", predicted_crop: "Potato", result_type: "disease", possible_disease: "Early blight", plant_match: "match", image_quality: "good", needs_expert_review: false });
  assert.equal(response.status, 400);
  assert.equal((await response.json()).status, "invalid_input");
});

test("strict server schema rejects client certainty, medicine, and dosage injection fields", async () => {
  for (const body of [
    { source: "local_phase9", selected_plant_id: "tomato", class_id: 3, diagnosis_certain: true },
    { source: "local_phase9", selected_plant_id: "tomato", class_id: 3, medicine: "invented product", dosage: "2 ml/L" },
    { source: "groq", selected_plant_id: "tomato", predicted_crop: "Tomato", result_type: "disease", possible_disease: "Early blight", plant_match: "match", image_quality: "good", needs_expert_review: false, dosage: "2 ml/L" },
  ]) {
    const response = await post(body);
    assert.equal(response.status, 400);
    assert.equal((await response.json()).status, "invalid_input");
  }
});

test("healthy result returns prevention guidance with no disease medicine", async () => {
  const response = await post({ source: "local_phase9", selected_plant_id: "potato", class_id: 2 });
  const body = await response.json();
  assert.equal(body.status, "guidance_available");
  assert.equal(body.guidance_kind, "healthy_prevention");
  assert.deepEqual(body.record.chemical_options, []);
  assert.match(body.message, /No disease medicine is indicated/);
});
