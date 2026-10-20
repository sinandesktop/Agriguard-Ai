import assert from "node:assert/strict";
import sharp from "sharp";
import { test } from "node:test";
import { createAssessmentHandler, MAX_IMAGE_BYTES } from "../lib/plant-assessment.ts";

const validAssessment = {
  plant_match: "match",
  possible_plant: "Solanum tuberosum",
  predicted_crop: "potato",
  result_type: "disease",
  possible_disease: "Potato___Early_blight",
  explanation: "Brown spots may fit early blight.",
  ai_suggestions: ["Take another clear photo of the whole leaf.", "Keep garden tools clean."],
  image_quality: "good",
  visible_evidence: ["A brown leaf spot is visible."],
  uncertainties: ["A photograph cannot confirm the cause."],
  needs_expert_review: false,
};

async function validPng() {
  return sharp({ create: { width: 32, height: 24, channels: 3, background: { r: 20, g: 120, b: 30 } } }).png().toBuffer();
}

function makeHandler({ apiKey = "test-key", generate = async () => JSON.stringify(validAssessment), now } = {}) {
  return createAssessmentHandler({ getApiKey: () => apiKey, generate, now });
}

async function request(handler, { plant = "potato", provider = "groq", image = undefined, mime = "image/png", consent = "true", customName } = {}) {
  const form = new FormData();
  form.set("plant_id", plant);
  form.set("provider", provider);
  form.set("external_consent", consent);
  if (customName !== undefined) form.set("custom_plant_name", customName);
  if (image !== undefined) form.set("image", new File([image], "leaf.png", { type: mime }));
  return handler(new Request("http://localhost/api/assess-plant", {
    method: "POST",
    headers: { "x-real-ip": "192.0.2.10" },
    body: form,
  }));
}

test("returns a schema-validated response from a mocked Groq result", async () => {
  const png = await validPng();
  let received;
  const handler = makeHandler({ generate: async (input) => { received = input; return JSON.stringify(validAssessment); } });
  const response = await request(handler, { image: png });
  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.status, "assessment_ready");
  assert.deepEqual(payload.assessment, validAssessment);
  assert.equal(payload.guidance.status, "guidance_available");
  assert.equal(payload.guidance.record.model_label, "Potato___Early_blight");
  assert.equal(received.provider, "groq");
  assert.equal(received.model, "qwen/qwen3.8-27b");
  assert.equal(received.plantDescription, "potato");
  assert.equal(received.imageBytes.length > 0, true);
  assert.equal("apiKey" in payload, false);
});

test("missing Groq API key returns a safe configuration response without calling the provider", async () => {
  let called = false;
  const handler = makeHandler({ apiKey: "", generate: async () => { called = true; return "{}"; } });
  const response = await request(handler, { image: await validPng() });
  const text = await response.text();
  assert.equal(response.status, 503);
  assert.match(text, /GROQ_API_KEY/);
  assert.equal(called, false);
  assert.doesNotMatch(text, /test-key|stack|trace/i);
});

test("Groq selection uses its own server key and fixed vision model", async () => {
  let received;
  const handler = createAssessmentHandler({
    getApiKey: () => "groq-test-key",
    generate: async (input) => { received = input; return JSON.stringify(validAssessment); },
  });
  const response = await request(handler, { provider: "groq", image: await validPng() });
  assert.equal(response.status, 200);
  assert.equal(received.provider, "groq");
  assert.equal(received.apiKey, "groq-test-key");
  assert.equal(received.model, "qwen/qwen3.8-27b");
  assert.doesNotMatch(await response.text(), /groq-test-key/);
});

test("invalid provider selection is rejected before reading provider credentials", async () => {
  let keyReads = 0;
  const handler = createAssessmentHandler({ getApiKey: () => { keyReads++; return "secret"; }, generate: async () => "{}" });
  const response = await request(handler, { provider: "unknown", image: await validPng() });
  assert.equal(response.status, 400);
  assert.equal((await response.json()).status, "invalid_provider");
  assert.equal(keyReads, 0);
});

test("empty, malformed JSON, and schema-invalid provider output get distinct safe categories", async () => {
  for (const [output, category] of [
    [undefined, "empty_provider_response"],
    ["", "empty_provider_response"],
    ["not json", "response_validation_failed"],
    [JSON.stringify({ ...validAssessment, result_type: "disease", possible_disease: null }), "response_validation_failed"],
    [JSON.stringify({ ...validAssessment, visible_evidence: ["Apply 2 ml/L of a fungicide."] }), "response_validation_failed"],
    [JSON.stringify({ ...validAssessment, explanation: "Spray 2 ml/L of a fungicide." }), "response_validation_failed"],
    [JSON.stringify({ ...validAssessment, ai_suggestions: ["Spray fungicide every week."] }), "response_validation_failed"],
    [JSON.stringify(Object.fromEntries(Object.entries(validAssessment).filter(([key]) => key !== "ai_suggestions"))), "response_validation_failed"],
  ]) {
    const response = await request(makeHandler({ generate: async () => output }), { image: await validPng() });
    assert.equal(response.status, 502);
    assert.equal((await response.json()).status, category);
  }
});

test("non-actionable caution about medicines is not mistaken for treatment instructions", async () => {
  const cautious = { ...validAssessment, visible_evidence: ["The image alone cannot support medicine or pesticide recommendations."] };
  const response = await request(makeHandler({ generate: async () => JSON.stringify(cautious) }), { image: await validPng() });
  assert.equal(response.status, 200);
  assert.deepEqual((await response.json()).assessment.visible_evidence, cautious.visible_evidence);
});

test("rejects unsupported and malformed image types before provider use", async () => {
  const handler = makeHandler({ generate: async () => { throw new Error("must not call provider"); } });
  const unsupported = await request(handler, { image: Buffer.from("plain text"), mime: "text/plain" });
  assert.equal(unsupported.status, 415);
  const malformed = await request(handler, { image: Buffer.from("not an image") });
  assert.equal(malformed.status, 415);
});

test("rejects images and complete request bodies above their limits", async () => {
  const tooLargeFile = await request(makeHandler(), { image: Buffer.alloc(MAX_IMAGE_BYTES + 1) });
  assert.equal(tooLargeFile.status, 413);

  const oversizedBody = new Request("http://localhost/api/assess-plant", {
    method: "POST",
    headers: { "content-type": "multipart/form-data; boundary=oversize", "x-real-ip": "192.0.2.20" },
    body: new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(MAX_IMAGE_BYTES + 65_537)); controller.close(); } }),
    duplex: "half",
  });
  const bounded = await makeHandler()(oversizedBody);
  assert.equal(bounded.status, 413);
});

test("rejects invalid plants and requires explicit external-processing consent", async () => {
  const invalidPlant = await request(makeHandler(), { plant: "unsupported-crop", image: await validPng() });
  assert.equal(invalidPlant.status, 400);
  assert.equal((await invalidPlant.json()).status, "invalid_plant");
  const noConsent = await request(makeHandler(), { image: await validPng(), consent: "false" });
  assert.equal(noConsent.status, 403);
  assert.equal((await noConsent.json()).status, "consent_required");
});

test("Groq may return a validated Banana disease outside the local model classes", async () => {
  const banana = { ...validAssessment, possible_plant: "banana", predicted_crop: "Banana", possible_disease: "Sigatoka leaf spot", explanation: "Possible leaf lesions are visible.", plant_match: "match" };
  const response = await request(makeHandler({ generate: async () => JSON.stringify(banana) }), { plant: "banana", image: await validPng() });
  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.assessment.possible_disease, "Sigatoka leaf spot");
  assert.equal(payload.guidance.status, "general_reference");
  assert.equal(payload.guidance.steps.length, 3);
  assert.ok(payload.guidance.sources.some((source) => source.url.includes("celkau.in/Crops/fruits/banana/leafspot.php")));
  assert.doesNotMatch(JSON.stringify(payload.guidance), /dosage|spray|pesticide|chemical product/i);
});

test("uncertain Banana result remains uncertain without a fabricated disease", async () => {
  const output = { ...validAssessment, plant_match: "match", possible_plant: "banana", predicted_crop: "Banana", result_type: "uncertain", possible_disease: null, explanation: "The plant appears to be banana, but disease is unclear.", needs_expert_review: false };
  const response = await request(makeHandler({ generate: async () => JSON.stringify(output) }), { plant: "banana", image: await validPng() });
  assert.equal(response.status, 200);
  const payload = await response.json();
  const assessment = payload.assessment;
  assert.equal(assessment.result_type, "uncertain");
  assert.equal(assessment.possible_disease, null);
  assert.equal(assessment.needs_expert_review, true);
  assert.equal(payload.guidance.status, "guidance_withheld");
});

test("timeouts and provider errors return sanitized status messages", async () => {
  const timeout = Object.assign(new Error("private provider detail"), { name: "TimeoutError" });
  const quota = Object.assign(new Error("private token marker"), { status: 429 });
  const modelMissing = Object.assign(new Error("private model detail"), { status: 404 });
  const invalidCredentials = Object.assign(new Error("private credential detail"), { status: 403 });
  const invalidRequest = Object.assign(new Error("private structured schema detail"), { status: 400 });
  const unavailable = Object.assign(new Error("private gateway body"), { status: 503 });
  const networkReset = Object.assign(new Error("private network detail"), { code: "ECONNRESET" });
  const networkTimeout = Object.assign(new Error("private timeout detail"), { code: "ETIMEDOUT" });
  for (const [error, status, code] of [
    [timeout, 504, "provider_timeout"],
    [quota, 503, "provider_quota"],
    [modelMissing, 503, "model_unavailable"],
    [invalidCredentials, 503, "provider_credentials"],
    [invalidRequest, 502, "provider_invalid_request"],
    [unavailable, 503, "provider_unavailable"],
    [networkReset, 503, "provider_unavailable"],
    [networkTimeout, 504, "provider_timeout"],
  ]) {
    const response = await request(makeHandler({ generate: async () => { throw error; } }), { image: await validPng() });
    const text = await response.text();
    assert.equal(response.status, status);
    assert.match(text, new RegExp(code));
    assert.doesNotMatch(text, /private provider detail|private token marker|test-key|stack|trace/i);
  }
});

test("provider diagnostic logs contain only sanitized category, status and error name", async () => {
  const previousWarn = console.warn;
  const diagnostics = [];
  console.warn = (...parts) => diagnostics.push(parts);
  try {
    const privateError = Object.assign(new Error("raw provider response with secret=never-log"), { status: 401, name: "ApiError" });
    const response = await request(makeHandler({ generate: async () => { throw privateError; } }), { image: await validPng() });
    assert.equal(response.status, 503);
    assert.equal((await response.json()).status, "provider_credentials");
  } finally {
    console.warn = previousWarn;
  }
  const logged = JSON.stringify(diagnostics);
  assert.match(logged, /provider_credentials/);
  assert.doesNotMatch(logged, /raw provider response|secret=never-log|test-key/);
});

test("rate limit is enforced within one handler instance", async () => {
  const handler = makeHandler();
  const emptyForm = () => {
    const form = new FormData();
    form.set("plant_id", "tomato");
    form.set("external_consent", "true");
    return form;
  };
  for (let i = 0; i < 5; i++) {
    const response = await handler(new Request("http://localhost/api/assess-plant", { method: "POST", headers: { "x-real-ip": "192.0.2.30" }, body: emptyForm() }));
    assert.equal(response.status, 400);
  }
  const limited = await handler(new Request("http://localhost/api/assess-plant", { method: "POST", headers: { "x-real-ip": "192.0.2.30" }, body: emptyForm() }));
  assert.equal(limited.status, 429);
});
