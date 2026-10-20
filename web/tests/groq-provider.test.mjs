import assert from "node:assert/strict";
import { test } from "node:test";
import { createGroqRequestBody, GROQ_MODEL, GROQ_DETERMINISM_SEED } from "../lib/groq-request.ts";

test("Groq request uses low-variance sampling and one evidence-supported diagnosis", () => {
  const body = createGroqRequestBody("banana", "test-base64-image");
  assert.equal(body.temperature, 0);
  assert.equal(body.seed, GROQ_DETERMINISM_SEED);
  assert.equal(body.seed, 1701);
  assert.equal(body.model, GROQ_MODEL);
  assert.equal(body.model, "qwen/qwen3.8-27b");
  assert.match(body.messages[0].content, /one canonical common disease name/i);
  assert.match(body.messages[0].content, /set result_type to uncertain/i);
  assert.deepEqual(body.messages[1].content[1], {
    type: "image_url",
    image_url: { url: "data:image/jpeg;base64,test-base64-image" },
  });
});
