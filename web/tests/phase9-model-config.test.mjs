import assert from "node:assert/strict";
import { test } from "node:test";
import { isPhase9ModelAvailable, resolvePhase9ModelUrl } from "../app/real-inference/phase9-model-config.ts";

test("development uses the local model route when no asset URL is configured", () => {
  assert.equal(resolvePhase9ModelUrl(undefined, false), "/api/models/phase9");
  assert.equal(isPhase9ModelAvailable(undefined, false), true);
});

test("production requires an externally hosted model asset", () => {
  assert.equal(isPhase9ModelAvailable(undefined, true), false);
  assert.throws(() => resolvePhase9ModelUrl(undefined, true), /not configured for this deployment/);
});

test("production accepts a public HTTPS URL and rejects unsafe or malformed URLs", () => {
  assert.equal(resolvePhase9ModelUrl("https://assets.example.test/phase9.onnx", true), "https://assets.example.test/phase9.onnx");
  assert.equal(isPhase9ModelAvailable("http://assets.example.test/phase9.onnx", true), false);
  assert.equal(isPhase9ModelAvailable("https://user:password@assets.example.test/model.onnx", true), false);
  assert.equal(isPhase9ModelAvailable("not a URL", true), false);
});
