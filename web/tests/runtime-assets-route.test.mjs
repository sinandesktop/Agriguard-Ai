import assert from "node:assert/strict";
import { test } from "node:test";
import { GET } from "../app/api/runtime/ort/[asset]/route.ts";

test("allowlisted ONNX Runtime Web assets are served from the installed package", async () => {
  const response = await GET(new Request("https://example.test/api/runtime/ort/ort-wasm-simd-threaded.jsep.mjs"), {
    params: Promise.resolve({ asset: "ort-wasm-simd-threaded.jsep.mjs" }),
  });
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /javascript/);
  assert.ok(Number(response.headers.get("content-length")) > 0);
  assert.ok((await response.text()).length > 1_000);
});

test("runtime asset route rejects files outside its allowlist", async () => {
  const response = await GET(new Request("https://example.test/api/runtime/ort/other.mjs"), {
    params: Promise.resolve({ asset: "other.mjs" }),
  });
  assert.equal(response.status, 404);
});
