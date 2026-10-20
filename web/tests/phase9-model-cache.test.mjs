import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import { test } from "node:test";
import { loadCachedPhase9ModelSource } from "../app/real-inference/phase9-model-cache.ts";

function makeMemoryCache() {
  const records = new Map();
  return {
    records,
    storage: {
      open: async () => ({
        match: async (key) => records.get(key)?.clone(),
        put: async (key, response) => { records.set(key, response.clone()); },
        delete: async (key) => records.delete(key),
      }),
    },
  };
}

const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");

test("verified model bytes are stored and reused after a page reload", async () => {
  const bytes = new TextEncoder().encode("synthetic model bytes");
  const { storage, records } = makeMemoryCache();
  let fetchCount = 0;
  let persistCount = 0;
  const runtime = {
    caches: storage,
    subtle: webcrypto.subtle,
    persistStorage: async () => { persistCount++; return true; },
    fetch: async () => {
      fetchCount++;
      return new Response(bytes, { headers: { "content-length": String(bytes.byteLength) } });
    },
  };

  const first = await loadCachedPhase9ModelSource("https://model.test/phase9.onnx", runtime, { sizeBytes: bytes.byteLength, sha256: hash(bytes) });
  assert.equal(first.restoredFromPersistentCache, false);
  assert.equal(first.storedForNextVisit, true);
  assert.deepEqual([...first.source], [...bytes]);
  assert.equal(records.size, 1);

  const reopened = await loadCachedPhase9ModelSource("https://model.test/phase9.onnx", runtime, { sizeBytes: bytes.byteLength, sha256: hash(bytes) });
  assert.equal(reopened.restoredFromPersistentCache, true);
  assert.equal(reopened.storedForNextVisit, true);
  assert.deepEqual([...reopened.source], [...bytes]);
  assert.equal(fetchCount, 1);
  assert.equal(persistCount, 2);
});

test("a downloaded model with the wrong checksum is rejected and removed from cache", async () => {
  const bytes = new TextEncoder().encode("not the expected model");
  const { storage, records } = makeMemoryCache();
  await assert.rejects(loadCachedPhase9ModelSource("https://model.test/phase9.onnx", {
    caches: storage,
    subtle: webcrypto.subtle,
    fetch: async () => new Response(bytes, { headers: { "content-length": String(bytes.byteLength) } }),
  }, { sizeBytes: bytes.byteLength, sha256: "0".repeat(64) }), /integrity check/);
  assert.equal(records.size, 0);
});

test("unsupported persistent storage falls back to ONNX Runtime URL loading", async () => {
  const url = "https://model.test/phase9.onnx";
  const result = await loadCachedPhase9ModelSource(url, { fetch: async () => { throw new Error("must not fetch"); } });
  assert.equal(result.source, url);
  assert.equal(result.restoredFromPersistentCache, false);
  assert.equal(result.storedForNextVisit, false);
});
