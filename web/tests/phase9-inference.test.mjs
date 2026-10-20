import assert from "node:assert/strict";
import { test } from "node:test";
import * as ort from "onnxruntime-web";
import { runPhase9Inference } from "../app/real-inference/phase9-inference.ts";
import { CLASS_LABELS } from "../app/real-inference/metadata.ts";
import { imageToRgb } from "../app/real-inference/image.ts";

const fakeImage = new File(["local image bytes"], "leaf.png", { type: "image/png" });

test("Phase 9 local inference uses float32 [1, 8132] and the six-class label map", async () => {
  const features = new Float32Array(8132);
  let callCount = 0;
  const prediction = await runPhase9Inference(fakeImage, {
    decode: async () => ({ rgb: new Uint8Array([0, 0, 0]), width: 1, height: 1 }),
    extract: async () => features,
    loadSession: async () => ({
      inputNames: ["X"],
      outputNames: ["label", "probabilities"],
      run: async (inputs) => {
        callCount++;
        assert.equal(inputs.X.type, "float32");
        assert.deepEqual(inputs.X.dims, [1, 8132]);
        assert.equal(inputs.X.data, features);
        return {
          label: new ort.Tensor("int64", BigInt64Array.from([4n]), [1]),
          probabilities: new ort.Tensor("float32", Float32Array.from([0, 1, 2, 3, 5, 4]), [1, 6]),
        };
      },
    }),
  });
  assert.equal(callCount, 1);
  assert.equal(prediction.classId, 4);
  assert.equal(prediction.label, CLASS_LABELS[4]);
});

test("local model loading failure is surfaced and does not invoke an external request", async () => {
  let fetchCalls = 0;
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => { fetchCalls++; throw new Error("unexpected network use"); };
  try {
    await assert.rejects(runPhase9Inference(fakeImage, {
      decode: async () => ({ rgb: new Uint8Array([0, 0, 0]), width: 1, height: 1 }),
      extract: async () => new Float32Array(8132),
      loadSession: async () => { throw new Error("model unavailable"); },
    }), /model unavailable/);
    assert.equal(fetchCalls, 0);
  } finally { globalThis.fetch = previousFetch; }
});

test("unexpected ONNX output shape is rejected", async () => {
  await assert.rejects(runPhase9Inference(fakeImage, {
    decode: async () => ({ rgb: new Uint8Array([0, 0, 0]), width: 1, height: 1 }),
    extract: async () => new Float32Array(8132),
    loadSession: async () => ({
      inputNames: ["X"], outputNames: ["label", "probabilities"],
      run: async () => ({ label: new ort.Tensor("int64", BigInt64Array.from([0n]), [1]), probabilities: new ort.Tensor("float32", new Float32Array(5), [1, 5]) }),
    }),
  }), /unexpected shape/);
});

test("JPEG, PNG, and WebP are accepted by the bounded browser decoder contract", async () => {
  const oldBitmap = globalThis.createImageBitmap;
  const oldDocument = globalThis.document;
  globalThis.createImageBitmap = async (_file, options) => {
    assert.equal(options.imageOrientation, "none");
    return { width: 2, height: 1, close() {} };
  };
  globalThis.document = { createElement: () => ({
    width: 0, height: 0,
    getContext: () => ({ drawImage() {}, getImageData: () => ({ data: Uint8ClampedArray.from([255, 0, 0, 255, 0, 255, 0, 255]) }) }),
  }) };
  try {
    for (const type of ["image/jpeg", "image/png", "image/webp"]) {
      const decoded = await imageToRgb(new File(["image"], `test.${type.slice(6)}`, { type }));
      assert.equal(decoded.width, 2);
      assert.equal(decoded.height, 1);
      assert.deepEqual([...decoded.rgb], [255, 0, 0, 0, 255, 0]);
    }
    await assert.rejects(imageToRgb(new File(["bad"], "bad.gif", { type: "image/gif" })), /JPEG, PNG, or WebP/);
  } finally {
    if (oldBitmap) globalThis.createImageBitmap = oldBitmap; else delete globalThis.createImageBitmap;
    if (oldDocument) globalThis.document = oldDocument; else delete globalThis.document;
  }
});
