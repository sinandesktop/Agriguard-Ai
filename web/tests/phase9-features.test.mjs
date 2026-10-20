import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { extractPhase9Features, PHASE9_FEATURE_COUNT } from "../app/real-inference/phase9-features.ts";
import { CLASS_LABELS } from "../app/real-inference/metadata.ts";

const fixtureUrl = new URL("../../ml/.dev_artifacts/phase14/browser-feature-fixtures.json", import.meta.url);

function decodePixels(fixture) {
  return new Uint8Array(Buffer.from(fixture.rgb_base64, "base64"));
}

function compare(actual, expected) {
  assert.equal(actual.length, expected.length);
  let maxAbs = 0;
  let sumAbs = 0;
  let outsideTolerance = 0;
  for (let i = 0; i < actual.length; i++) {
    const difference = Math.abs(actual[i] - expected[i]);
    maxAbs = Math.max(maxAbs, difference);
    sumAbs += difference;
    if (difference > 1e-6 + 1e-5 * Math.abs(expected[i])) outsideTolerance++;
  }
  return { maxAbs, meanAbs: sumAbs / actual.length, outsideTolerance };
}

test("Phase 9 browser features match Python HOG, HSV and concatenation on exact same RGB pixels", async () => {
  const fixture = JSON.parse(await readFile(fixtureUrl, "utf8"));
  assert.equal(fixture.entries.length, 8);
  assert.deepEqual(Object.values(fixture.class_id_to_label), [...CLASS_LABELS]);
  assert.deepEqual(fixture.onnx_interface.inputs, [{ name: "X", type: "tensor(float)", shape: [null, 8132] }]);
  assert.deepEqual(fixture.onnx_interface.outputs, [
    { name: "label", type: "tensor(int64)", shape: [null] },
    { name: "probabilities", type: "tensor(float)", shape: [null, 6] },
  ]);
  assert.equal(fixture.onnx_interface.python_onnx_prediction_matches, fixture.entries.length);
  assert.match(fixture.onnx_interface.score_output_semantics, /not probabilities/i);
  const totals = { hog: { max: 0, sum: 0, count: 0 }, hsv: { max: 0, sum: 0, count: 0 }, combined: { max: 0, sum: 0, count: 0 } };
  for (const entry of fixture.entries) {
    const actual = extractPhase9Features(decodePixels(entry), entry.width, entry.height);
    assert.equal(actual.hog.length, 8100, entry.id);
    assert.equal(actual.hsv.length, 32, entry.id);
    assert.equal(actual.combined.length, PHASE9_FEATURE_COUNT, entry.id);
    for (const [name, a, e] of [["hog", actual.hog, entry.hog], ["hsv", actual.hsv, entry.hsv], ["combined", actual.combined, [...entry.hog, ...entry.hsv]]]) {
      const stat = compare(a, e);
      totals[name].max = Math.max(totals[name].max, stat.maxAbs);
      totals[name].sum += stat.meanAbs * a.length;
      totals[name].count += a.length;
      assert.equal(stat.outsideTolerance, 0, `${entry.id} ${name} exceeded tolerance: ${JSON.stringify(stat)}`);
    }
  }
  const report = Object.fromEntries(Object.entries(totals).map(([name, value]) => [name, { maxAbs: value.max, meanAbs: value.sum / value.count }]));
  console.log(JSON.stringify({ phase9FeatureParity: report, fixtureCount: fixture.entries.length }));
});

test("malformed RGB buffers and dimensions fail before model invocation", () => {
  assert.throws(() => extractPhase9Features(new Uint8Array(3), 2, 2), /RGB pixels/);
  assert.throws(() => extractPhase9Features(new Uint8Array(12), 0, 2), /RGB pixels/);
  const invalid = Array(12).fill(0); invalid[0] = 256;
  assert.throws(() => extractPhase9Features(invalid, 2, 2), /integer values/);
});
