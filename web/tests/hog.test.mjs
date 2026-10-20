import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { extractHog } from "../app/real-inference/hog.ts";
import { pillowBilinearGray } from "../app/real-inference/pillow-resize.ts";

const path = new URL("../../ml/.dev_artifacts/phase4/browser-fixtures/fixtures.json", import.meta.url);

test("browser HOG agrees with Python held-out fixtures", async () => {
  const data = JSON.parse(await readFile(path, "utf8"));
  assert.ok(data.fixtures.length >= 12, "expected at least two held-out fixtures per class");
  let maxAbsolute = 0, sumAbsolute = 0, count = 0, mismatches = 0;
  for (const fixture of data.fixtures) {
    const first = extractHog(fixture.grayscale_128);
    const again = extractHog(fixture.grayscale_128);
    assert.equal(first.length, 8100);
    assert.deepEqual(first, again, "HOG output must be deterministic");
    for (let i = 0; i < first.length; i++) {
      const delta = Math.abs(first[i] - fixture.hog_float32[i]);
      maxAbsolute = Math.max(maxAbsolute, delta); sumAbsolute += delta; count++;
      if (delta > 1e-6 + 1e-5 * Math.abs(fixture.hog_float32[i])) mismatches++;
    }
  }
  const meanAbsolute = sumAbsolute / count;
  console.log(JSON.stringify({ maxAbsolute, meanAbsolute, mismatches, featureValues: count }));
  assert.equal(mismatches, 0, `feature tolerances exceeded: max abs ${maxAbsolute}, mean abs ${meanAbsolute}`);
});

test("invalid pixel length and values fail clearly", () => {
  assert.throws(() => extractHog(new Uint8Array(4)), /128/);
  const invalid = new Float32Array(128 * 128); invalid[0] = Number.NaN;
  assert.throws(() => extractHog(invalid), /finite/);
});

test("Pillow bilinear grayscale resize agrees with held-out Python pixels", async () => {
  const data = JSON.parse(await readFile(path, "utf8"));
  let maxAbsolute = 0, sumAbsolute = 0, count = 0, different = 0;
  for (const fixture of data.fixtures) {
    const [width, height] = fixture.source_size;
    const actual = pillowBilinearGray(Uint8Array.from(fixture.grayscale_source), width, height);
    assert.equal(actual.length, 128 * 128);
    for (let i = 0; i < actual.length; i++) {
      const delta = Math.abs(actual[i] - fixture.grayscale_128[i]);
      maxAbsolute = Math.max(maxAbsolute, delta); sumAbsolute += delta; count++;
      if (delta !== 0) different++;
    }
  }
  const meanAbsolute = sumAbsolute / count;
  console.log(JSON.stringify({ resizeMaxAbsolute: maxAbsolute, resizeMeanAbsolute: meanAbsolute, resizeDifferentPixels: different, pixelCount: count }));
  assert.equal(different, 0, `Pillow resize pixel mismatch: max abs ${maxAbsolute}, mean abs ${meanAbsolute}`);
});
