"use client";

import { ChangeEvent, useState } from "react";
import * as ort from "onnxruntime-web";
import { extractHog } from "./hog";
import { imageToGrayscale } from "./image";
import { CLASS_LABELS, PREPROCESSING } from "./metadata";

type ReferenceFixture = { source: string; source_sha256: string; source_size: [number, number]; label: string; predicted_class_id: number; grayscale_source: ArrayLike<number>; grayscale_128: ArrayLike<number>; hog_float32: number[]; svm_decision_values: number[] };
type FixtureFile = { fixtures: ReferenceFixture[] };

export default function RealInference() {
  const [model, setModel] = useState<File | null>(null);
  const [photo, setPhoto] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [result, setResult] = useState("");
  const [busy, setBusy] = useState(false);
  async function run() {
    if (!model || !photo) return;
    setBusy(true); setResult("Processing locally in the browser...");
    try {
      const processed = await imageToGrayscale(photo);
      const pixels = processed.pixels;
      const features = extractHog(pixels);
      const session = await ort.InferenceSession.create(new Uint8Array(await model.arrayBuffer()), { executionProviders: ["wasm"] });
      try {
        if (session.inputNames.length !== 1 || session.inputNames[0] !== "X" || session.outputNames.length !== 2 || session.outputNames[0] !== "label" || session.outputNames[1] !== "probabilities") throw new Error("Unexpected ONNX model schema.");
        const outputs = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", features, [1, features.length]) });
        const ids = outputs[session.outputNames[0]].data as ArrayLike<number | bigint>;
        const scores = outputs[session.outputNames[1]].data as ArrayLike<number>;
        const classId = Number(ids[0]);
        if (!Number.isInteger(classId) || classId < 0 || classId >= CLASS_LABELS.length || scores.length !== CLASS_LABELS.length) throw new Error("Model output did not match the saved six-class mapping.");
        const ordered = Array.from(scores, (score, id) => ({ id, score: Number(score) })).sort((a, b) => b.score - a.score);
        if (ordered[0].id !== classId) throw new Error("The model label and decision-score order disagree.");
        const margin = ordered[0].score - ordered[1].score;
        let diagnostic = "";
        if (referenceFile) {
          const fileHash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", await photo.arrayBuffer())), (byte) => byte.toString(16).padStart(2, "0")).join("");
          const reference = (JSON.parse(await referenceFile.text()) as FixtureFile).fixtures.find((item) => item.source_sha256 === fileHash);
          if (reference) {
            const stats = (actual: ArrayLike<number>, expected: ArrayLike<number>, floor: number) => {
              let maxAbs = 0, sumAbs = 0, maxRel = 0, sumRel = 0;
              for (let i = 0; i < actual.length; i++) {
                const abs = Math.abs(Number(actual[i]) - Number(expected[i]));
                const rel = abs / Math.max(Math.abs(Number(expected[i])), floor);
                maxAbs = Math.max(maxAbs, abs); sumAbs += abs; maxRel = Math.max(maxRel, rel); sumRel += rel;
              }
              return `abs max/mean ${maxAbs.toExponential(3)}/${(sumAbs / actual.length).toExponential(3)}; relative max/mean ${(maxRel * 100).toPrecision(3)}%/${(sumRel / actual.length * 100).toPrecision(3)}%`;
            };
            const sourceDiff = processed.sourceWidth === reference.source_size[0] && processed.sourceHeight === reference.source_size[1]
              ? stats(processed.sourceGray, reference.grayscale_source, 1)
              : `dimension mismatch: browser ${processed.sourceWidth} x ${processed.sourceHeight}, reference ${reference.source_size[0]} x ${reference.source_size[1]}`;
            diagnostic = `\nExact fixture source: ${reference.source}\nPython reference (${reference.label}): saved prediction ${CLASS_LABELS[reference.predicted_class_id]}\nDecoded source grayscale (floor 1): ${sourceDiff}\nResized grayscale pixels (floor 1): ${stats(pixels, reference.grayscale_128, 1)}\nHOG (floor 1e-6): ${stats(features, reference.hog_float32, 1e-6)}\nDecision values (floor 1e-6): ${stats(scores, reference.svm_decision_values, 1e-6)}\nThese are measured differences, not a pass/fail threshold.`;
          } else diagnostic = "\nNo Python reference entry has this exact image SHA-256. No fixture comparison was made.";
        }
        setResult(`${CLASS_LABELS[classId]}\nTop SVM decision value: ${ordered[0].score.toFixed(4)}; top-two gap: ${margin.toFixed(4)} (uncalibrated; not a probability or confidence).${diagnostic}`);
      } finally { await session.release(); }
    } catch (error) { setResult(`Could not complete inference: ${error instanceof Error ? error.message : String(error)}`); }
    finally { setBusy(false); }
  }
  return <section className="harness">
    <label className="field"><span>Phase 3 ONNX model</span><input type="file" accept=".onnx,application/octet-stream" onChange={(e: ChangeEvent<HTMLInputElement>) => setModel(e.target.files?.[0] ?? null)} /></label>
    <label className="field"><span>PlantVillage image</span><input type="file" accept="image/jpeg,image/png,image/bmp,image/webp" onChange={(e: ChangeEvent<HTMLInputElement>) => setPhoto(e.target.files?.[0] ?? null)} /></label>
    <label className="field"><span>Optional Python reference fixtures.json</span><input type="file" accept=".json,application/json" onChange={(e: ChangeEvent<HTMLInputElement>) => setReferenceFile(e.target.files?.[0] ?? null)} /></label>
    <button className="button" onClick={run} disabled={!model || !photo || busy}>{busy ? "Running..." : "Run local inference"}</button>
    <p className="status" role="status" aria-live="polite" style={{ whiteSpace: "pre-line" }}>{result || "No image is uploaded. Selected files are read locally by this page."}</p>
    <p className="intro">Preprocessing: {PREPROCESSING} Canvas resizing is browser dependent, so compare its grayscale pixels and HOG values with the included Python reference fixtures before interpreting parity.</p>
    <p className="intro">PlantVillage has controlled backgrounds. This prediction is a model output, not a verified diagnosis. Decision values are uncalibrated and are not probabilities.</p>
  </section>;
}
