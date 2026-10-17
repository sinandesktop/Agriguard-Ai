"use client";

import { ChangeEvent, useState } from "react";
import * as ort from "onnxruntime-web";

type VectorFile = {
  feature_count: number;
  vectors: number[][];
  expected_class_ids: number[];
};

export default function FeasibilityHarness() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [vectorFile, setVectorFile] = useState<File | null>(null);
  const [status, setStatus] = useState("Choose both generated files to run inference.");
  const [results, setResults] = useState<number[] | null>(null);
  const [matches, setMatches] = useState<boolean | null>(null);

  async function runCheck() {
    if (!modelFile || !vectorFile) {
      setStatus("Select both the .onnx model and test_vectors.json file.");
      return;
    }

    setStatus("Running inference in this browser…");
    setResults(null);
    setMatches(null);
    try {
      const spec = JSON.parse(await vectorFile.text()) as VectorFile;
      if (
        !Number.isInteger(spec.feature_count) ||
        spec.vectors.length === 0 ||
        spec.vectors.some((row) => row.length !== spec.feature_count) ||
        spec.expected_class_ids.length !== spec.vectors.length
      ) {
        throw new Error("The selected vector file has an invalid shape.");
      }

      const session = await ort.InferenceSession.create(
        new Uint8Array(await modelFile.arrayBuffer()),
        { executionProviders: ["wasm"] },
      );
      const values = Float32Array.from(spec.vectors.flat());
      const input = new ort.Tensor("float32", values, [spec.vectors.length, spec.feature_count]);
      const outputMap = await session.run({ [session.inputNames[0]]: input });
      // The ONNX model emits numeric labels, though ONNX Runtime types also
      // permit string and bigint tensor data for other model schemas.
      const output = outputMap[session.outputNames[0]].data as unknown as ArrayLike<number | bigint>;
      const classIds = Array.from(output, (value) => Number(value));
      const isMatch = classIds.every((value, index) => value === spec.expected_class_ids[index]);

      setResults(classIds);
      setMatches(isMatch);
      setStatus(isMatch ? "Browser and Python class IDs match." : "Browser and Python class IDs differ.");
      await session.release();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown inference error.";
      setStatus(`Could not complete the check: ${message}`);
    }
  }

  function onModelChange(event: ChangeEvent<HTMLInputElement>) {
    setModelFile(event.target.files?.[0] ?? null);
  }

  function onVectorChange(event: ChangeEvent<HTMLInputElement>) {
    setVectorFile(event.target.files?.[0] ?? null);
  }

  return (
    <div className="harness">
      <label className="field">
        <span>ONNX model file</span>
        <input type="file" accept=".onnx,application/octet-stream" onChange={onModelChange} />
      </label>
      <label className="field">
        <span>Numeric test-vector JSON</span>
        <input type="file" accept=".json,application/json" onChange={onVectorChange} />
      </label>
      <button className="button" type="button" onClick={runCheck} disabled={!modelFile || !vectorFile}>
        Run browser inference
      </button>
      <p className="status" role="status" aria-live="polite">{status}</p>
      {matches !== null && (
        <p className={matches ? "result success" : "result failure"}>
          {matches ? "Parity check passed." : "Parity check did not match."}
        </p>
      )}
      {results && <p className="result">Browser class IDs: {results.join(", ")}</p>}
    </div>
  );
}
