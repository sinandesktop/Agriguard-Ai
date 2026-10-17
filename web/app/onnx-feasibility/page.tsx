import FeasibilityHarness from "./harness";

export default function OnnxFeasibilityPage() {
  return (
    <main className="shell">
      <section className="panel">
        <p className="eyebrow">Developer-only · synthetic numeric data</p>
        <h1>Browser ONNX feasibility check</h1>
        <p className="intro">
          Select the ONNX model and test-vector JSON created by the Python
          feasibility script. This page displays integer class IDs only.
        </p>
        <FeasibilityHarness />
      </section>
    </main>
  );
}
