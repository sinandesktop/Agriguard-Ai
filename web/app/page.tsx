import Link from "next/link";

export default function HomePage() {
  return (
    <main className="shell">
      <section className="panel">
        <p className="eyebrow">AgriGuard AI · development foundation</p>
        <h1>Local plant assessment, built carefully.</h1>
        <p className="intro">
          The planned classifier uses handcrafted HOG features and a classical
          SVM. This first phase checks only whether a synthetic numeric SVM can
          run through ONNX in a browser.
        </p>
        <div className="notice" role="note">
          This is a developer feasibility workspace. It does not identify
          plants or diseases and does not provide agricultural advice.
        </div>
        <Link className="button" href="/onnx-feasibility">
          Open numeric ONNX check
        </Link>
      </section>
    </main>
  );
}
