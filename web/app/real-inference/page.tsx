import RealInference from "./real-inference";

export default function Page() {
  return <main className="page"><div className="eyebrow">Developer only · local browser inference</div><h1>Real image ONNX check</h1><p className="intro">Select the Phase 3 model and a PlantVillage image. Image decoding, HOG extraction and SVM inference run in this browser tab.</p><RealInference /></main>;
}
