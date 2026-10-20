import * as ort from "onnxruntime-web";
import { imageToRgb, type DecodedRgbImage } from "./image";
import { CLASS_LABELS } from "./metadata";
import { PHASE9_FEATURE_COUNT } from "./phase9-features";
import { configuredPhase9ModelUrl } from "./phase9-model-config";

const INPUT_NAME = "X";
const LABEL_OUTPUT = "label";
const SCORE_OUTPUT = "probabilities";

type Phase9Session = Pick<ort.InferenceSession, "inputNames" | "outputNames" | "run">;
type WorkerFeatures = { ok: true; combined: Float32Array } | { ok: false; message: string };
type Dependencies = {
  loadSession?: () => Promise<Phase9Session>;
  decode?: (file: File) => Promise<DecodedRgbImage>;
  extract?: (image: DecodedRgbImage) => Promise<Float32Array>;
};

let sharedSession: Promise<Phase9Session> | undefined;

export function resetPhase9SessionForTests(): void {
  sharedSession = undefined;
}

export function loadPhase9Session(): Promise<Phase9Session> {
  if (!sharedSession) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "/api/runtime/ort/";
    sharedSession = ort.InferenceSession.create(configuredPhase9ModelUrl(), { executionProviders: ["wasm"] }).then((session) => {
      if (session.inputNames.length !== 1 || session.inputNames[0] !== INPUT_NAME ||
          session.outputNames.length !== 2 || session.outputNames[0] !== LABEL_OUTPUT || session.outputNames[1] !== SCORE_OUTPUT) {
        void session.release();
        throw new Error("The downloaded local model does not match the verified Phase 9 ONNX interface.");
      }
      return session;
    }).catch((error: unknown) => {
      sharedSession = undefined;
      throw error;
    });
  }
  return sharedSession;
}

function extractInWorker(image: DecodedRgbImage): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(new URL("./phase9-feature.worker.ts", import.meta.url), { type: "module" });
    } catch {
      reject(new Error("This browser could not start local image processing."));
      return;
    }
    worker.onmessage = (event: MessageEvent<WorkerFeatures>) => {
      worker.terminate();
      if (!event.data.ok) reject(new Error(event.data.message));
      else resolve(event.data.combined);
    };
    worker.onerror = () => { worker.terminate(); reject(new Error("Local feature extraction failed in this browser.")); };
    worker.postMessage({ rgb: image.rgb, width: image.width, height: image.height }, [image.rgb.buffer]);
  });
}

export async function runPhase9Inference(file: File, dependencies: Dependencies = {}): Promise<{ classId: number; label: string }> {
  const modelPromise = (dependencies.loadSession ?? loadPhase9Session)();
  const decoded = await (dependencies.decode ?? imageToRgb)(file);
  const features = await (dependencies.extract ?? extractInWorker)(decoded);
  if (features.length !== PHASE9_FEATURE_COUNT || !(features instanceof Float32Array)) {
    throw new Error("Local feature extraction returned an incompatible 8,132-value vector.");
  }
  const session = await modelPromise;
  const input = new ort.Tensor("float32", features, [1, PHASE9_FEATURE_COUNT]);
  const outputs = await session.run({ [INPUT_NAME]: input });
  const labels = outputs[LABEL_OUTPUT];
  const scores = outputs[SCORE_OUTPUT];
  if (!labels || labels.dims.length !== 1 || labels.dims[0] !== 1 || labels.data.length !== 1 ||
      !scores || scores.dims.length !== 2 || scores.dims[0] !== 1 || scores.dims[1] !== CLASS_LABELS.length || scores.data.length !== CLASS_LABELS.length) {
    throw new Error("The local model returned an output with an unexpected shape.");
  }
  const classId = Number(labels.data[0]);
  if (!Number.isInteger(classId) || classId < 0 || classId >= CLASS_LABELS.length) {
    throw new Error("The local model returned a class ID outside the saved six-class mapping.");
  }
  return { classId, label: CLASS_LABELS[classId] };
}
