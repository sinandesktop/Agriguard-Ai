import { extractPhase9Features } from "./phase9-features";

const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<{ rgb: Uint8Array; width: number; height: number }>) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
};

workerScope.onmessage = (event: MessageEvent<{ rgb: Uint8Array; width: number; height: number }>) => {
  try {
    const { rgb, width, height } = event.data;
    const features = extractPhase9Features(rgb, width, height);
    workerScope.postMessage(
      { ok: true, hog: features.hog, hsv: features.hsv, combined: features.combined },
      [features.hog.buffer, features.hsv.buffer, features.combined.buffer],
    );
  } catch (error) {
    workerScope.postMessage({ ok: false, message: error instanceof Error ? error.message : "Feature extraction failed." });
  }
};
