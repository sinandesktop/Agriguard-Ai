/** Single verified Phase 9 class-id mapping shared by browser UI and server policy. */
export const PHASE9_CLASS_RESULTS = [
  { classId: 0, label: "Potato___Early_blight", plantId: "potato", diseaseId: "early_blight", displayName: "Early blight", healthy: false },
  { classId: 1, label: "Potato___Late_blight", plantId: "potato", diseaseId: "late_blight", displayName: "Late blight", healthy: false },
  { classId: 2, label: "Potato___healthy", plantId: "potato", diseaseId: "healthy", displayName: "Healthy-class result", healthy: true },
  { classId: 3, label: "Tomato___Early_blight", plantId: "tomato", diseaseId: "early_blight", displayName: "Early blight", healthy: false },
  { classId: 4, label: "Tomato___Late_blight", plantId: "tomato", diseaseId: "late_blight", displayName: "Late blight", healthy: false },
  { classId: 5, label: "Tomato___healthy", plantId: "tomato", diseaseId: "healthy", displayName: "Healthy-class result", healthy: true },
] as const;

export type Phase9ClassResult = (typeof PHASE9_CLASS_RESULTS)[number];
export type SupportedCropId = "tomato" | "potato";

export const CLASS_LABELS = PHASE9_CLASS_RESULTS.map((item) => item.label) as unknown as readonly [string, ...string[]];

export function getPhase9ClassResult(classId: number): Phase9ClassResult | undefined {
  return Number.isInteger(classId) ? PHASE9_CLASS_RESULTS[classId] : undefined;
}

export function getPhase9ClassByLabel(label: string): Phase9ClassResult | undefined {
  return PHASE9_CLASS_RESULTS.find((item) => item.label === label);
}

export function formatPhase9Result(result: Phase9ClassResult): string {
  return result.healthy
    ? `The model did not identify one of its modeled diseases for ${result.plantId}. It may still miss disease.`
    : `Possible disease: ${result.displayName}`;
}
