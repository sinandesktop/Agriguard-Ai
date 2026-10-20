import catalogJson from "../data/treatments.json" with { type: "json" };
import externalGuidanceJson from "../data/external_guidance.json" with { type: "json" };
import { getPhase9ClassResult, type SupportedCropId } from "./disease-mapping";

export const DISEASE_IDS = ["healthy", "early_blight", "late_blight"] as const;
export const PLANT_IDS = ["tomato", "potato"] as const;

export type PlantId = (typeof PLANT_IDS)[number];
export type DiseaseId = (typeof DISEASE_IDS)[number];
export type TreatmentStatus = "verified" | "needs_review" | "no_verified_recommendation";
export type TreatmentCategory = "cultural" | "sanitation" | "biological" | "chemical";

export interface ChemicalRecommendation {
  product_or_active_ingredient: string;
  formulation: string;
  dosage: string;
  application_method: string;
  interval: string;
  safety_precautions: string[];
  protective_equipment: string[];
  pre_harvest_interval: string;
  source_title: string;
  source_url: string;
  last_verified: string;
  verification_status: "verified";
}

export interface TreatmentRecord {
  plant_id: PlantId;
  disease_id: DiseaseId;
  model_label: string;
  plant_species: string;
  condition: string;
  symptoms: string[];
  diagnostic_limitations: string[];
  treatment_categories: TreatmentCategory[];
  non_chemical_management: string[];
  chemical_options: ChemicalRecommendation[];
  chemical_status: "no_verified_recommendation";
  status: TreatmentStatus;
  applicable_region: string;
  sources: { title: string; url: string; source_date: string | null; supports: string }[];
  verified_on: string;
}

const records = catalogJson.records as TreatmentRecord[];
const byKey = new Map(records.map((record) => [`${record.plant_id}:${record.disease_id}`, record]));

/** Minimal server-side reference context; provider output still cannot select or write treatment. */
export function getVerifiedPromptTreatmentContext(plantDescription: string): string {
  if (plantDescription !== "tomato" && plantDescription !== "potato") return "No matching verified catalogue records for this plant.";
  const context = records
    .filter((record) => record.plant_id === plantDescription && record.status === "verified" && record.chemical_options.length === 0)
    .map(({ model_label, condition, non_chemical_management, chemical_status, sources }) => ({
      model_label,
      condition,
      non_chemical_management,
      chemical_status,
      sources: sources.map(({ title, url }) => ({ title, url })),
    }));
  const serialized = JSON.stringify(context);
  return serialized.length <= 4_000 ? serialized : "Verified catalogue context omitted because it exceeded the prompt-size limit.";
}

export type AssessmentGuidanceRequest =
  | { source: "local_phase9"; selected_plant_id: unknown; class_id: unknown }
  | {
      source: "groq";
      selected_plant_id: unknown;
      predicted_crop: unknown;
      result_type: unknown;
      possible_disease: unknown;
      plant_match: unknown;
      image_quality: unknown;
      needs_expert_review: unknown;
    };

export type AssessmentGuidanceResult =
  | { status: "guidance_available"; guidance_kind: "possible_disease" | "healthy_prevention"; conditional_on_predicted_crop: boolean; message: string; record: Omit<Pick<TreatmentRecord, "plant_id" | "disease_id" | "model_label" | "condition" | "diagnostic_limitations" | "treatment_categories" | "non_chemical_management" | "chemical_options" | "chemical_status" | "sources">, "sources"> & { sources: { title: string; url: string; supports: string }[] } }
  | { status: "general_reference"; heading: string; message: string; steps: string[]; sources: { title: string; url: string }[] }
  | { status: "guidance_withheld"; reason: "uncertain" | "crop_mismatch" | "review_required" | "unknown_result" | "unavailable"; message: string; sources?: { title: string; url: string }[] }
  | { status: "invalid_input"; message: string };

const withhold = (reason: Extract<AssessmentGuidanceResult, { status: "guidance_withheld" }>["reason"], message: string, sources: { title: string; url: string }[] = []): AssessmentGuidanceResult => ({ status: "guidance_withheld", reason, message, ...(sources.length ? { sources } : {}) });
const safeNextStep = "Disease-specific guidance is withheld. Retake a clear photo or ask a qualified agricultural expert.";
const unavailableMessage = "Verified treatment guidance is not available for this crop and possible disease.";

function catalogCrop(value: unknown): PlantId | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  if (["tomato", "solanum lycopersicum"].includes(normalized)) return "tomato";
  if (["potato", "solanum tuberosum"].includes(normalized)) return "potato";
  return undefined;
}

function catalogDisease(crop: PlantId, value: unknown): DiseaseId | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  if (normalized === "healthy" || normalized === `${crop}___healthy`.toLowerCase()) return "healthy";
  if (["early blight", "early_blight", "early blight (alternaria solani)", `${crop}___early_blight`.toLowerCase()].includes(normalized)) return "early_blight";
  if (["late blight", "late_blight", "late blight (phytophthora infestans)", `${crop}___late_blight`.toLowerCase()].includes(normalized)) return "late_blight";
  return undefined;
}

type ExternalGuidanceEntry = (typeof externalGuidanceJson.entries)[number];
const externalGuidanceEntries = externalGuidanceJson.entries as ExternalGuidanceEntry[];

function externalCropReference(input: Extract<AssessmentGuidanceRequest, { source: "groq" }>): Extract<AssessmentGuidanceResult, { status: "general_reference" }> | undefined {
  if (typeof input.predicted_crop !== "string" || typeof input.possible_disease !== "string") return undefined;
  if (input.result_type !== "disease" || typeof input.possible_disease !== "string") return undefined;
  const cropName = input.predicted_crop.trim().toLowerCase();
  const crop = cropName === "chili" ? "chilli" : cropName;
  const disease = input.possible_disease.trim().toLowerCase();
  const entry = externalGuidanceEntries.find((candidate) => candidate.crop === crop && candidate.aliases.includes(disease));
  if (!entry) return undefined;
  return {
    status: "general_reference",
    heading: entry.heading,
    message: entry.message,
    steps: entry.steps,
    sources: entry.sources.map(({ title, url }) => ({ title, url })),
  };
}

function exactRecord(plantId: PlantId, diseaseId: DiseaseId): TreatmentRecord | undefined {
  return records.find((record) => record.plant_id === plantId && record.disease_id === diseaseId && record.status === "verified");
}

function available(plantId: PlantId, diseaseId: DiseaseId, selectedPlant: string, uncertain = false): AssessmentGuidanceResult {
  const record = exactRecord(plantId, diseaseId);
  if (!record || record.chemical_options.length !== 0 || record.chemical_status !== "no_verified_recommendation") {
    return withhold("unavailable", "Verified guidance for this possible result is unavailable. Ask a qualified agricultural expert.");
  }
  return {
    status: "guidance_available",
    guidance_kind: diseaseId === "healthy" ? "healthy_prevention" : "possible_disease",
    message: diseaseId === "healthy"
      ? "This healthy-class result is not proof the plant is disease-free. No disease medicine is indicated."
      : plantId !== selectedPlant
        ? "Conditional guidance only: use this information only if the predicted crop is confirmed. Do not apply chemical products based on this result."
        : uncertain
          ? "Cautious, source-backed non-chemical guidance for an uncertain possible disease. Confirm the crop and diagnosis before taking action."
          : "General, source-backed management guidance for a possible disease; this is not a confirmed diagnosis.",
    conditional_on_predicted_crop: plantId !== selectedPlant.trim().toLowerCase(),
    record: {
      plant_id: record.plant_id,
      disease_id: record.disease_id,
      model_label: record.model_label,
      condition: record.condition,
      diagnostic_limitations: record.diagnostic_limitations,
      non_chemical_management: record.non_chemical_management,
      treatment_categories: record.treatment_categories,
      chemical_options: [],
      chemical_status: "no_verified_recommendation",
      sources: record.sources.map(({ title, url, supports }) => ({ title, url, supports })),
    },
  };
}

/**
 * Server-side policy for mapping only exact model labels to existing catalogue records.
 * Request fields describe an assessment; they are never treated as proof or certainty.
 */
export function resolveAssessmentGuidance(input: AssessmentGuidanceRequest): AssessmentGuidanceResult {
  if (!input || typeof input !== "object") return { status: "invalid_input", message: "Assessment data is invalid." };

  if (input.source === "local_phase9") {
    if (!PLANT_IDS.includes(input.selected_plant_id as PlantId)) return { status: "invalid_input", message: "Choose a supported tomato or potato assessment." };
    const selectedPlant = input.selected_plant_id as SupportedCropId;
    if (typeof input.class_id !== "number" || !Number.isInteger(input.class_id)) {
      return { status: "invalid_input", message: "The local model class identifier is invalid." };
    }
    const result = getPhase9ClassResult(input.class_id);
    if (!result) return { status: "invalid_input", message: "The local model class identifier is unsupported." };
    return available(result.plantId, result.diseaseId, selectedPlant);
  }

  if (input.source !== "groq") return { status: "invalid_input", message: "The assessment source is unsupported." };
  if (typeof input.selected_plant_id !== "string" || input.selected_plant_id.length < 1 || input.selected_plant_id.length > 80) {
    return { status: "invalid_input", message: "The selected plant value is invalid." };
  }
  if (input.result_type === "uncertain" || input.result_type === "cannot_assess") return withhold("uncertain", safeNextStep);
  if (input.result_type !== "disease" && input.result_type !== "healthy") return withhold("unknown_result", safeNextStep);
  const generalReference = externalCropReference(input);
  if (generalReference) {
    if (input.image_quality !== "good" || input.plant_match !== "match") {
      return withhold("review_required", "The provider recommends expert confirmation before choosing disease-specific treatment.", generalReference.sources);
    }
    return generalReference;
  }
  const crop = catalogCrop(input.predicted_crop);
  if (!crop) return withhold("unavailable", unavailableMessage);
  const disease = input.result_type === "healthy" ? "healthy" : catalogDisease(crop, input.possible_disease);
  if (!disease) return withhold("unavailable", unavailableMessage);
  const record = exactRecord(crop, disease);
  const sources = input.result_type === "disease" ? record?.sources.map(({ title, url }) => ({ title, url })) ?? [] : [];
  if (input.plant_match === "uncertain") return withhold("review_required", "The plant identity is uncertain, so disease-specific guidance is withheld until it is confirmed.", sources);
  if (input.image_quality !== "good") return withhold("review_required", "The photo quality is insufficient for disease-specific guidance. Retake a clear, close-up photo.", sources);
  if (input.needs_expert_review !== false) return withhold("review_required", "The provider recommends expert confirmation before disease-specific guidance is shown.", sources);

  const selectedCrop = input.selected_plant_id.trim().toLowerCase();
  const cropMismatch = selectedCrop !== crop;
  if (input.plant_match === "mismatch" && !cropMismatch) return withhold("review_required", safeNextStep, sources);
  return available(crop, disease, selectedCrop, false);
}
