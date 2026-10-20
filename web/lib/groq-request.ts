import { buildPlantContextPrompt, SYSTEM_PROMPT } from "./plant-assessment";

export const GROQ_MODEL = "qwen/qwen3.8-27b";
export const GROQ_DETERMINISM_SEED = 1701;

const groqSystemPrompt = `${SYSTEM_PROMPT} For consistency, use one canonical common disease name, not a list of alternatives or synonyms. Select the single best-supported disease only when visible evidence distinguishes it. If multiple diseases remain plausible from this image, set result_type to uncertain and possible_disease to null. Do not change the diagnosis merely to provide a different answer; base it on the visible features.`;

export function createGroqRequestBody(plantDescription: string, imageBase64: string) {
  return {
    model: GROQ_MODEL,
    messages: [
      { role: "system", content: groqSystemPrompt },
      {
        role: "user",
        content: [
          { type: "text", text: buildPlantContextPrompt(plantDescription) },
          { type: "image_url", image_url: { url: `data:image/jpeg;base64,${imageBase64}` } },
        ],
      },
    ],
    response_format: {
      type: "json_schema",
      json_schema: { name: "plant_assessment", strict: true, schema: groqResponseSchema },
    },
    temperature: 0,
    seed: GROQ_DETERMINISM_SEED,
    max_completion_tokens: 700,
    stream: false,
  };
}

const groqResponseSchema = {
  type: "object",
  properties: {
    plant_match: { type: "string", enum: ["match", "mismatch", "uncertain"] },
    possible_plant: { anyOf: [{ type: "string" }, { type: "null" }] },
    predicted_crop: { type: "string" },
    result_type: { type: "string", enum: ["disease", "healthy", "uncertain", "cannot_assess"] },
    possible_disease: { anyOf: [{ type: "string" }, { type: "null" }] },
    explanation: { type: "string" },
    ai_suggestions: { type: "array", items: { type: "string" }, minItems: 1, maxItems: 3 },
    image_quality: { type: "string", enum: ["good", "poor", "uncertain"] },
    visible_evidence: { type: "array", items: { type: "string" } },
    uncertainties: { type: "array", items: { type: "string" } },
    needs_expert_review: { type: "boolean" },
  },
  required: ["plant_match", "possible_plant", "predicted_crop", "result_type", "possible_disease", "explanation", "ai_suggestions", "image_quality", "visible_evidence", "uncertainties", "needs_expert_review"],
  additionalProperties: false,
} as const;
