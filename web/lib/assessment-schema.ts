import { z } from "zod";

/** External-provider assessment schema. It is intentionally independent of Phase 9's six-class map. */
export const AssessmentSchema = z.object({
  plant_match: z.enum(["match", "mismatch", "uncertain"]),
  possible_plant: z.string().trim().min(1).max(80).nullable(),
  predicted_crop: z.string().trim().min(1).max(80),
  result_type: z.enum(["disease", "healthy", "uncertain", "cannot_assess"]),
  possible_disease: z.string().trim().min(1).max(100).nullable(),
  explanation: z.string().trim().min(1).max(240),
  ai_suggestions: z.array(z.string().trim().min(1).max(220)).min(1).max(3),
  image_quality: z.enum(["good", "poor", "uncertain"]),
  visible_evidence: z.array(z.string().trim().min(1).max(180)).max(5),
  uncertainties: z.array(z.string().trim().min(1).max(180)).max(5),
  needs_expert_review: z.boolean(),
}).strict().superRefine((assessment, context) => {
  if (assessment.result_type === "disease" && !assessment.possible_disease) {
    context.addIssue({ code: "custom", path: ["possible_disease"], message: "A disease result requires a possible disease name." });
  }
  if (assessment.result_type !== "disease" && assessment.possible_disease !== null) {
    context.addIssue({ code: "custom", path: ["possible_disease"], message: "Only a disease result may include a disease name." });
  }
  if (assessment.result_type === "healthy" && assessment.predicted_crop.toLowerCase() === "unknown") {
    context.addIssue({ code: "custom", path: ["predicted_crop"], message: "A healthy result requires an identified plant." });
  }
});

export type Assessment = z.infer<typeof AssessmentSchema>;
