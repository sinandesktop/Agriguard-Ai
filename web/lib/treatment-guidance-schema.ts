import { z } from "zod";

const RecordSchema = z.object({
  plant_id: z.enum(["tomato", "potato"]),
  disease_id: z.enum(["healthy", "early_blight", "late_blight"]),
  model_label: z.string().min(1).max(80),
  condition: z.string().min(1).max(180),
  diagnostic_limitations: z.array(z.string().max(240)).max(8),
  treatment_categories: z.array(z.enum(["cultural", "sanitation", "biological", "chemical"])).max(4),
  non_chemical_management: z.array(z.string().max(240)).max(10),
  chemical_options: z.array(z.never()).max(0),
  chemical_status: z.literal("no_verified_recommendation"),
  sources: z.array(z.object({ title: z.string().max(180), url: z.url(), supports: z.string().max(240) }).strict()).max(8),
}).strict();

export const TreatmentGuidanceSchema = z.discriminatedUnion("status", [
  z.object({
    status: z.literal("general_reference"),
    heading: z.string().min(1).max(100),
    message: z.string().min(1).max(320),
    steps: z.array(z.string().min(1).max(240)).max(6),
    sources: z.array(z.object({ title: z.string().min(1).max(180), url: z.url() }).strict()).max(8),
  }).strict(),
  z.object({
    status: z.literal("guidance_available"),
    guidance_kind: z.enum(["possible_disease", "healthy_prevention"]),
    conditional_on_predicted_crop: z.boolean(),
    message: z.string().min(1).max(240),
    record: RecordSchema,
  }).strict(),
  z.object({
    status: z.literal("guidance_withheld"),
    reason: z.enum(["uncertain", "crop_mismatch", "review_required", "unknown_result", "unavailable"]),
    message: z.string().min(1).max(240),
    sources: z.array(z.object({ title: z.string().min(1).max(180), url: z.url() }).strict()).max(8).optional(),
  }).strict(),
  z.object({ status: z.literal("invalid_input"), message: z.string().min(1).max(240) }).strict(),
]);

export type TreatmentGuidance = z.infer<typeof TreatmentGuidanceSchema>;
