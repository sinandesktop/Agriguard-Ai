import sharp from "sharp";
import { z } from "zod";
import { AssessmentSchema } from "./assessment-schema";
import type { Assessment } from "./assessment-schema";
import { getVerifiedPromptTreatmentContext, resolveAssessmentGuidance } from "./treatment-catalog";

export const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
const MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 64 * 1024;
const MAX_PIXELS = 20_000_000;
const REQUEST_WINDOW_MS = 60_000;
const REQUESTS_PER_WINDOW = 5;
const PROVIDER_TIMEOUT_MS = 30_000;
const GROQ_MODEL = "qwen/qwen3.8-27b";
export type AssessmentProvider = "groq";

const PlantIdSchema = z.enum(["tomato", "potato", "mango", "banana", "chilli", "other"]);
export type AssessmentPlant = z.infer<typeof PlantIdSchema>;

type ProviderInput = {
  provider: AssessmentProvider;
  apiKey: string;
  model: string;
  plantDescription: string;
  imageBytes: Buffer;
  signal: AbortSignal;
};

type Dependencies = {
  getApiKey: (provider: AssessmentProvider) => string | undefined;
  generate: (input: ProviderInput) => Promise<string | undefined>;
  now?: () => number;
};

type Bucket = { startedAt: number; count: number };

function json(status: number, body: Record<string, unknown>): Response {
  return Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
}

function getClientKey(request: Request): string {
  const realIp = request.headers.get("x-real-ip")?.trim();
  if (realIp && realIp.length <= 100) return realIp;
  const forwarded = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  return forwarded && forwarded.length <= 100 ? forwarded : "unknown-client";
}

function rateLimited(buckets: Map<string, Bucket>, request: Request, now: number): boolean {
  const key = getClientKey(request);
  const current = buckets.get(key);
  if (!current || now - current.startedAt >= REQUEST_WINDOW_MS) {
    buckets.set(key, { startedAt: now, count: 1 });
    return false;
  }
  if (current.count >= REQUESTS_PER_WINDOW) return true;
  current.count += 1;
  return false;
}

async function readBoundedBody(request: Request): Promise<Uint8Array | null> {
  const reader = request.body?.getReader();
  if (!reader) return new Uint8Array();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > MAX_REQUEST_BYTES) {
        await reader.cancel();
        return null;
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const body = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

async function readForm(request: Request): Promise<FormData | null | "too_large"> {
  if (!request.headers.get("content-type")?.toLowerCase().startsWith("multipart/form-data;")) return null;
  const contentEncoding = request.headers.get("content-encoding");
  if (contentEncoding && contentEncoding.toLowerCase() !== "identity") return null;
  const contentLength = Number(request.headers.get("content-length"));
  if (Number.isFinite(contentLength) && contentLength > MAX_REQUEST_BYTES) return "too_large";

  const body = await readBoundedBody(request);
  if (body === null) return "too_large";
  try {
    const headers = new Headers(request.headers);
    headers.delete("content-length");
    const requestBody = new ArrayBuffer(body.byteLength);
    new Uint8Array(requestBody).set(body);
    const boundedRequest = new Request(request.url, { method: "POST", headers, body: requestBody });
    return await boundedRequest.formData();
  } catch {
    return null;
  }
}

function parsePlant(form: FormData): { plant: AssessmentPlant; description: string } | null {
  const plantParsed = PlantIdSchema.safeParse(form.get("plant_id"));
  if (!plantParsed.success) return null;
  const plant = plantParsed.data;
  const customName = form.get("custom_plant_name");
  if (plant === "other") {
    if (typeof customName !== "string") return null;
    const clean = customName.trim();
    if (!clean || clean.length > 80 || !/^[\p{L}\p{N} .'-]+$/u.test(clean)) return null;
    return { plant, description: clean };
  }
  if (customName !== null) return null;
  return { plant, description: plant };
}

async function normalizeImage(file: File): Promise<Buffer | null | "too_large"> {
  if (file.size < 1) return null;
  if (file.size > MAX_IMAGE_BYTES) return "too_large";
  const supported = new Set(["image/jpeg", "image/png", "image/webp"]);
  if (!supported.has(file.type.toLowerCase())) return null;
  const input = Buffer.from(await file.arrayBuffer());
  try {
    const image = sharp(input, { limitInputPixels: MAX_PIXELS, failOn: "error" });
    const metadata = await image.metadata();
    const mimeByFormat: Record<string, string> = { jpeg: "image/jpeg", png: "image/png", webp: "image/webp" };
    if (!metadata.format || mimeByFormat[metadata.format] !== file.type.toLowerCase()) return null;
    if (!metadata.width || !metadata.height || metadata.width * metadata.height > MAX_PIXELS) return null;
    return await image.rotate().resize({ width: 1280, height: 1280, fit: "inside", withoutEnlargement: true })
      .jpeg({ quality: 85 }).toBuffer();
  } catch {
    return null;
  }
}

export function classifyProviderFailure(error: unknown): { status: number; code: string; message: string } {
  const candidate = error as { name?: unknown; status?: unknown; code?: unknown } | null;
  const name = typeof candidate?.name === "string" ? candidate.name : "";
  const errorCode = typeof candidate?.code === "string" ? candidate.code : "";
  const status = Number(candidate?.status ?? candidate?.code);
  if (name === "AbortError" || name === "TimeoutError" || errorCode === "ETIMEDOUT") {
    return { status: 504, code: "provider_timeout", message: "The image assessment timed out. Please try again later." };
  }
  if (status === 404) {
    return { status: 503, code: "model_unavailable", message: "The selected online assessment model is unavailable to this API account." };
  }
  if (status === 401 || status === 403) {
    return { status: 503, code: "provider_credentials", message: "The selected online assessment provider rejected its server credentials." };
  }
  if (status === 429) {
    return { status: 503, code: "provider_quota", message: "The selected online provider returned a rate-limit or usage-limit response." };
  }
  if (status === 400 || status === 422) {
    return { status: 502, code: "provider_invalid_request", message: "The online assessment request was rejected. The service configuration may need an update." };
  }
  if ((status >= 500 && status <= 599) || name === "FetchError" || name === "NetworkError" || name === "TypeError" || ["ECONNRESET", "ENOTFOUND", "EAI_AGAIN", "ECONNREFUSED"].includes(errorCode)) {
    return { status: 503, code: "provider_unavailable", message: "The selected online provider is temporarily unavailable. Please try again later." };
  }
  return { status: 502, code: "provider_error", message: "The external image assessment could not be completed." };
}

function parseAssessment(text: string | undefined): { assessment: Assessment } | { code: "empty_provider_response" | "response_validation_failed"; reason?: "oversized" | "malformed_json" | "schema_invalid" | "unsafe_treatment_text" } {
  if (!text?.trim()) return { code: "empty_provider_response" };
  if (text.length > 8_000) return { code: "response_validation_failed", reason: "oversized" };
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { code: "response_validation_failed", reason: "malformed_json" };
  }
  const result = AssessmentSchema.safeParse(parsed);
  if (!result.success) return { code: "response_validation_failed", reason: "schema_invalid" };
  const generatedText = [result.data.possible_plant ?? "", result.data.predicted_crop, result.data.possible_disease ?? "", result.data.explanation, ...result.data.visible_evidence, ...result.data.uncertainties, ...result.data.ai_suggestions].join(" ");
  const disallowedTreatmentContent = /\b(?:mancozeb|chlorothalonil|carbendazim|copper oxychloride)\b|\b\d+(?:\.\d+)?\s*(?:g|kg|ml|l|ppm|%)\s*(?:\/\s*(?:l|ha|acre))?\b|\b(?:apply|spray|mix|use|treat)\b.{0,80}\b(?:fungicide|pesticide|insecticide|herbicide|chemical|medicine|product)\b/i;
  if (disallowedTreatmentContent.test(generatedText)) return { code: "response_validation_failed", reason: "unsafe_treatment_text" };
  return { assessment: {
    ...result.data,
    needs_expert_review: result.data.needs_expert_review || result.data.plant_match !== "match" || result.data.image_quality !== "good" || result.data.result_type === "uncertain" || result.data.result_type === "cannot_assess",
  } };
}

function responseFailure(code: "empty_provider_response" | "response_validation_failed"): { status: number; code: string; message: string } {
  const name = "Groq";
  return code === "empty_provider_response"
    ? { status: 502, code, message: `${name} returned an empty assessment. Please try again or ask an agricultural expert.` }
    : { status: 502, code, message: `${name} returned a response that could not be safely validated. Please try again or ask an agricultural expert.` };
}

export const SYSTEM_PROMPT = `Assess only what is visible in the submitted plant-leaf photograph. The selected plant is user-provided context, not proof of identity. You may identify any plant species or crop, including plants outside AgriGuard's local model classes, when the image provides enough evidence. Return predicted_crop as a concise plant name (or "unknown") and possible_plant as the visible plant name when identifiable. Set result_type to exactly one of: disease (a possible disease is visually supported), healthy (appears healthy, not proof disease-free), uncertain (plant identified but disease cannot be distinguished), or cannot_assess (plant/image cannot be assessed reliably). Include possible_disease only for result_type disease; otherwise return null. Never invent a disease to fill a field. Return a concise farmer-friendly explanation (at most 240 characters), visible evidence, uncertainties, image quality, plant-match status, and expert-review status. Separate visible evidence from inferred causes. This is a preliminary visual assessment, not a confirmed diagnosis. Always return 1 to 3 short ai_suggestions selected for this image and outcome. These are unverified AI-generated general care suggestions, not verified treatment. For a possible disease, suggest cautious, low-risk non-chemical steps only. For healthy, suggest monitoring. For uncertain or cannot_assess, suggest retaking a clear photo and seeking local agricultural advice rather than disease treatment. Never recommend medicine, pesticides, fungicides, chemical or biological products, active ingredients, formulations, dosages, intervals, recipes, or product names. Never instruct the user to destroy a plant or make an irreversible intervention. Do not invent links or claim suggestions are sourced. The server will show verified catalogue information separately when an exact record exists. Do not provide confidence percentages. Set needs_expert_review when uncertain, mismatched, low quality, or potentially serious. Return only schema fields; use no more than five evidence items and five uncertainty items (each at most 180 characters), no more than three ai_suggestions (each at most 220 characters), and possible_plant at most 80 characters.`;

export function buildPlantContextPrompt(plantDescription: string): string {
  return `Selected plant (user-provided data): ${JSON.stringify(plantDescription)}. Assess whether the visible plant matches this selection. If the selected plant is "other", the name may be inaccurate. Verified catalogue context for reference only: ${getVerifiedPromptTreatmentContext(plantDescription)}. Never return medicine, pesticide, dosage, or treatment instructions. The ai_suggestions field is a separate, unverified list of cautious non-chemical general care suggestions, not a verified treatment. The server independently resolves source-backed guidance from an exact catalogue record after validating your assessment.`;
}

export function createAssessmentHandler(dependencies: Dependencies) {
  const buckets = new Map<string, Bucket>();
  let calls = 0;
  const now = dependencies.now ?? Date.now;

  return async function handleAssessment(request: Request): Promise<Response> {
    if (request.method !== "POST") return json(405, { status: "method_not_allowed", message: "Use POST for image assessment." });
    if (rateLimited(buckets, request, now())) return json(429, { status: "rate_limited", message: "Too many requests. Please wait before trying again." });
    calls += 1;
    if (calls % 128 === 0) {
      const cutoff = now() - REQUEST_WINDOW_MS;
      for (const [key, value] of buckets) if (value.startedAt < cutoff) buckets.delete(key);
    }

    const form = await readForm(request);
    if (form === "too_large") return json(413, { status: "image_too_large", message: "The upload exceeds the 4 MiB request limit." });
    if (!form) return json(400, { status: "invalid_request", message: "Send a valid multipart form request." });
    if (form.get("external_consent") !== "true") return json(403, { status: "consent_required", message: "Explicit consent is required before sending the image to the selected online provider." });

    const provider = form.get("provider");
    if (provider !== "groq") return json(400, { status: "invalid_provider", message: "Choose Groq for online assessment." });

    const selectedPlant = parsePlant(form);
    if (!selectedPlant) return json(400, { status: "invalid_plant", message: "Choose a supported plant or provide a valid custom plant name." });
    const imagePart = form.get("image");
    if (!(imagePart instanceof File)) return json(400, { status: "invalid_image", message: "A leaf image file is required." });
    const normalizedImage = await normalizeImage(imagePart);
    if (normalizedImage === "too_large") return json(413, { status: "image_too_large", message: "The image exceeds the 4 MiB limit." });
    if (!normalizedImage) return json(415, { status: "invalid_image", message: "Use a readable JPEG, PNG, or WebP image." });

    const apiKey = dependencies.getApiKey(provider);
    if (!apiKey?.trim()) return json(503, { status: "provider_not_configured", provider, message: "Groq is not configured. Add GROQ_API_KEY to the server-only local environment and restart the web server." });

    let providerText: string | undefined;
    try {
      providerText = await dependencies.generate({
        provider,
        apiKey,
        model: GROQ_MODEL,
        plantDescription: selectedPlant.description,
        imageBytes: normalizedImage,
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
    } catch (error) {
      const failure = classifyProviderFailure(error);
      const candidate = error as { name?: unknown; status?: unknown } | null;
      const providerStatus = Number(candidate?.status);
      console.warn("[AgriGuard online assessment] provider request failed", {
        provider,
        category: failure.code,
        providerStatus: Number.isInteger(providerStatus) ? providerStatus : undefined,
        errorName: typeof candidate?.name === "string" ? candidate.name.slice(0, 60) : undefined,
      });
      return json(failure.status, { status: failure.code, message: failure.message });
    }

    const parsed = parseAssessment(providerText);
    if (!parsed || "code" in parsed) {
      const failure = responseFailure(parsed && "code" in parsed ? parsed.code : "response_validation_failed");
      console.warn("[AgriGuard online assessment] provider response rejected", { provider, category: failure.code, reason: "reason" in parsed ? parsed.reason : undefined });
      return json(failure.status, { status: failure.code, message: failure.message });
    }
    const guidance = resolveAssessmentGuidance({
      source: provider,
      selected_plant_id: selectedPlant.description,
      predicted_crop: parsed.assessment.predicted_crop,
      result_type: parsed.assessment.result_type,
      possible_disease: parsed.assessment.possible_disease,
      plant_match: parsed.assessment.plant_match,
      image_quality: parsed.assessment.image_quality,
      needs_expert_review: parsed.assessment.needs_expert_review,
    });
    return json(200, { status: "assessment_ready", assessment: parsed.assessment, guidance });
  };
}

