import { z } from "zod";
import { resolveAssessmentGuidance } from "@/lib/treatment-catalog";

export const runtime = "nodejs";

const LocalRequestSchema = z.object({
  source: z.literal("local_phase9"),
  selected_plant_id: z.enum(["tomato", "potato"]),
  class_id: z.number().int().min(0).max(5),
}).strict();

const TreatmentRequestSchema = LocalRequestSchema;
const MAX_BODY_BYTES = 16 * 1024;

async function readBoundedJson(request: Request): Promise<unknown | null | "too_large"> {
  const length = Number(request.headers.get("content-length"));
  if (Number.isFinite(length) && length > MAX_BODY_BYTES) return "too_large";
  const reader = request.body?.getReader();
  if (!reader) return null;
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY_BYTES) {
        await reader.cancel();
        return "too_large";
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  try { return JSON.parse(new TextDecoder().decode(bytes)); } catch { return null; }
}

export async function POST(request: Request) {
  if (!request.headers.get("content-type")?.toLowerCase().startsWith("application/json")) {
    return Response.json({ status: "invalid_input", message: "Request body must be JSON." }, { status: 400 });
  }
  const body = await readBoundedJson(request);
  if (body === "too_large") return Response.json({ status: "invalid_input", message: "Request body is too large." }, { status: 413 });
  const parsed = TreatmentRequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json({ status: "invalid_input", message: "Assessment data is invalid or contains unsupported fields." }, { status: 400 });
  }
  const result = resolveAssessmentGuidance(parsed.data);
  const statusCode = result.status === "invalid_input" ? 400 : 200;
  return Response.json(result, { status: statusCode, headers: { "Cache-Control": "no-store" } });
}
