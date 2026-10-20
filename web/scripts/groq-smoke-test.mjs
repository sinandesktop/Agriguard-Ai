import sharp from "sharp";
import { AssessmentSchema } from "../lib/assessment-schema.ts";

const baseUrl = (process.env.AGRIGUARD_SMOKE_BASE_URL ?? "http://127.0.0.1:3000").replace(/\/$/, "");
console.warn("This opt-in smoke test sends a synthetic image through the local assessment route to Groq and may consume API quota. No user image is used or saved.");

try {
  const syntheticJpeg = await sharp({
    create: { width: 48, height: 48, channels: 3, background: { r: 42, g: 112, b: 54 } },
  }).jpeg({ quality: 85 }).toBuffer();
  const form = new FormData();
  form.set("plant_id", "potato");
  form.set("provider", "groq");
  form.set("external_consent", "true");
  form.set("image", new Blob([syntheticJpeg], { type: "image/jpeg" }), "synthetic-smoke.jpg");

  const response = await fetch(`${baseUrl}/api/assess-plant`, {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(25_000),
  });
  let body;
  try { body = await response.json(); } catch { body = undefined; }

  if (!response.ok) {
    const category = typeof body?.status === "string" ? body.status : "invalid_server_response";
    console.error(`Groq smoke test failed: ${category}. ${typeof body?.message === "string" ? body.message : "The local assessment route returned an error."}`);
    process.exitCode = 1;
  } else if (body?.status !== "assessment_ready" || !AssessmentSchema.safeParse(body.assessment).success) {
    console.error("Groq smoke test: the route responded, but its assessment did not pass the expected schema.");
    process.exitCode = 1;
  } else {
    console.log("Groq smoke test: local route, provider connectivity, and structured response validation passed. Assessment content was not printed.");
  }
} catch (error) {
  const code = error?.name === "TimeoutError" || error?.name === "AbortError" ? "local_route_timeout" : "local_route_unavailable";
  console.error(`Groq smoke test failed: ${code}. Ensure the development server is running and try again.`);
  process.exitCode = 1;
}
