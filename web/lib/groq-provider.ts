import "server-only";
import { createGroqRequestBody } from "./groq-request";

export async function generateGroqAssessment(input: {
  apiKey: string;
  plantDescription: string;
  imageBytes: Buffer;
  signal: AbortSignal;
}): Promise<string | undefined> {
  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${input.apiKey}`,
      "Content-Type": "application/json",
    },
    signal: input.signal,
    body: JSON.stringify(createGroqRequestBody(input.plantDescription, input.imageBytes.toString("base64"))),
  });

  if (!response.ok) throw Object.assign(new Error("Groq request failed"), { status: response.status });
  const result: unknown = await response.json();
  if (!result || typeof result !== "object") return undefined;
  const choices = (result as { choices?: unknown }).choices;
  if (!Array.isArray(choices) || !choices[0] || typeof choices[0] !== "object") return undefined;
  const message = (choices[0] as { message?: unknown }).message;
  if (!message || typeof message !== "object") return undefined;
  const content = (message as { content?: unknown }).content;
  return typeof content === "string" ? content : undefined;
}
