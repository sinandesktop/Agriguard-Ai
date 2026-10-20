import { generateGroqAssessment } from "@/lib/groq-provider";
import { createAssessmentHandler } from "@/lib/plant-assessment";

export const runtime = "nodejs";

const handleAssessment = createAssessmentHandler({
  getApiKey: () => process.env.GROQ_API_KEY,
  generate: generateGroqAssessment,
});

export const POST = handleAssessment;
