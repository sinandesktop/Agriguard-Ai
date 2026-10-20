const DEVELOPMENT_MODEL_URL = "/api/models/phase9";

/**
 * Resolve the model asset URL without ever exposing a storage credential.
 * Production deployments must configure a public, immutable model URL because
 * the ONNX file is too large to include in the Git/Vercel function bundle.
 */
export function resolvePhase9ModelUrl(configuredUrl: string | undefined, isProduction: boolean): string {
  const value = configuredUrl?.trim();
  if (!value) {
    if (isProduction) {
      throw new Error("The local model is not configured for this deployment. Ask the site administrator to configure its model asset.");
    }
    return DEVELOPMENT_MODEL_URL;
  }

  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error("The configured local model URL is invalid.");
  }
  if (parsed.protocol !== "https:" || parsed.username || parsed.password || parsed.hash) {
    throw new Error("The configured local model URL must be a public HTTPS asset URL without credentials or a fragment.");
  }
  return parsed.toString();
}

export function isPhase9ModelAvailable(configuredUrl: string | undefined, isProduction: boolean): boolean {
  try {
    resolvePhase9ModelUrl(configuredUrl, isProduction);
    return true;
  } catch {
    return false;
  }
}

export function configuredPhase9ModelUrl(): string {
  return resolvePhase9ModelUrl(process.env.NEXT_PUBLIC_PHASE9_MODEL_URL, process.env.NODE_ENV === "production");
}

export function isConfiguredPhase9ModelAvailable(): boolean {
  return isPhase9ModelAvailable(process.env.NEXT_PUBLIC_PHASE9_MODEL_URL, process.env.NODE_ENV === "production");
}
