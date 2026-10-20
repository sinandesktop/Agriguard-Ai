"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { AssessmentSchema } from "@/lib/assessment-schema";
import type { Assessment } from "@/lib/assessment-schema";
import { getPhase9ClassResult, type Phase9ClassResult } from "@/lib/disease-mapping";
import { TreatmentGuidanceSchema } from "@/lib/treatment-guidance-schema";
import type { TreatmentGuidance } from "@/lib/treatment-guidance-schema";
import { AGRICULTURAL_REFERENCES } from "@/lib/agricultural-references";
import { runPhase9Inference } from "../real-inference/phase9-inference";
import { isConfiguredPhase9ModelAvailable } from "../real-inference/phase9-model-config";

const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
const LOCAL_MODEL_AVAILABLE = isConfiguredPhase9ModelAvailable();
const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const PLANTS = [
  ["tomato", "Tomato"],
  ["potato", "Potato"],
  ["mango", "Mango"],
  ["banana", "Banana"],
  ["chilli", "Chilli"],
  ["other", "Other / specify"],
] as const;
type AssessmentSource = "" | "local" | "groq";

const SAFE_ERRORS: Record<string, string> = {
  provider_not_configured: "The selected online provider is not configured on this server yet.",
  model_unavailable: "The online assessment model is unavailable for this service right now.",
  provider_credentials: "The online assessment service could not be reached. Please try again later.",
  provider_quota: "Groq returned a rate-limit or quota response. Wait and check your Groq usage limits before retrying.",
  provider_timeout: "The assessment took too long. Please try again.",
  rate_limited: "Too many requests were made. Please wait a little before trying again.",
  invalid_image: "This image could not be read. Choose a clear JPEG, PNG, or WebP image.",
  image_too_large: "This image is larger than the 4 MiB limit. Choose a smaller image.",
  invalid_plant: "Choose a supported plant and check the plant name.",
  consent_required: "Please confirm consent before sending a photo for online assessment.",
  invalid_provider_response: "A safe assessment could not be produced. Please try again or ask an agricultural expert.",
  provider_error: "The online assessment could not be completed. Please try again later.",
  provider_invalid_request: "The online assessment request was rejected. Please retry later or ask the site administrator to check the service configuration.",
  provider_unavailable: "The selected online provider is temporarily unavailable. Please try again later.",
  empty_provider_response: "The selected provider returned an empty assessment. Please try again or ask an agricultural expert.",
  response_validation_failed: "The selected provider returned an assessment that could not be safely checked. Please try again or ask an agricultural expert.",
};

type UiError = { code: string; message: string };

function assessmentError(code: unknown): UiError {
  const safeCode = typeof code === "string" ? code : "provider_error";
  return { code: safeCode, message: SAFE_ERRORS[safeCode] ?? "The assessment could not be completed. Please try again." };
}

type LocalInference = (file: File) => Promise<{ classId: number; label: string }>;

export default function AssessmentForm({ localInference = runPhase9Inference }: { localInference?: LocalInference } = {}) {
  const [plantId, setPlantId] = useState("");
  const [customPlant, setCustomPlant] = useState("");
  const [source, setSource] = useState<AssessmentSource>("");
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [consent, setConsent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [localResult, setLocalResult] = useState<Phase9ClassResult | null>(null);
  const [guidance, setGuidance] = useState<TreatmentGuidance | null>(null);
  const [localError, setLocalError] = useState("");
  const [error, setError] = useState<UiError | null>(null);
  const [inputError, setInputError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const plantSelectRef = useRef<HTMLSelectElement>(null);
  const submissionInFlight = useRef(false);
  const requestSequence = useRef(0);

  useEffect(() => {
    if (!image) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(image);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  const selectedPlantLabel = useMemo(() => PLANTS.find(([id]) => id === plantId)?.[1] ?? "", [plantId]);
  const validPlant = Boolean(plantId) && (plantId !== "other" || Boolean(customPlant.trim()));
  const externalSelected = source === "groq";
  const externalProviderLabel = "Groq";
  const canSubmit = validPlant && Boolean(image) && Boolean(source) && (!externalSelected || consent) && !loading;

  function clearAssessmentState() {
    setAssessment(null);
    setLocalResult(null);
    setGuidance(null);
    setLocalError("");
    setError(null);
    setInputError("");
  }

  function invalidatePendingAssessment() {
    requestSequence.current += 1;
    if (submissionInFlight.current) {
      submissionInFlight.current = false;
      setLoading(false);
    }
  }

  function choosePlant(event: ChangeEvent<HTMLSelectElement>) {
    invalidatePendingAssessment();
    setPlantId(event.currentTarget.value);
    setSource("");
    setConsent(false);
    clearAssessmentState();
  }

  function changeCustomPlant(event: ChangeEvent<HTMLInputElement>) {
    invalidatePendingAssessment();
    setCustomPlant(event.currentTarget.value);
    setSource("");
    setConsent(false);
    clearAssessmentState();
  }

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    invalidatePendingAssessment();
    const file = event.currentTarget.files?.[0] ?? null;
    setConsent(false);
    clearAssessmentState();
    if (!file) return;
    if (!ACCEPTED_TYPES.has(file.type.toLowerCase())) {
      setImage(null);
      setInputError("Choose a JPEG, PNG, or WebP image.");
      event.currentTarget.value = "";
      return;
    }
    if (file.size < 1 || file.size > MAX_IMAGE_BYTES) {
      setImage(null);
      setInputError(file.size > MAX_IMAGE_BYTES ? "The image must be 4 MiB or smaller." : "The selected image is empty.");
      event.currentTarget.value = "";
      return;
    }
    setImage(file);
    setInputError("");
  }

  function removeImage() {
    invalidatePendingAssessment();
    setImage(null);
    setConsent(false);
    clearAssessmentState();
    if (inputRef.current) inputRef.current.value = "";
  }

  function resetAssessment() {
    invalidatePendingAssessment();
    setPlantId("");
    setCustomPlant("");
    setSource("");
    setImage(null);
    setConsent(false);
    clearAssessmentState();
    if (inputRef.current) inputRef.current.value = "";
    plantSelectRef.current?.focus();
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submissionInFlight.current) return;
    clearAssessmentState();
    if (!validPlant) {
      setInputError(plantId === "other" ? "Enter the plant name to continue." : "Select a plant to continue.");
      return;
    }
    if (!image) {
      setInputError("Choose a leaf photo to continue.");
      return;
    }
    if (!source) {
      setInputError("Choose local assessment or Groq.");
      return;
    }
    if (externalSelected && !consent) {
      setInputError(`Consent is required before this photo is sent to ${externalProviderLabel}.`);
      return;
    }

    const requestId = ++requestSequence.current;
    if (source === "local") {
      if (plantId !== "tomato" && plantId !== "potato") {
        setLocalError("Local assessment is available only for tomato and potato.");
        return;
      }
      submissionInFlight.current = true;
      setLoading(true);
      try {
        const prediction = await localInference(image);
        if (requestSequence.current !== requestId) return;
        const mapped = getPhase9ClassResult(prediction.classId);
        if (!mapped || mapped.label !== prediction.label) throw new Error("The local output did not match the saved class mapping.");
        setLocalResult(mapped);
        const guidanceResult = await loadGuidance({ source: "local_phase9", selected_plant_id: plantId, class_id: mapped.classId });
        if (requestSequence.current === requestId) setGuidance(guidanceResult);
      } catch {
        if (requestSequence.current === requestId) setLocalError("Local assessment could not be completed. You can retry or explicitly choose the optional online assessment.");
      } finally {
        if (requestSequence.current === requestId) {
          submissionInFlight.current = false;
          setLoading(false);
        }
      }
      return;
    }

    const body = new FormData();
    body.set("plant_id", plantId);
    body.set("provider", source);
    body.set("external_consent", "true");
    if (plantId === "other") body.set("custom_plant_name", customPlant.trim());
    body.set("image", image);

    submissionInFlight.current = true;
    setLoading(true);
    try {
      const response = await fetch("/api/assess-plant", { method: "POST", body });
      if (requestSequence.current !== requestId) return;
      let payload: unknown;
      try { payload = await response.json(); } catch { payload = null; }
      if (requestSequence.current !== requestId) return;
      if (!response.ok || !payload || typeof payload !== "object") {
        const code = payload && typeof payload === "object" ? (payload as { status?: unknown }).status : "provider_error";
        setError(assessmentError(code));
        return;
      }
      const candidate = payload as { status?: unknown; assessment?: unknown; guidance?: unknown };
      const parsed = AssessmentSchema.safeParse(candidate.assessment);
      if (candidate.status !== "assessment_ready" || !parsed.success) {
        setError(assessmentError("invalid_provider_response"));
        return;
      }
      setAssessment(parsed.data);
      const guidanceResult = TreatmentGuidanceSchema.safeParse(candidate.guidance);
      setGuidance(guidanceResult.success ? guidanceResult.data : {
        status: "guidance_withheld",
        reason: "unavailable",
        message: "Verified guidance could not be loaded. Ask a qualified agricultural expert.",
      });
    } catch {
      if (requestSequence.current === requestId) setError(assessmentError("provider_error"));
    } finally {
      if (requestSequence.current === requestId) {
        submissionInFlight.current = false;
        setLoading(false);
      }
    }
  }

  async function loadGuidance(request: Record<string, unknown>): Promise<TreatmentGuidance> {
    try {
      const response = await fetch("/api/treatment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      const parsed = TreatmentGuidanceSchema.safeParse(await response.json());
      if (parsed.success) return parsed.data;
    } catch { /* Report a safe local message below; do not hide the assessment itself. */ }
    return { status: "guidance_withheld", reason: "unavailable", message: "Verified guidance could not be loaded. Ask a qualified agricultural expert." };
  }

  function GuidancePanel({ value }: { value: TreatmentGuidance | null }) {
    if (!value) return null;
    if (value.status === "general_reference") {
      return (
        <div className="guidance-panel result-caution">
          <h4>{value.heading}</h4>
          <p>{value.message}</p>
          <ul>{value.steps.map((step, index) => <li key={`${index}-${step}`}>{step}</li>)}</ul>
          {value.sources.length > 0 && <p className="guidance-sources">Sources: {value.sources.map((item, index) => <span key={item.url}>{index > 0 ? "; " : ""}<a href={item.url} target="_blank" rel="noreferrer">{item.title}</a></span>)}</p>}
          <p className="result-disclaimer">General non-chemical information only. No medicine, pesticide, or dosage has been verified for this image.</p>
        </div>
      );
    }
    if (value.status === "invalid_input") {
      return <div className="guidance-panel result-caution"><h3>Verified guidance unavailable</h3><p>{value.message}</p></div>;
    }
    if (value.status === "guidance_withheld") {
      return (
        <div className="guidance-panel result-caution">
          <h3>Verified guidance unavailable</h3>
          <p>{value.message}</p>
          {value.sources && value.sources.length > 0 && (
            <>
              <p>Source for further reading:</p>
              <ul className="guidance-sources">{value.sources.map((item) => <li key={item.url}><a href={item.url} target="_blank" rel="noreferrer">{item.title}</a></li>)}</ul>
              <p className="result-disclaimer">The source may include chemical-control information. AgriGuard has not verified current product approvals or label directions; confirm with a local agricultural expert and follow the current product label.</p>
            </>
          )}
        </div>
      );
    }
    return (
      <div className="guidance-panel">
        <h4>{value.guidance_kind === "healthy_prevention" ? "General prevention guidance" : "Verified non-chemical guidance"}</h4>
        <p>{value.message}</p>
        {value.conditional_on_predicted_crop && <p className="mismatch-warning"><strong>Conditional on the predicted crop being correct.</strong> Confirm the plant before acting; do not use chemical products based on this assessment.</p>}
        <ul>{value.record.non_chemical_management.map((step, index) => <li key={`${index}-${step}`}>{step}</li>)}</ul>
        {value.record.sources.length > 0 && <p className="guidance-sources">Sources: {value.record.sources.map((item, index) => <span key={item.url}>{index > 0 ? "; " : ""}<a href={item.url} target="_blank" rel="noreferrer">{item.title}</a></span>)}</p>}
        <p className="result-disclaimer">A verified chemical recommendation is not available yet.</p>
      </div>
    );
  }

  const selectedCropForComparison = plantId === "other" ? customPlant.trim().toLowerCase() : plantId;
  const predictedCropNormalized = assessment?.predicted_crop.trim().toLowerCase() ?? "unknown";
  const assessmentCropMismatch = Boolean(assessment && predictedCropNormalized !== "unknown" && predictedCropNormalized !== selectedCropForComparison);
  const assessmentDiagnosis = !assessment
    ? ""
    : assessment.result_type === "disease"
      ? `${assessment.predicted_crop} - ${assessment.possible_disease}`
      : assessment.result_type === "healthy"
        ? `${assessment.predicted_crop} - appears healthy`
        : assessment.result_type === "uncertain"
          ? `${assessment.predicted_crop} - disease uncertain`
          : "Could not assess this plant reliably";
  const uncertain = assessment && (
    assessment.plant_match !== "match" ||
    assessment.image_quality !== "good" ||
    assessment.result_type === "uncertain" ||
    assessment.result_type === "cannot_assess" ||
    assessmentCropMismatch
  );
  const assessmentNotice = assessment?.result_type === "cannot_assess"
    ? "This photo could not be assessed reliably. Try a clear, close-up photo or ask a local agricultural expert."
    : assessment?.result_type === "uncertain" || assessment?.plant_match === "uncertain"
        ? "The plant or disease could not be identified clearly. Retake a clear photo or ask a local agricultural expert."
        : assessment && assessment.image_quality !== "good"
          ? "The provider marked the photo quality as unclear. Try a closer, well-lit photo of the affected leaf."
          : assessment?.needs_expert_review
            ? "The provider recommends expert confirmation before acting on this possible result."
            : "The plant identification may not match your selection. Confirm the plant before using the guidance below.";

  return (
    <section id="assessment" className="assessment-shell" aria-labelledby="assessment-heading">
      <header className="assessment-header">
        <p className="eyebrow">Start with a clear photo</p>
        <h2 id="assessment-heading">Check your plant</h2>
        <p>Choose the plant, add a leaf photo, then select how you would like to assess it.</p>
      </header>

      <form className="assessment-card" onSubmit={submit} noValidate aria-labelledby="assessment-heading">
        <div className="form-step">
          <label htmlFor="plant-select">1. Select the plant</label>
          <select ref={plantSelectRef} id="plant-select" value={plantId} onChange={choosePlant} aria-describedby="plant-help" disabled={loading}>
            <option value="">Choose a plant</option>
            {PLANTS.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
          </select>
          <p className="field-help" id="plant-help">Choose the plant shown in your photo.</p>
        </div>

        {plantId === "other" && (
          <div className="form-step">
            <label htmlFor="custom-plant">Plant name</label>
            <input id="custom-plant" value={customPlant} onChange={changeCustomPlant} maxLength={80} autoComplete="off" required aria-describedby="custom-plant-help" disabled={loading} />
            <p className="field-help" id="custom-plant-help">Enter the plant name before continuing.</p>
          </div>
        )}

        {plantId === "tomato" || plantId === "potato" ? (
          <div className="local-unavailable" role="note">
            {LOCAL_MODEL_AVAILABLE ? <><strong>Local assessment available</strong><span>Your photo stays in this browser.</span></> : <><strong>Local assessment unavailable</strong><span>This deployment has not been configured with the model file yet.</span></>}
          </div>
        ) : plantId ? (
          <div className="local-unavailable" role="note">
            <strong>Online assessment required</strong>
            <span>Local assessment supports tomato and potato.</span>
          </div>
        ) : null}

        <div className="form-step">
          <label htmlFor="leaf-image">2. Add a leaf photo</label>
          <input
            ref={inputRef}
            id="leaf-image"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            capture="environment"
            onChange={chooseImage}
            aria-describedby="image-help"
            disabled={loading}
          />
        <p className="field-help" id="image-help">JPEG, PNG, or WebP. Maximum 4 MiB. You can use your camera or choose a photo.</p>
          {previewUrl && image && (
            <div className="image-preview">
              <img src={previewUrl} alt={`Preview of selected leaf photo: ${image.name}`} />
              <div className="preview-caption">
                <span title={image.name}>{image.name}</span>
                <div className="preview-actions">
                  <button className="replace-button" type="button" onClick={() => inputRef.current?.click()} disabled={loading}>Replace photo</button>
                  <button className="text-button" type="button" onClick={removeImage} disabled={loading}>Remove photo</button>
                </div>
              </div>
            </div>
          )}
          {inputError && <p className="inline-error" role="alert">{inputError}</p>}
        </div>

        {externalSelected && (
          <div className="external-notice" role="note">
            <strong>External processing disclosure</strong>
            <p>Your photo will be sent to {externalProviderLabel} only after you agree and analyze. AgriGuard AI does not intentionally store it, but we cannot control the provider’s processing or data-retention practices. Do not upload photos containing people or private information unless necessary.</p>
          </div>
        )}

        <fieldset className="source-choice" disabled={loading}>
          <legend>3. Choose how to assess it</legend>
          {(plantId === "tomato" || plantId === "potato") && LOCAL_MODEL_AVAILABLE && (
            <label className="source-option">
              <input type="radio" name="assessment-source" value="local" checked={source === "local"} onChange={() => { invalidatePendingAssessment(); setSource("local"); setConsent(false); clearAssessmentState(); }} />
              <span><strong>Local assessment</strong><small>Runs in this browser. Your photo stays here.</small></span>
            </label>
          )}
          {plantId && (
            <label className="source-option">
              <input type="radio" name="assessment-source" value="groq" checked={source === "groq"} onChange={() => { invalidatePendingAssessment(); setSource("groq"); setConsent(false); clearAssessmentState(); }} />
              <span><strong>Groq</strong><small>Your photo is sent to Groq only after you agree and analyze.</small></span>
            </label>
          )}
          {!plantId && <p className="field-help">Select a plant to see supported assessment paths.</p>}
        </fieldset>

        {externalSelected && (
          <label className="consent-control">
            <input type="checkbox" checked={consent} onChange={(event) => { invalidatePendingAssessment(); setConsent(event.currentTarget.checked); clearAssessmentState(); }} disabled={loading} />
            <span>I agree to send this photo and plant selection to {externalProviderLabel} for an online assessment.</span>
          </label>
        )}

        <button className="assessment-button" type="submit" disabled={!canSubmit}>{loading ? "Analyzing leaf…" : "Analyze leaf"}</button>

        {loading && <p className="loading-state" role="status" aria-live="polite">{source === "local" ? "Checking this photo on your device. The first check may take a little longer." : `Preparing your photo and requesting ${externalProviderLabel}…`}</p>}
        {localError && <section className="assessment-error" role="alert"><h2>Local assessment unavailable</h2><p>{localError}</p></section>}
        {error && (
          <section className="assessment-error" role="alert">
            <h2>Assessment could not be completed</h2>
            <p>{error.message}</p>
            <button className="secondary-button" type="submit" disabled={!canSubmit}>Try again</button>
          </section>
        )}

        {assessment && (
          <section className={`assessment-result ${uncertain ? "result-caution" : "result-neutral"}`} aria-live="polite" aria-labelledby="result-heading">
            <h2 id="result-heading">Possible diagnosis</h2>
            <p className="diagnosis-name">{assessmentDiagnosis}</p>
            <p>Selected crop: {selectedPlantLabel}{plantId === "other" ? ` (${customPlant.trim()})` : ""}</p>
            <p className="assessment-source">Assessment source: Groq</p>
            {assessmentCropMismatch && <p className="mismatch-warning" role="alert"><strong>Your selected plant is {plantId === "other" ? customPlant.trim() : selectedPlantLabel}, but the assessment identified {assessment.predicted_crop}. This prediction may be wrong. Confirm the plant before using the guidance below.</strong></p>}
            {uncertain && !assessmentCropMismatch && <p className="mismatch-warning" role="alert">{assessmentNotice}</p>}
            <p>{assessment.explanation}</p>
            <section className="ai-suggestions" aria-label="AI-generated suggestions">
              <h3>AI suggestions — not verified</h3>
              <ul>{assessment.ai_suggestions.map((suggestion, index) => <li key={`${index}-${suggestion}`}>{suggestion}</li>)}</ul>
              <p className="result-disclaimer">These suggestions are generated by Groq; they may be wrong. Confirm crop-specific actions with a local agricultural expert. No chemical or dosage advice is provided here.</p>
            </section>
            <h3>Recommended treatment</h3>
            <GuidancePanel value={guidance} />
            <section className="agricultural-references" aria-label="Agricultural reference portals">
              <h3>More agricultural information</h3>
              <p>These official portals may have crop-specific information. They may not cover this exact plant or condition; confirm relevance before acting.</p>
              <ul>{AGRICULTURAL_REFERENCES.map((reference) => <li key={reference.url}><a href={reference.url} target="_blank" rel="noreferrer">{reference.title}</a></li>)}</ul>
            </section>
            <h3>What to do next</h3>
            <p>{assessment.needs_expert_review || assessmentCropMismatch || assessment.image_quality !== "good" ? "Confirm the plant and diagnosis with a qualified local agricultural expert before taking disease-specific action." : "Watch for changes and seek local advice if symptoms persist or worsen."}</p>
            {assessment.result_type === "healthy" && <p>A healthy appearance can still miss disease.</p>}
            <p className="result-disclaimer">This is an AI assessment, not a confirmed diagnosis.</p>
            <button className="secondary-button reset-button" type="button" onClick={resetAssessment}>Check another leaf</button>
          </section>
        )}
        {localResult && (
          <section className="assessment-result result-caution" aria-live="polite" aria-labelledby="local-result-heading">
            <h2 id="local-result-heading">Possible diagnosis</h2>
            <p className="diagnosis-name">{localResult.plantId === "potato" ? "Potato" : "Tomato"} - {localResult.healthy ? "possible healthy appearance" : localResult.displayName}</p>
            <p>Selected crop: {selectedPlantLabel}</p>
            <p className="assessment-source">Assessment source: Local model; your photo stays in this browser.</p>
            {localResult.plantId !== plantId && <p className="mismatch-warning" role="alert"><strong>Your selected crop is {selectedPlantLabel}, but the model predicted {localResult.plantId === "potato" ? "Potato" : "Tomato"}. This prediction may be wrong. Confirm which plant is in the photo before applying treatment. Any guidance below is conditional on the predicted crop being correct.</strong></p>}
            <h3>Recommended treatment</h3>
            <GuidancePanel value={guidance} />
            <h3>What to do next</h3>
            <p>{localResult.healthy ? "The model did not identify one of its modeled diseases, but it can still miss disease. Monitor for new symptoms." : localResult.plantId !== plantId ? "Confirm the plant and diagnosis with a qualified local agricultural expert before taking disease-specific action." : "Watch for changes and seek local advice if symptoms persist or worsen."}</p>
            <p className="result-disclaimer">This is an AI assessment, not a confirmed diagnosis.</p>
            <button className="secondary-button reset-button" type="button" onClick={resetAssessment}>Check another leaf</button>
          </section>
        )}
      </form>

    </section>
  );
}
