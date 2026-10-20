import assert from "node:assert/strict";
import { afterEach, before, test } from "node:test";
import { JSDOM } from "jsdom";

const dom = new JSDOM("<!doctype html><html><body></body></html>", { url: "http://localhost" });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  HTMLElement: dom.window.HTMLElement,
  Node: dom.window.Node,
  File: dom.window.File,
  FormData: dom.window.FormData,
  IS_REACT_ACT_ENVIRONMENT: true,
});
Object.defineProperty(globalThis, "navigator", { configurable: true, value: dom.window.navigator });
let blobCounter = 0;
Object.defineProperty(URL, "createObjectURL", { configurable: true, value: () => `blob:test-${++blobCounter}` });
Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: () => undefined });

let render: typeof import("@testing-library/react").render;
let screen: typeof import("@testing-library/react").screen;
let within: typeof import("@testing-library/react").within;
let fireEvent: typeof import("@testing-library/react").fireEvent;
let waitFor: typeof import("@testing-library/react").waitFor;
let cleanup: typeof import("@testing-library/react").cleanup;
let user: import("@testing-library/user-event").UserEvent;
let AssessmentForm: typeof import("../app/assessment/assessment-form").default;
let HeroPhoto: typeof import("../app/hero-photo").default;
let SiteHeader: typeof import("../app/site-header").default;

before(async () => {
  const testing = await import("@testing-library/react");
  render = testing.render;
  screen = testing.screen;
  within = testing.within;
  fireEvent = testing.fireEvent;
  waitFor = testing.waitFor;
  cleanup = testing.cleanup;
  const userEvent = (await import("@testing-library/user-event")).default;
  user = userEvent.setup({ document: dom.window.document });
  AssessmentForm = (await import("../app/assessment/assessment-form")).default;
  HeroPhoto = (await import("../app/hero-photo")).default;
  SiteHeader = (await import("../app/site-header")).default;
});

afterEach(() => {
  cleanup();
});

test("mobile navigation opens, follows its anchor, and closes", async () => {
  render(<SiteHeader />);
  const toggle = screen.getByRole("button", { name: "Open navigation menu" });
  assert.equal(toggle.getAttribute("aria-expanded"), "false");
  await user.click(toggle);
  assert.equal(screen.getByRole("button", { name: "Close navigation menu" }).getAttribute("aria-expanded"), "true");
  const mobileNav = screen.getByRole("navigation", { name: "Mobile navigation" });
  await user.click(within(mobileNav).getByRole("link", { name: "How it works" }));
  assert.equal(screen.getByRole("button", { name: "Open navigation menu" }).getAttribute("aria-expanded"), "false");
});

const validAssessment = {
  plant_match: "match",
  possible_plant: "potato",
  predicted_crop: "Potato",
  result_type: "disease",
  possible_disease: "Early blight",
  explanation: "Brown leaf spots may fit early blight.",
  ai_suggestions: ["Take another clear photo of the whole leaf.", "Keep garden tools clean."],
  image_quality: "good",
  visible_evidence: ["Brown lesions are visible on the leaf."],
  uncertainties: ["The photo does not show the full plant."],
  needs_expert_review: false,
};

const WITHHELD_GUIDANCE = { status: "guidance_withheld", reason: "uncertain", message: "Disease-specific guidance is withheld. Retake a clear photo or ask a qualified agricultural expert." };
const EARLY_BLIGHT_GUIDANCE = {
  status: "guidance_available", guidance_kind: "possible_disease", conditional_on_predicted_crop: false,
  message: "General, source-backed management guidance for a possible disease; this is not a confirmed diagnosis.",
  record: {
    plant_id: "tomato", disease_id: "early_blight", model_label: "Tomato___Early_blight", condition: "early blight (Alternaria solani)",
    diagnostic_limitations: ["A photograph cannot confirm the cause."],
    treatment_categories: ["cultural", "sanitation"],
    non_chemical_management: ["Use disease-free seed from healthy fruit.", "Collect and dispose of affected crop debris."],
    chemical_options: [], chemical_status: "no_verified_recommendation",
    sources: [{ title: "Tomato early blight source", url: "https://example.org/tomato-early-blight", supports: "Management practices." }],
  },
};

function mockFetch(handler: typeof fetch, treatmentResponse: unknown = WITHHELD_GUIDANCE) {
  globalThis.fetch = async (input, init) => {
    if (input === "/api/treatment") return Response.json(treatmentResponse);
    const response = await handler(input, init);
    if (input === "/api/assess-plant" && response.ok) {
      const payload = await response.json() as Record<string, unknown>;
      return Response.json({ ...payload, guidance: treatmentResponse });
    }
    return response;
  };
}

test("initial form uses farmer wording and keeps model implementation details out of view", () => {
  render(<AssessmentForm />);
  assert.ok(screen.getByRole("heading", { name: "Check your plant" }));
  assert.ok(screen.getByRole("button", { name: "Analyze leaf" }));
  assert.equal((screen.getByLabelText("2. Add a leaf photo") as HTMLInputElement).getAttribute("capture"), null);
  fireEvent.change(screen.getByLabelText("1. Select the plant"), { target: { value: "potato" } });
  assert.ok(screen.getByRole("radio", { name: /Groq/ }));
  assert.doesNotMatch(document.body.textContent ?? "", /HOG|HSV|SVM|310 MB|Phase 9/);
});

test("hero photograph has a graceful illustrated fallback on load failure", () => {
  render(<HeroPhoto />);
  fireEvent.error(screen.getByAltText("Sunlit green tomato leaves covered with water droplets"));
  assert.ok(screen.getByRole("img", { name: "Illustration of a healthy green leaf" }));
});

async function selectPlantAndImage(plant = "potato", filename = "leaf.png") {
  await user.selectOptions(screen.getByLabelText("1. Select the plant"), plant);
  const image = new File([new Uint8Array([1, 2, 3])], filename, { type: "image/png" });
  await user.upload(screen.getByLabelText("2. Add a leaf photo"), image);
  return image;
}

async function chooseExternal() {
  await user.click(screen.getByRole("radio", { name: /Groq/ }));
}

test("preview can be removed and replaced with another image", async () => {
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage();
  assert.match((screen.getByAltText(/Preview of selected leaf photo/) as HTMLImageElement).src, /blob:test-/);
  assert.ok(screen.getByRole("button", { name: "Replace photo" }));
  assert.ok(screen.getByRole("button", { name: "Remove photo" }));
  await user.click(screen.getByRole("button", { name: "Remove photo" }));
  assert.equal(screen.queryByAltText(/Preview of selected leaf photo/), null);
  await user.upload(screen.getByLabelText("2. Add a leaf photo"), new File(["replacement"], "new-leaf.webp", { type: "image/webp" }));
  assert.match((screen.getByAltText(/Preview of selected leaf photo/) as HTMLImageElement).alt, /new-leaf.webp/);
});

test("unsupported image types are rejected and no request is sent", async () => {
  let calls = 0;
  mockFetch(async () => { calls += 1; return Response.json({}); });
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await user.selectOptions(screen.getByLabelText("1. Select the plant"), "potato");
  const input = screen.getByLabelText("2. Add a leaf photo") as HTMLInputElement;
  fireEvent.change(input, { target: { files: [new File(["x"], "bad.gif", { type: "image/gif" })] } });
  assert.match(screen.getByRole("alert").textContent ?? "", /JPEG, PNG, or WebP/);
  assert.equal(calls, 0);
});

test("images larger than 4 MiB are rejected", async () => {
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await user.selectOptions(screen.getByLabelText("1. Select the plant"), "potato");
  const tooLarge = new File([new Uint8Array(4 * 1024 * 1024 + 1)], "large.png", { type: "image/png" });
  fireEvent.change(screen.getByLabelText("2. Add a leaf photo"), { target: { files: [tooLarge] } });
  assert.match(screen.getByRole("alert").textContent ?? "", /4 MiB or smaller/);
});

test("submission requires explicit consent and sends exactly one deliberate request", async () => {
  let calls = 0;
  let captured: FormData | undefined;
  mockFetch(async (input, init) => {
    calls += 1;
    assert.equal(input, "/api/assess-plant");
    assert.equal(init?.method, "POST");
    captured = init?.body as FormData;
    return Response.json({ status: "assessment_ready", assessment: validAssessment });
  });
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage();
  await chooseExternal();
  const submit = screen.getByRole("button", { name: "Analyze leaf" });
  assert.equal((submit as HTMLButtonElement).disabled, true);
  assert.equal(calls, 0);
  await user.click(screen.getByRole("checkbox"));
  assert.equal((submit as HTMLButtonElement).disabled, false);
  await user.click(submit);
  await screen.findByText("Potato - Early blight");
  assert.equal(calls, 1);
  assert.equal(captured?.get("plant_id"), "potato");
  assert.equal(captured?.get("external_consent"), "true");
  assert.equal((captured?.get("image") as File).name, "leaf.png");
});

test("Groq result displays its disease name and exact approved guidance after consented submit", async () => {
  const potatoGuidance = { ...EARLY_BLIGHT_GUIDANCE, record: { ...EARLY_BLIGHT_GUIDANCE.record, plant_id: "potato", model_label: "Potato___Early_blight", condition: "early blight (Alternaria solani)" } };
  let assessmentCalls = 0;
  mockFetch(async (input, init) => {
    assert.equal(input, "/api/assess-plant");
    assessmentCalls++;
    assert.equal((init?.body as FormData).get("provider"), "groq");
    return Response.json({ status: "assessment_ready", assessment: validAssessment });
  }, potatoGuidance);
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage("potato");
  await chooseExternal();
  assert.ok(screen.getByText("External processing disclosure"));
  await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Potato - Early blight");
  assert.ok(screen.getByText("Assessment source: Groq"));
  assert.ok(screen.getByText("Verified non-chemical guidance"));
  assert.ok(screen.getByText("Collect and dispose of affected crop debris."));
  assert.equal(assessmentCalls, 1);
});

test("Groq is an explicit consent-gated provider and the selected provider is sent to the server", async () => {
  let captured: FormData | undefined;
  mockFetch(async (_input, init) => {
    captured = init?.body as FormData;
    return Response.json({ status: "assessment_ready", assessment: validAssessment });
  });
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage("potato");
  await user.click(screen.getByRole("radio", { name: /Groq/ }));
  assert.ok(screen.getByText("External processing disclosure"));
  assert.equal((screen.getByRole("button", { name: /Analyzing leaf|Analyze leaf/ }) as HTMLButtonElement).disabled, true);
  await user.click(screen.getByRole("checkbox", { name: /send this photo and plant selection to Groq/i }));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Potato - Early blight");
  assert.equal(captured?.get("provider"), "groq");
  assert.equal(captured?.get("external_consent"), "true");
  assert.ok(screen.getByText("Assessment source: Groq"));
});

test("external Banana diagnosis stays visible but expert-review results withhold crop-specific references", async () => {
  const bananaAssessment = { ...validAssessment, possible_plant: "banana", predicted_crop: "Banana", possible_disease: "banana leaf spot", explanation: "Possible leaf lesions are visible.", needs_expert_review: true };
  const bananaGuidance = { status: "guidance_withheld", reason: "review_required", message: "The provider recommends expert confirmation, so crop-specific comparison guidance is withheld." };
  mockFetch(async () => Response.json({ status: "assessment_ready", assessment: bananaAssessment }), bananaGuidance);
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage("banana");
  await chooseExternal();
  await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Banana - banana leaf spot");
  assert.ok(screen.getByText("Verified guidance unavailable"));
  assert.ok(screen.getByText(/crop-specific comparison guidance is withheld/i));
  assert.ok(screen.getByText("AI suggestions — not verified"));
  assert.ok(screen.getByText("Take another clear photo of the whole leaf."));
  assert.ok(screen.getByRole("link", { name: "Kerala Agricultural University Agri Infotech Portal" }));
  assert.ok(screen.getByRole("link", { name: "Tamil Nadu Agricultural University Agritech Portal" }));
  assert.equal(screen.queryByRole("link", { name: /Banana Sigatoka leaf spot|Banana yellow Sigatoka/ }), null);
});

test("a delayed assessment cannot replace results after its crop context changes", async () => {
  let resolveAssessment!: (response: Response) => void;
  let responseReturned = false;
  mockFetch(async () => {
    const response = await new Promise<Response>((resolve) => { resolveAssessment = resolve; });
    responseReturned = true;
    return response;
  });
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage("banana");
  await chooseExternal();
  await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));

  assert.equal((screen.getByLabelText("1. Select the plant") as HTMLSelectElement).disabled, true);
  assert.equal((screen.getByLabelText("2. Add a leaf photo") as HTMLInputElement).disabled, true);
  assert.equal((screen.getByRole("checkbox") as HTMLInputElement).disabled, true);
  assert.equal((screen.getByRole("button", { name: /Analyzing leaf/ }) as HTMLButtonElement).disabled, true);

  // Exercise the stale-response guard even if a context-change event slips past the disabled controls.
  fireEvent.change(screen.getByLabelText("1. Select the plant"), { target: { value: "tomato" } });
  assert.equal((screen.getByLabelText("1. Select the plant") as HTMLSelectElement).value, "tomato");
  resolveAssessment(Response.json({ status: "assessment_ready", assessment: validAssessment }));
  await waitFor(() => assert.equal(responseReturned, true));
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(screen.queryByText("Potato - Early blight"), null);
  assert.equal(screen.queryByText("Banana - banana leaf spot"), null);
  assert.equal(screen.queryByRole("heading", { name: "Possible diagnosis" }), null);
});

test("uncertain external output is shown cautiously without treatment advice", async () => {
  mockFetch(async () => Response.json({ status: "assessment_ready", assessment: {
    ...validAssessment,
    plant_match: "uncertain",
    result_type: "uncertain",
    possible_disease: null,
    image_quality: "uncertain",
    needs_expert_review: true,
  } }));
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage();
  await chooseExternal();
  await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Potato - disease uncertain");
  assert.ok(screen.getByText(/Potato - disease uncertain/));
  assert.ok(screen.getByText(/plant or disease could not be identified clearly/i));
  assert.ok(screen.getByText(/This is an AI assessment, not a confirmed diagnosis/));
  assert.equal(screen.queryByText(/Apply .* pesticide/i), null);
});

test("mismatch, poor quality, and expert review each produce a caution state", async (context) => {
  const cautionCases = [
    { name: "plant mismatch", changes: { plant_match: "mismatch" } },
    { name: "poor image quality", changes: { image_quality: "poor" } },
    { name: "expert review", changes: { needs_expert_review: true } },
  ] as const;
  for (const item of cautionCases) {
    await context.test(item.name, async () => {
      const reviewGuidance = {
        status: "guidance_withheld", reason: "review_required", message: "The provider recommends expert confirmation before disease-specific guidance is shown.",
        sources: [{ title: "Potato early blight — KAU e-Crop Doctor", url: "https://celkau.in/ecropdoctor/vegetables/potato/earlybight.php" }],
      };
      mockFetch(async () => Response.json({ status: "assessment_ready", assessment: { ...validAssessment, ...item.changes } }), reviewGuidance);
      user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
      render(<AssessmentForm />);
      await selectPlantAndImage();
      await chooseExternal();
      await user.click(screen.getByRole("checkbox"));
      await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
      if (item.name === "expert review") {
        await screen.findByText("Potato - Early blight");
        assert.ok(screen.getByText(/provider recommends expert confirmation/i));
        assert.equal((screen.getByRole("link", { name: /KAU e-Crop Doctor/ }) as HTMLAnchorElement).href, "https://celkau.in/ecropdoctor/vegetables/potato/earlybight.php");
        assert.ok(screen.getByText(/AgriGuard has not verified current product approvals or label directions/i));
        assert.equal(screen.queryByText(/image or result is uncertain/i), null);
      } else if (item.name === "poor image quality") {
        await screen.findByText(/provider marked the photo quality as unclear/i);
      } else {
        await screen.findByText(/plant identification may not match your selection/i);
      }
    });
  }
});

test("plant selection is required and custom plants use the exact API fields", async () => {
  let captured: FormData | undefined;
  mockFetch(async (_input, init) => {
    captured = init?.body as FormData;
    return Response.json({ status: "assessment_ready", assessment: { ...validAssessment, result_type: "uncertain", possible_disease: null } });
  });
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  assert.equal((screen.getByRole("button", { name: /Analyzing leaf|Analyze leaf/ }) as HTMLButtonElement).disabled, true);
  await user.selectOptions(screen.getByLabelText("1. Select the plant"), "other");
  await user.type(screen.getByLabelText("Plant name"), "Basil");
  await user.upload(screen.getByLabelText("2. Add a leaf photo"), new File(["x"], "basil.png", { type: "image/png" }));
  await chooseExternal();
  await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Potato - disease uncertain");
  assert.ok(screen.getByText(/selected plant is Basil, but the assessment identified Potato/i));
  assert.equal(captured?.get("plant_id"), "other");
  assert.equal(captured?.get("custom_plant_name"), "Basil");
});

test("server failure is shown safely and retry works", async () => {
  let calls = 0;
  mockFetch(async () => {
    calls += 1;
    return calls === 1
      ? Response.json({ status: "provider_quota", message: "secret key=do-not-show" }, { status: 503 })
      : Response.json({ status: "assessment_ready", assessment: validAssessment });
  });
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage();
  await chooseExternal();
  await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  const alert = await screen.findByRole("alert");
  assert.match(alert.textContent ?? "", /rate.limit or quota response/i);
  assert.doesNotMatch(alert.textContent ?? "", /secret key|do-not-show/);
  await user.click(screen.getByRole("button", { name: "Try again" }));
  await waitFor(() => assert.equal(calls, 2));
  await screen.findByText("Potato - Early blight");
});

test("malformed response is rejected without displaying provider text", async () => {
  mockFetch(async () => Response.json({ status: "assessment_ready", assessment: { possible_disease: "provider-secret" } }));
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage();
  await chooseExternal();
  await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  const alert = await screen.findByRole("alert");
  assert.match(alert.textContent ?? "", /safe assessment could not be produced/i);
  assert.doesNotMatch(alert.textContent ?? "", /provider-secret/);
});

test("changing plant context clears prior consent", async () => {
  const u = user.setup({ document: dom.window.document }); user = u;
  render(<AssessmentForm />);
  await selectPlantAndImage();
  await chooseExternal();
  await user.click(screen.getByRole("checkbox"));
  assert.equal((screen.getByRole("checkbox") as HTMLInputElement).checked, true);
  await user.upload(screen.getByLabelText("2. Add a leaf photo"), new File(["different"], "second.png", { type: "image/png" }));
  assert.equal((screen.getByRole("checkbox") as HTMLInputElement).checked, false);
  await user.click(screen.getByRole("checkbox"));
  await user.selectOptions(screen.getByLabelText("1. Select the plant"), "tomato");
  assert.equal(screen.queryByRole("checkbox"), null);
  await chooseExternal();
  assert.equal((screen.getByRole("checkbox") as HTMLInputElement).checked, false);
});

test("loading state is visible while the assessment request is pending", async () => {
  let release: ((response: Response) => void) | undefined;
  mockFetch(() => new Promise<Response>((resolve) => { release = resolve; }));
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage();
  await chooseExternal();
  await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  assert.ok(screen.getByRole("status").textContent?.includes("Preparing your photo"));
  assert.equal((screen.getByRole("button", { name: /Analyzing leaf|Analyze leaf/ }) as HTMLButtonElement).disabled, true);
  release?.(Response.json({ status: "assessment_ready", assessment: validAssessment }));
  await screen.findByText("Potato - Early blight");
});

test("tomato and potato local results show the disease name and approved catalogue guidance without online assessment", async () => {
  let fetchCalls = 0;
  mockFetch(async (input) => { fetchCalls++; assert.equal(input, "/api/assess-plant"); throw new Error("Online route should not run"); }, EARLY_BLIGHT_GUIDANCE);
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm localInference={async (file) => ({ classId: file.name.startsWith("tomato") ? 3 : 0, label: file.name.startsWith("tomato") ? "Tomato___Early_blight" : "Potato___Early_blight" })} />);
  await selectPlantAndImage("potato");
  await user.click(screen.getByRole("radio", { name: /Local assessment/ }));
  assert.equal(screen.queryByRole("checkbox"), null);
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Potato - Early blight");
  assert.ok(screen.getByText(/Assessment source: Local model/).textContent !== null);
  assert.ok(screen.getByText("Verified non-chemical guidance"));
  assert.ok(screen.getByText("Use disease-free seed from healthy fruit."));
  assert.equal(fetchCalls, 0);

  cleanup();
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm localInference={async () => ({ classId: 3, label: "Tomato___Early_blight" })} />);
  await selectPlantAndImage("tomato", "tomato-leaf.png");
  await user.click(screen.getByRole("radio", { name: /Local assessment/ }));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Tomato - Early blight");
  assert.ok(screen.getByText("Selected crop: Tomato"));
});

test("healthy local classes use cautious wording and no disease medicine", async () => {
  const guidance = { ...EARLY_BLIGHT_GUIDANCE, guidance_kind: "healthy_prevention", message: "No disease medicine is indicated.", record: { ...EARLY_BLIGHT_GUIDANCE.record, plant_id: "potato", disease_id: "healthy", model_label: "Potato___healthy", condition: "healthy", non_chemical_management: ["No disease medicine is indicated for a healthy classification."] } };
  mockFetch(async () => { throw new Error("Online assessment must not be called"); }, guidance);
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm localInference={async () => ({ classId: 2, label: "Potato___healthy" })} />);
  await selectPlantAndImage("potato");
  await user.click(screen.getByRole("radio", { name: /Local assessment/ }));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText(/Potato - possible healthy appearance/);
  assert.ok(screen.getByText(/can still miss disease/i));
  assert.ok(screen.getByText("General prevention guidance"));
  assert.ok(screen.getByText("No disease medicine is indicated for a healthy classification."));
});

test("Check another leaf clears the result and form and returns focus to crop selection", async () => {
  mockFetch(async () => { throw new Error("Local classification must not call online assessment"); });
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm localInference={async () => ({ classId: 2, label: "Potato___healthy" })} />);
  await selectPlantAndImage("potato");
  await user.click(screen.getByRole("radio", { name: /Local assessment/ }));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Potato - possible healthy appearance");
  await user.click(screen.getByRole("button", { name: "Check another leaf" }));
  assert.equal((screen.getByLabelText("1. Select the plant") as HTMLSelectElement).value, "");
  assert.equal((screen.getByLabelText("2. Add a leaf photo") as HTMLInputElement).value, "");
  assert.equal(screen.queryByText("Potato - possible healthy appearance"), null);
  assert.equal(screen.queryByRole("radio", { name: /Local assessment/ }), null);
  assert.equal(document.activeElement, screen.getByLabelText("1. Select the plant"));
});

test("renders the mapped possible disease or careful healthy wording for all six Phase 9 class IDs", async () => {
  const cases = [
    { classId: 0, label: "Potato___Early_blight", crop: "potato", expected: "Potato - Early blight" },
    { classId: 1, label: "Potato___Late_blight", crop: "potato", expected: "Potato - Late blight" },
    { classId: 2, label: "Potato___healthy", crop: "potato", expected: "Potato - possible healthy appearance" },
    { classId: 3, label: "Tomato___Early_blight", crop: "tomato", expected: "Tomato - Early blight" },
    { classId: 4, label: "Tomato___Late_blight", crop: "tomato", expected: "Tomato - Late blight" },
    { classId: 5, label: "Tomato___healthy", crop: "tomato", expected: "Tomato - possible healthy appearance" },
  ];
  for (const item of cases) {
    cleanup();
    user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
    mockFetch(async () => { throw new Error("Local classification must not call online assessment"); });
    render(<AssessmentForm localInference={async () => ({ classId: item.classId, label: item.label })} />);
    await selectPlantAndImage(item.crop);
    await user.click(screen.getByRole("radio", { name: /Local assessment/ }));
    await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
    await screen.findByText(item.expected);
  }
});

test("crop mismatch displays predicted crop and conditional guidance", async () => {
  const potatoGuidance = { ...EARLY_BLIGHT_GUIDANCE, conditional_on_predicted_crop: true, record: { ...EARLY_BLIGHT_GUIDANCE.record, plant_id: "potato", model_label: "Potato___Early_blight" } };
  mockFetch(async () => { throw new Error("No external assessment expected"); }, potatoGuidance);
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm localInference={async () => ({ classId: 0, label: "Potato___Early_blight" })} />);
  await selectPlantAndImage("tomato");
  await user.click(screen.getByRole("radio", { name: /Local assessment/ }));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Potato - Early blight");
  assert.ok(screen.getByText(/your selected crop is Tomato, but the model predicted Potato/i));
  assert.ok(screen.getByText("Verified non-chemical guidance"));
  assert.ok(screen.getByText(/Conditional on the predicted crop being correct/));
});

test("external assessment keeps the predicted crop separate and shares conditional catalogue guidance", async () => {
  const potatoGuidance = { ...EARLY_BLIGHT_GUIDANCE, conditional_on_predicted_crop: true, record: { ...EARLY_BLIGHT_GUIDANCE.record, plant_id: "potato", model_label: "Potato___Early_blight" } };
  mockFetch(async () => Response.json({ status: "assessment_ready", assessment: validAssessment }), potatoGuidance);
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm />);
  await selectPlantAndImage("tomato");
  await chooseExternal();
  await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Potato - Early blight");
  assert.ok(screen.getByText(/selected plant is Tomato, but the assessment identified Potato/i));
  assert.ok(screen.getByText(/Conditional on the predicted crop being correct/));
  assert.equal(screen.getByText("Selected crop: Tomato").textContent, "Selected crop: Tomato");
});

test("local model failure stays local until the user explicitly chooses Groq", async () => {
  let fetchCalls = 0;
  mockFetch(async (_input, init) => {
    fetchCalls++;
    assert.equal(init?.method, "POST");
    return Response.json({ status: "assessment_ready", assessment: { ...validAssessment, result_type: "uncertain", possible_disease: null } });
  });
  user = (await import("@testing-library/user-event")).default.setup({ document: dom.window.document });
  render(<AssessmentForm localInference={async () => { throw new Error("do not show internal runtime error"); }} />);
  await selectPlantAndImage("tomato");
  await user.click(screen.getByRole("radio", { name: /Local assessment/ }));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  assert.match((await screen.findByRole("alert")).textContent ?? "", /retry or explicitly choose/);
  assert.equal(fetchCalls, 0);
  assert.doesNotMatch(document.body.textContent ?? "", /internal runtime error/);
  await chooseExternal();
  await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Analyze leaf" }));
  await screen.findByText("Potato - disease uncertain");
  assert.ok(screen.getByText(/selected plant is Tomato, but the assessment identified Potato/i));
  assert.equal(fetchCalls, 1);
});
