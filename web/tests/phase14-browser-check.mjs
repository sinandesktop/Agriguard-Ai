import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import WebSocket from "ws";

const ROOT = path.resolve("..");
const FIXTURE_PATH = path.join(ROOT, "ml/.dev_artifacts/phase14/browser-feature-fixtures.json");
const CHROME = process.env.CHROME_PATH ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const ORIGIN = process.env.AGRIGUARD_TEST_ORIGIN ?? "http://localhost:3180";
const UI_RESULT_TEXT = {
  Tomato___healthy: "Possible result: The model did not identify one of its modeled diseases for tomato.",
  Tomato___Early_blight: "Possible result: Early blight",
  Tomato___Late_blight: "Possible result: Late blight",
  Potato___healthy: "Possible result: The model did not identify one of its modeled diseases for potato.",
  Potato___Early_blight: "Possible result: Early blight",
  Potato___Late_blight: "Possible result: Late blight",
};

class DevTools {
  socket;
  nextId = 0;
  pending = new Map();
  events = [];

  constructor(socket) { this.socket = socket; }

  static async connect(url) {
    const socket = new WebSocket(url);
    await new Promise((resolve, reject) => {
      socket.once("open", resolve);
      socket.once("error", reject);
    });
    const connection = new DevTools(socket);
    socket.on("message", (data) => {
      const message = JSON.parse(data.toString());
      if (message.id && connection.pending.has(message.id)) {
        const { resolve, reject } = connection.pending.get(message.id);
        connection.pending.delete(message.id);
        message.error ? reject(new Error(message.error.message)) : resolve(message.result);
      } else connection.events.push(message);
    });
    return connection;
  }

  send(method, params = {}) {
    const id = ++this.nextId;
    const result = new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
    this.socket.send(JSON.stringify({ id, method, params }));
    return result;
  }

  async evaluate(expression) {
    const response = await this.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (response.exceptionDetails) throw new Error(response.exceptionDetails.exception?.description ?? response.exceptionDetails.text);
    return response.result?.value;
  }
}

async function waitFor(fn, timeout = 300_000) {
  const end = Date.now() + timeout;
  let last;
  while (Date.now() < end) {
    try {
      const value = await fn();
      if (value) return value;
    } catch (error) { last = error; }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Browser check timed out${last ? `: ${last}` : ""}`);
}

const fixture = JSON.parse(await readFile(FIXTURE_PATH, "utf8"));
const entries = fixture.entries.filter((entry) => entry.source_kind === "plantvillage");
assert.equal(entries.length, 6, "expected one existing non-test image per Phase 9 class");

const sampleLimit = Number(process.env.PHASE14_BROWSER_SAMPLE_COUNT ?? entries.length);
const profile = await mkdtemp(path.join(os.tmpdir(), "agriguard-phase14-chrome-"));
const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--remote-debugging-port=0", "--remote-allow-origins=*",
  `--user-data-dir=${path.join(profile, "profile")}`, "about:blank",
], { stdio: "ignore", windowsHide: true });
let connection;

try {
  const activePortPath = path.join(profile, "profile", "DevToolsActivePort");
  console.log("browser-check: waiting for Chrome");
  const activePort = await waitFor(async () => {
    const contents = await readFile(activePortPath, "utf8").catch(() => "");
    return contents.split(/\r?\n/)[0] || null;
  }, 30_000);
  console.log("browser-check: Chrome ready");
  const targetResponse = await fetch(`http://127.0.0.1:${activePort}/json/new?${encodeURIComponent(ORIGIN)}`, { method: "PUT" });
  assert.equal(targetResponse.status, 200, "Chrome could not open the local app tab");
  const target = await targetResponse.json();
  connection = await DevTools.connect(target.webSocketDebuggerUrl);
  await connection.send("Page.enable");
  await connection.send("Runtime.enable");
  await connection.send("DOM.enable");
  await connection.send("Network.enable");
  console.log("browser-check: connected to page");
  await waitFor(() => connection.evaluate("document.querySelector('#leaf-image') !== null"), 60_000);
  console.log("browser-check: form ready");

  const predictions = [];
  for (const entry of entries.slice(0, sampleLimit)) {
    console.log(`browser-check: starting ${entry.id}`);
    const imagePath = path.join(ROOT, "ml/data/plantvillage-source/raw/color", entry.source_path);
    const plant = entry.label.startsWith("Tomato") ? "tomato" : "potato";
    await connection.evaluate(`(() => { const e=document.querySelector('#plant-select'); const setter=Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set; setter.call(e,${JSON.stringify(plant)}); e.dispatchEvent(new Event('change',{bubbles:true})); })()`);
    await waitFor(() => connection.evaluate(`document.querySelector('#plant-select')?.value === ${JSON.stringify(plant)} && document.querySelector('input[name=assessment-source][value=local]') !== null`));
    const documentInfo = await connection.send("DOM.getDocument", { depth: -1, pierce: true });
    const inputQuery = await connection.send("DOM.querySelector", { nodeId: documentInfo.root.nodeId, selector: "#leaf-image" });
    await connection.send("DOM.setFileInputFiles", { nodeId: inputQuery.nodeId, files: [imagePath] });
    console.log(`browser-check: file selected ${entry.id}`);
    await waitFor(() => connection.evaluate("document.querySelector('#leaf-image')?.files?.length === 1"));

    // Check Chrome decoded the lossless PNG back to the exact Python RGB pixel array.
    const browserPixels = await connection.evaluate(`(async()=>{const f=document.querySelector('#leaf-image').files[0];const b=await createImageBitmap(f,{imageOrientation:'none',colorSpaceConversion:'none',premultiplyAlpha:'none'});const c=document.createElement('canvas');c.width=b.width;c.height=b.height;const x=c.getContext('2d',{willReadFrequently:true,colorSpace:'srgb'});x.drawImage(b,0,0);const d=x.getImageData(0,0,b.width,b.height).data;const a=new Uint8Array(b.width*b.height*3);for(let i=0,j=0;i<a.length;i+=3,j+=4){a[i]=d[j];a[i+1]=d[j+1];a[i+2]=d[j+2]}b.close();let s='';for(let i=0;i<a.length;i+=0x8000)s+=String.fromCharCode(...a.subarray(i,i+0x8000));return btoa(s)})()`);
    assert.equal(browserPixels, entry.rgb_base64, `${entry.id} did not decode to the exact fixture RGB pixels`);

    console.log(`browser-check: pixels match ${entry.id}`);
    const localControls = await connection.evaluate("JSON.stringify({ selectedPlant: document.querySelector('#plant-select')?.value, localOption: document.querySelector('input[name=assessment-source][value=local]') !== null, formText: document.querySelector('form')?.innerText.slice(0, 500) })");
    assert.equal(JSON.parse(localControls).localOption, true, `Local inference option not rendered: ${localControls}`);
    await connection.evaluate("document.querySelector('input[name=assessment-source][value=local]').click()");
    await connection.evaluate("document.querySelector('.assessment-button').click()");
    console.log(`browser-check: inference submitted ${entry.id}`);
    const rendered = await waitFor(async () => {
      const text = await connection.evaluate("document.querySelector('.assessment-result h2')?.textContent || ''");
      if (text.startsWith("Possible result:")) return text;
      const error = await connection.evaluate("document.querySelector('.assessment-error')?.textContent || document.querySelector('.assessment-result')?.textContent || ''");
      if (error && /could not|unavailable|failed|Error/i.test(error)) return { error };
      return null;
    });
    if (typeof rendered !== "string") throw new Error(`Local inference failed: ${rendered.error}`);
    console.log(`browser-check: rendered ${entry.id} ${rendered}`);
    const expectedLabel = fixture.class_id_to_label[String(entry.class_id)];
    assert.equal(rendered, UI_RESULT_TEXT[expectedLabel], `Unexpected browser model prediction for ${entry.id}`);
    const network = await connection.evaluate("performance.getEntriesByType('resource').map(e=>e.name)");
    assert.ok(network.some((url) => url.includes("/api/models/phase9")), "local inference did not load the Phase 9 model");
    assert.equal(network.some((url) => url.includes("/api/assess-plant")), false, "local image was sent to the external assessment endpoint");
    predictions.push({ id: entry.id, sourcePath: entry.source_path, label: entry.label, expectedModelLabel: expectedLabel, browserResult: rendered });
  }
  console.log(JSON.stringify({ modelSha256: fixture.phase9_model_sha256, browserPredictionsMatched: predictions.length, sourceFiles: "Original PlantVillage JPEGs from existing train/validation manifests (non-test)", predictions, imageUploadRouteRequested: false }, null, 2));
} finally {
  connection?.socket.close();
  spawnSync("taskkill.exe", ["/PID", String(chrome.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore", timeout: 15_000 });
  await Promise.race([new Promise((resolve) => chrome.once("exit", resolve)), new Promise((resolve) => setTimeout(resolve, 5_000))]);
  const profileRoot = path.resolve(profile);
  if (profileRoot.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
    for (let attempt = 0; attempt < 5; attempt++) {
      try { await rm(profileRoot, { recursive: true, force: true }); break; }
      catch (error) { if (attempt === 4) throw error; await new Promise((resolve) => setTimeout(resolve, 1_000)); }
    }
  }
}
