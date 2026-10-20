import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { join } from "node:path";
import { Readable } from "node:stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ALLOWED_ASSETS = new Set([
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
]);

export async function GET(_request: Request, context: { params: Promise<{ asset: string }> }) {
  const { asset } = await context.params;
  if (!ALLOWED_ASSETS.has(asset)) return Response.json({ error: "Runtime asset not found." }, { status: 404, headers: { "Cache-Control": "no-store" } });
  // `asset` must remain allowlisted above. `join` uses the host OS separator;
  // a hard-coded Windows backslash rejects valid paths on Vercel's Linux runtime.
  const assetPath = join(process.cwd(), "node_modules", "onnxruntime-web", "dist", asset);
  try {
    const info = await stat(assetPath);
    if (!info.isFile()) throw new Error("runtime asset is not a file");
    return new Response(Readable.toWeb(createReadStream(assetPath)) as ReadableStream, {
      headers: {
        "Content-Type": asset.endsWith(".mjs") ? "text/javascript; charset=utf-8" : "application/wasm",
        "Content-Length": String(info.size),
        "Cache-Control": "public, max-age=86400",
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch {
    return Response.json({ error: "ONNX Runtime Web assets are unavailable." }, { status: 503, headers: { "Cache-Control": "no-store" } });
  }
}
