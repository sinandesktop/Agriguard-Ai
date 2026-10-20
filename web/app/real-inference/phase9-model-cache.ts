const MODEL_CACHE_NAME = "agriguard-phase9-model-1d741fbe7ee6d9a81f0bba78e7a2e84e63f4636000bda7ff8bd3223ffa28f212";
const MODEL_SIZE_BYTES = 310066343;
const MODEL_SHA256 = "1d741fbe7ee6d9a81f0bba78e7a2e84e63f4636000bda7ff8bd3223ffa28f212";

type StoredResponse = Pick<Response, "arrayBuffer">;
type ModelCache = Pick<Cache, "match" | "put" | "delete">;
type PersistentCacheStorage = Pick<CacheStorage, "open">;
type ModelCacheRuntime = {
  caches?: PersistentCacheStorage;
  fetch?: typeof fetch;
  subtle?: Pick<SubtleCrypto, "digest">;
  persistStorage?: () => Promise<boolean>;
};

export type CachedModelSource = {
  source: string | Uint8Array;
  restoredFromPersistentCache: boolean;
  storedForNextVisit: boolean;
};

const browserRuntime = (): ModelCacheRuntime => ({
  caches: typeof globalThis.caches === "undefined" ? undefined : globalThis.caches,
  fetch: typeof globalThis.fetch === "undefined" ? undefined : globalThis.fetch.bind(globalThis),
  subtle: typeof globalThis.crypto === "undefined" ? undefined : globalThis.crypto.subtle,
  persistStorage: typeof navigator === "undefined" || !navigator.storage?.persist
    ? undefined
    : () => navigator.storage.persist(),
});

function bytesToHex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes), (value) => value.toString(16).padStart(2, "0")).join("");
}

/**
 * Cache the immutable Phase 9 model in origin storage so a browser can avoid
 * transferring the ~296 MiB asset again after a reload. Browser storage remains
 * best-effort: if it is unavailable or full, inference falls back to the URL.
 */
export async function loadCachedPhase9ModelSource(
  url: string,
  runtime: ModelCacheRuntime = browserRuntime(),
  expected: { sizeBytes?: number; sha256?: string } = {},
): Promise<CachedModelSource> {
  const cacheStorage = runtime.caches;
  const fetcher = runtime.fetch;
  const subtle = runtime.subtle;
  if (!cacheStorage || !fetcher || !subtle) {
    return { source: url, restoredFromPersistentCache: false, storedForNextVisit: false };
  }

  const sizeBytes = expected.sizeBytes ?? MODEL_SIZE_BYTES;
  const sha256 = expected.sha256 ?? MODEL_SHA256;
  const persistPromise = runtime.persistStorage?.().catch(() => false) ?? Promise.resolve(false);
  let cache: ModelCache;
  try {
    cache = await cacheStorage.open(MODEL_CACHE_NAME);
  } catch {
    await persistPromise;
    return { source: url, restoredFromPersistentCache: false, storedForNextVisit: false };
  }

  try {
    const cached = await cache.match(url);
    if (cached) {
      const bytes = new Uint8Array(await cached.arrayBuffer());
      if (bytes.byteLength === sizeBytes) {
        await persistPromise;
        return { source: bytes, restoredFromPersistentCache: true, storedForNextVisit: true };
      }
      await cache.delete(url);
    }
  } catch {
    // A damaged or unavailable cache entry is discarded and fetched again.
    try { await cache.delete(url); } catch { /* best effort */ }
  }

  const response = await fetcher(url, { mode: "cors", credentials: "omit" });
  if (!response.ok) {
    await persistPromise;
    throw new Error(`The local model download failed with HTTP ${response.status}.`);
  }
  const declaredSize = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredSize) && declaredSize > 0 && declaredSize !== sizeBytes) {
    await persistPromise;
    throw new Error("The downloaded local model has an unexpected file size.");
  }

  // Cache and read in parallel to avoid making a second 296 MiB byte copy.
  const storePromise = cache.put(url, response.clone()).then(() => true, () => false);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== sizeBytes) {
    await storePromise;
    try { await cache.delete(url); } catch { /* best effort */ }
    await persistPromise;
    throw new Error("The downloaded local model has an unexpected file size.");
  }

  const actualHash = bytesToHex(await subtle.digest("SHA-256", bytes));
  if (actualHash !== sha256) {
    await storePromise;
    try { await cache.delete(url); } catch { /* best effort */ }
    await persistPromise;
    throw new Error("The downloaded local model failed its integrity check.");
  }

  const storedForNextVisit = await storePromise;
  await persistPromise;
  return { source: bytes, restoredFromPersistentCache: false, storedForNextVisit };
}
