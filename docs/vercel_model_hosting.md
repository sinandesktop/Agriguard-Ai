# Hosting the Phase 9 model for Vercel

The Phase 9 ONNX model is 310,066,343 bytes (about 296 MiB). It is excluded from Git and is larger than Vercel's standard 250 MB Function bundle limit, so the application must not depend on packaging it inside a Vercel Function.

## Deployment design

- Browser feature extraction and ONNX inference remain local to the user's browser.
- In local development, the loader uses `/api/models/phase9`, which streams the model from the existing ignored local artifact.
- In production, the loader uses `NEXT_PUBLIC_PHASE9_MODEL_URL`, a public HTTPS URL to an immutable copy of the exact Phase 9 ONNX artifact.
- No uploaded leaf photo is sent to this model asset host. The browser downloads the model asset only; inference still runs in ONNX Runtime Web.
- The model is public and can be downloaded by anyone with its URL. Do not put storage credentials in this variable or in client code.

## One-time Vercel setup

1. In Vercel Storage, create a **public** Blob store. Public access is needed because the browser downloads this model without a user-specific storage token.
2. Upload this exact local file as an immutable, versioned object:
   `ml/models/phase9_domain_adaptation/phase9_exp_20261019_v1/selected_model.onnx`
3. Verify the local file before uploading:
   - Size: `310066343` bytes
   - SHA-256: `1d741fbe7ee6d9a81f0bba78e7a2e84e63f4636000bda7ff8bd3223ffa28f212`
4. Copy the resulting public HTTPS Blob URL and set it as `NEXT_PUBLIC_PHASE9_MODEL_URL` in the Vercel project's **Production** environment. Set it for Preview too if previews should provide local inference.
5. Redeploy. `NEXT_PUBLIC_` values are embedded into the browser bundle at build time, so changing the value requires a new build/deployment.
6. Open the production site, choose Tomato or Potato, and verify in browser DevTools Network that the model request returns HTTP 200 with the expected ONNX file size. Then run a local assessment. A browser request to a different origin also requires that host to permit CORS for the deployed site; resolve any CORS error at the asset host before considering the deployment verified.

Vercel documents public Blob objects as directly downloadable CDN assets, supports caching blobs up to 512 MB, and charges for data transfer. Since this model is large, users may download roughly 296 MiB on their first visit to local assessment; keep model loading lazy and monitor storage and transfer usage. See [Vercel Blob public storage](https://vercel.com/docs/vercel-blob/public-storage) and [Vercel Blob usage and pricing](https://vercel.com/docs/vercel-blob/usage-and-pricing).

## Local development

Leave `NEXT_PUBLIC_PHASE9_MODEL_URL` empty in local environment configuration to use the existing local model route. Do not add a real storage token to this variable. If the model is absent locally, the route reports it unavailable.

## Troubleshooting

- The form says local assessment is unavailable: the production build did not receive `NEXT_PUBLIC_PHASE9_MODEL_URL`; configure it and redeploy.
- The browser receives 403/404: check that the Blob object is public and the URL points to the uploaded model.
- The browser reports a CORS/network error: configure the asset host's CORS policy for the production origin or use a compatible public HTTPS asset host.
- The download succeeds but ONNX interface validation fails: verify that the uploaded object is the frozen Phase 9 artifact above; do not replace it with Phase 3 or another model.
- Inference errors after model loading: inspect the browser console for ONNX Runtime Web/WASM errors. The API key and external provider settings are unrelated to local inference.
