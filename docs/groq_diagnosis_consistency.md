# Groq diagnosis consistency

Groq assessment requests now set `temperature: 0` and a fixed seed, and the system prompt asks for one canonical disease label supported by visible evidence. If similar diseases cannot be distinguished, the prompt directs the model to return an uncertain result instead of switching among names.

This reduces sampling variation but cannot guarantee identical responses. Groq documents seeded generation as best-effort; backend/model updates and vision interpretation can still change a response. This change does not establish diagnostic accuracy or make the result a confirmed diagnosis.

Verification used mocked provider requests only:

```powershell
cd D:\PlantDiseaseAI\web
npx.cmd tsx --test tests\groq-provider.test.mjs tests\assess-plant.test.mjs
npm.cmd run typecheck
```

Results: 15 focused tests passed and TypeScript type-check passed. No live provider request was made. Restart the Next.js development server before checking the updated request settings.
