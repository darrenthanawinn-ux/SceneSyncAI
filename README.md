# 🎬 SceneSync AI

**Autonomous multi-agent pre-production copilot for filmmakers, screenwriters, and studio crews.**

Built for the **Google Cloud Summer Blockbuster Hackathon** — Replit & Gemini Enterprise Agent Platform track.

Upload a raw script (PDF or text) and SceneSync AI's coordinated agent pipeline will:

1. **Parse** the document (Document Processing) into clean screenplay text.
2. **Break it down** into structured scenes using Gemini multi-step reasoning.
3. **Extract production assets** — cast, props, locations, wardrobe, SFX notes — per scene.
4. **Generate cinematic 16:9 concept art / storyboard panels** for every scene using **Vertex AI Imagen 3**.

All of this is orchestrated by a native **Google Cloud Agent Development Kit (ADK)** agent, exposed through a FastAPI backend, and presented in a polished, responsive, dark-mode cinematic UI.

---

## ✨ Why it wins

| Requirement | How SceneSync AI delivers |
|---|---|
| **ADK / Agent Engine** | `backend/agent.py` builds a native `google.adk` `Agent` with `FunctionTool`s wrapping every pipeline stage — deployable as-is via `vertexai.agent_engines.create()`. |
| **GenMedia Core** | Document parsing → Gemini reasoning → **Imagen 3** 16:9 storyboard generation, fully wired end-to-end. |
| **Vertex AI Search grounding** | Config hook (`VERTEX_SEARCH_DATASTORE_ID`) ready for grounded generation. |
| **Replit-native** | `.replit` + `replit.nix` included; boots with one click, zero manual config. |
| **Security & polish** | Secret Manager resolution pattern, Gemini safety settings on every call, cinematic responsive Tailwind UI. |
| **Zero-error guarantee** | Every Google Cloud call is wrapped in defensive fallbacks (`MOCK_MODE`) so the app **never crashes**, even with zero cloud credentials attached — perfect for live judging. |

---

## 🗂 Project structure

```
SceneSync_AI/
├── backend/
│   ├── __init__.py
│   ├── config.py        # Settings, env vars, Secret Manager pattern, safety settings
│   ├── agent.py          # Multi-agent pipeline: Document → Breakdown → Assets → Imagen 3 + ADK Agent
│   └── main.py            # FastAPI app: upload/analyze endpoints, job orchestration, static hosting
├── frontend/
│   └── index.html          # Cinematic dark-mode single-page UI (Tailwind + vanilla JS)
├── requirements.txt
├── .env.example
├── .replit
├── replit.nix
└── README.md
```

---

## 🚀 Quick start on Replit

1. **Import this project** into Replit (Create Repl → Import from folder/zip, or drag-and-drop this whole directory).
2. Open the **Secrets** tab (🔒 icon) and add the environment variables from `.env.example` — at minimum:
   - `GOOGLE_CLOUD_PROJECT` — your Google Cloud project ID
   - `GOOGLE_CLOUD_LOCATION` — e.g. `us-central1`
   - Authenticate your Repl to Google Cloud (see below).
3. Click **Run**. Replit will install `requirements.txt` and start Uvicorn automatically (see `.replit`).
4. Open the webview — you'll land on the SceneSync AI UI, ready to accept a script upload.

> **No Google Cloud project yet?** No problem — SceneSync AI automatically runs in **MOCK_MODE**, using deterministic local simulations (regex-based scene parsing, heuristic asset extraction, and generated placeholder concept art) so the entire pipeline and UI work flawlessly with zero setup. This is ideal for a fast first run; attach real credentials any time to unlock live Gemini + Imagen 3 generation.

---

## 💻 Local setup (step-by-step)

### 1. Clone / extract the project

```bash
unzip SceneSync_AI.zip
cd SceneSync_AI
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
```

Leave `GOOGLE_CLOUD_PROJECT` blank (or set `MOCK_MODE=true`) to run entirely offline in simulation mode.

### 5. Authenticate with Google Cloud (for live generation)

**Option A — Application Default Credentials (recommended for local dev):**

```bash
gcloud auth application-default login
gcloud config set project your-gcp-project-id
```

**Option B — Service account key file:**

1. In Google Cloud Console → IAM & Admin → Service Accounts, create a service account with the **Vertex AI User** role.
2. Download its JSON key as `service-account.json` into the project root.
3. Set `GOOGLE_APPLICATION_CREDENTIALS=./service-account.json` in `.env`.

**Enable the required APIs** on your project:

```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable documentai.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### 6. Run the app

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit **http://localhost:8000** in your browser.

---

## 🔑 Environment variables reference

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | For live mode | Your GCP project ID. If unset, the app runs in `MOCK_MODE`. |
| `GOOGLE_CLOUD_LOCATION` | No (default `us-central1`) | Vertex AI region. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Local dev only | Path to a service account JSON key. |
| `MOCK_MODE` | No | Force local simulation mode regardless of credentials. |
| `GEMINI_REASONING_MODEL` | No | Gemini model for scene breakdown & asset extraction. |
| `IMAGEN_MODEL` | No | Imagen model used for storyboard art (default `imagen-3.0-generate-002`). |
| `VERTEX_SEARCH_DATASTORE_ID` | No | Vertex AI Search data store resource name, for grounding. |
| `MAX_SCENES_PER_SCRIPT` | No | Cap on scenes processed per upload (default 40). |
| `MAX_UPLOAD_MB` | No | Max upload size in MB (default 15). |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins, or `*`. |

See `.env.example` for the complete, commented list.

---

## 🧠 How the pipeline works

```
Upload (.pdf / .txt / .fountain / .fdx)
        │
        ▼
DocumentProcessor            → Document AI (or pypdf fallback) → clean text
        │
        ▼
ScriptBreakdownAgent          → Gemini structured JSON reasoning → Scene[]
        │                        (regex slugline fallback if Gemini unavailable)
        ▼
AssetExtractionAgent          → per-scene cast/props/locations/wardrobe/SFX
        │                        (heuristic keyword fallback if Gemini unavailable)
        ▼
StoryboardAgent                → Vertex AI Imagen 3, 16:9 cinematic concept art
        │                        (generated SVG placeholder fallback if Imagen unavailable)
        ▼
SceneSyncOrchestrator          → assembled JSON result, served to the frontend
```

The FastAPI layer (`backend/main.py`) runs this pipeline as a **background job** with live progress polling (`GET /api/jobs/{job_id}`), so the UI shows real-time stage-by-stage progress bars while agents work.

### Deploying the ADK agent to Vertex AI Agent Engine

`backend/agent.py` exposes `SceneSyncOrchestrator.build_adk_agent()`, which constructs a native `google.adk.agents.Agent` with the same tool functions used by the REST API. To deploy it:

```python
from vertexai import agent_engines
from backend.agent import SceneSyncOrchestrator

orchestrator = SceneSyncOrchestrator()
adk_agent = orchestrator.build_adk_agent()

remote_agent = agent_engines.create(
    adk_agent,
    requirements=["google-cloud-aiplatform[agent_engines,adk]>=1.101.0"],
)
print(remote_agent.resource_name)
```

Store the returned resource name in `AGENT_ENGINE_RESOURCE_NAME` for future reference.

---

## 🔌 API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check + current configuration (mock mode, models, ADK availability). |
| `POST` | `/api/scripts/upload` | Multipart file upload (`file`). Returns `{job_id, status, mock_mode}`. |
| `POST` | `/api/scripts/analyze-text` | JSON body `{script_text, title?}`. Returns `{job_id, status, mock_mode}`. |
| `GET` | `/api/jobs/{job_id}` | Poll job status/progress/stage. |
| `GET` | `/api/jobs/{job_id}/result` | Full structured breakdown once `status == "complete"`. |
| `DELETE` | `/api/jobs/{job_id}` | Remove a job from memory. |

Interactive OpenAPI docs are available at **`/docs`** once the server is running.

---

## 🛡 Security & safety notes

- **Secret Manager pattern**: `backend/config.py` resolves sensitive values from Google Cloud Secret Manager first, falling back to environment variables — the same code works locally and in production without changes.
- **Gemini safety settings**: every generative call applies `BLOCK_MEDIUM_AND_ABOVE` thresholds across hate speech, dangerous content, sexually explicit content, and harassment categories.
- **No secrets in code**: nothing is hard-coded; all credentials come from environment variables / Secret Manager / attached IAM identities.
- **Defensive error handling**: a global FastAPI exception handler ensures the API always returns clean JSON errors instead of crashing, and every Google Cloud SDK call degrades gracefully to a local fallback.

---

## 🧰 Tech stack

- **Backend**: Python 3.12, FastAPI, Uvicorn, Pydantic v2
- **AI**: Vertex AI Gemini (multi-step reasoning), Vertex AI Imagen 3 (image generation), Google Cloud Document AI, Google Cloud Agent Development Kit (ADK)
- **Frontend**: HTML5, Tailwind CSS (CDN), vanilla JavaScript, Font Awesome
- **Deployment**: Replit-native (`.replit`, `replit.nix`), Cloud Run-ready

---

## 🐛 Troubleshooting

- **"Simulated mode" badge showing even though I set `GOOGLE_CLOUD_PROJECT`**: confirm you've authenticated (`gcloud auth application-default login`) and that the Vertex AI API is enabled on that project.
- **PDF text extraction returns empty**: the PDF may be a scanned image with no embedded text layer. Configure a real Document AI processor ID in `DocumentProcessor._extract_with_document_ai`, or upload a text-based script instead.
- **Imagen 3 calls fail with a permission error**: ensure your identity has the **Vertex AI User** IAM role and that billing is enabled on the project.
- **Port already in use on Replit**: `.replit` binds to `8000` — stop any other running process or change `APP_PORT` in your Secrets and update `.replit` accordingly.

---

Built with 🎥 for the Google Cloud Summer Blockbuster Hackathon.
