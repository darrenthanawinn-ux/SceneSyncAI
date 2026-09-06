"""
SceneSync AI - FastAPI Application Entrypoint
================================================
Exposes the multi-agent pre-production pipeline over a clean REST API and
serves the cinematic dark-mode frontend.

Run locally:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /api/health                  -> liveness + configuration status
    POST /api/scripts/upload          -> upload a script (PDF/TXT), starts a background job
    POST /api/scripts/analyze-text    -> submit raw script text directly, starts a background job
    GET  /api/jobs/{job_id}           -> poll job status/progress
    GET  /api/jobs/{job_id}/result    -> full structured breakdown + storyboard result
    GET  /                            -> frontend UI
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agent import Scene, SceneSyncOrchestrator, new_job_id
from backend.config import get_settings

settings = get_settings()
logger = logging.getLogger("scenesync.main")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="SceneSync AI",
    description="Autonomous multi-agent pre-production copilot for filmmakers — "
                "script breakdown, production asset extraction, and Imagen 3 storyboard generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = SceneSyncOrchestrator()
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="scenesync-job")

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
_jobs_lock = Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _new_job(filename: str = "") -> str:
    job_id = new_job_id()
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",  # queued | running | complete | error
            "stage": "Queued",
            "progress": 0,
            "filename": filename,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            "scenes": None,
            "mock_mode": settings.effective_mock_mode,
        }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _jobs[job_id]["updated_at"] = time.time()


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _run_pipeline_job(job_id: str, script_text: str) -> None:
    """Executed on a worker thread — runs the full multi-agent pipeline."""
    try:
        _update_job(job_id, status="running", stage="Starting multi-agent pipeline", progress=2)

        def progress_callback(stage: str, pct: int):
            _update_job(job_id, stage=stage, progress=min(max(pct, 0), 99))

        scenes: List[Scene] = orchestrator.run_pipeline(
            script_text=script_text,
            generate_storyboards=True,
            progress_callback=progress_callback,
        )

        if not scenes:
            _update_job(
                job_id,
                status="error",
                stage="No scenes detected",
                progress=100,
                error="Could not detect any scenes in the provided script. "
                      "Please check the file and try again.",
            )
            return

        _update_job(
            job_id,
            status="complete",
            stage="Complete",
            progress=100,
            scenes=[s.to_dict() for s in scenes],
        )
        logger.info("Job %s complete: %d scenes generated.", job_id, len(scenes))
    except Exception as exc:  # noqa: BLE001 - job runner must never crash the process
        logger.exception("Job %s failed unexpectedly.", job_id)
        _update_job(job_id, status="error", stage="Failed", progress=100, error=str(exc))


async def _dispatch_job(job_id: str, script_text: str) -> None:
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline_job, job_id, script_text)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnalyzeTextRequest(BaseModel):
    script_text: str = Field(..., min_length=1, max_length=250_000)
    title: Optional[str] = Field(default=None, max_length=200)


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    mock_mode: bool


class HealthResponse(BaseModel):
    status: str
    app_name: str
    mock_mode: bool
    gemini_model: str
    imagen_model: str
    adk_available: bool
    project_configured: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    from backend.agent import ADK_AVAILABLE

    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        mock_mode=settings.effective_mock_mode,
        gemini_model=settings.GEMINI_REASONING_MODEL,
        imagen_model=settings.IMAGEN_MODEL,
        adk_available=ADK_AVAILABLE,
        project_configured=bool(settings.GOOGLE_CLOUD_PROJECT),
    )


@app.post("/api/scripts/upload", response_model=JobCreatedResponse, tags=["pipeline"])
async def upload_script(file: UploadFile = File(...)) -> JobCreatedResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file was uploaded.")

    allowed_suffixes = (".pdf", ".txt", ".fountain", ".fdx")
    lower_name = file.filename.lower()
    if not lower_name.endswith(allowed_suffixes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Please upload one of: {', '.join(allowed_suffixes)}",
        )

    file_bytes = await file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(file_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.MAX_UPLOAD_MB}MB limit.",
        )

    try:
        script_text = orchestrator.doc_processor.extract_text(
            file_bytes, file.content_type or "", file.filename
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Document extraction failed for upload %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract text from the uploaded file: {exc}",
        ) from exc

    if not script_text or not script_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text could be extracted from this file. If it's a scanned PDF, "
                   "try uploading a text-based script instead.",
        )

    job_id = _new_job(filename=file.filename)
    await _dispatch_job(job_id, script_text)
    return JobCreatedResponse(job_id=job_id, status="queued", mock_mode=settings.effective_mock_mode)


@app.post("/api/scripts/analyze-text", response_model=JobCreatedResponse, tags=["pipeline"])
async def analyze_text(payload: AnalyzeTextRequest) -> JobCreatedResponse:
    script_text = payload.script_text.strip()
    if not script_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="script_text cannot be empty.")

    job_id = _new_job(filename=payload.title or "Pasted Script")
    await _dispatch_job(job_id, script_text)
    return JobCreatedResponse(job_id=job_id, status="queued", mock_mode=settings.effective_mock_mode)


@app.get("/api/jobs/{job_id}", tags=["pipeline"])
async def get_job_status(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    summary = {k: v for k, v in job.items() if k != "scenes"}
    summary["scene_count"] = len(job["scenes"]) if job.get("scenes") else 0
    return JSONResponse(content=summary)


@app.get("/api/jobs/{job_id}/result", tags=["pipeline"])
async def get_job_result(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    if job["status"] == "error":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=job.get("error") or "Job failed.")
    if job["status"] != "complete":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job is still processing.")

    scenes = job.get("scenes") or []
    total_cast = sorted({c for s in scenes for c in s["assets"]["cast"]})
    total_props = sorted({p for s in scenes for p in s["assets"]["props"]})
    total_locations = sorted({l for s in scenes for l in s["assets"]["locations"]})
    total_wardrobe = sorted({w for s in scenes for w in s["assets"]["wardrobe"]})

    return JSONResponse(
        content={
            "job_id": job_id,
            "filename": job.get("filename"),
            "mock_mode": job.get("mock_mode"),
            "scene_count": len(scenes),
            "scenes": scenes,
            "aggregate_assets": {
                "cast": total_cast,
                "props": total_props,
                "locations": total_locations,
                "wardrobe": total_wardrobe,
            },
        }
    )


@app.delete("/api/jobs/{job_id}", tags=["pipeline"])
async def delete_job(job_id: str) -> JSONResponse:
    with _jobs_lock:
        existed = _jobs.pop(job_id, None) is not None
    if not existed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return JSONResponse(content={"deleted": True, "job_id": job_id})


# ---------------------------------------------------------------------------
# Global error handling (zero-crash guarantee)
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:  # noqa: ANN001
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. The SceneSync AI team has been notified.", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Frontend static hosting
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        index_path = FRONTEND_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend not found.")
        return FileResponse(str(index_path))

else:
    logger.warning("Frontend directory not found at %s — UI will not be served.", FRONTEND_DIR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=(settings.APP_ENV == "development"))
