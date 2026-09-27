from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import WEB_DIR, WORK_DIR, WORDLIST_PATH, load_settings, save_settings
from .jobs import JobManager

app = FastAPI(title="BeepClean", docs_url=None, redoc_url=None)
settings = load_settings()
jobs = JobManager(settings)

UPLOAD_DIR = WORK_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".ts",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/settings")
def get_settings() -> dict:
    return settings.__dict__


@app.post("/api/settings")
async def update_settings(request: Request) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object.")
    for key, value in payload.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    save_settings(settings)
    jobs.settings = settings
    return settings.__dict__


def _read_wordlist() -> dict:
    try:
        return json.loads(WORDLIST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"stems": [], "extra": [], "allow": []}


@app.get("/api/wordlist")
def get_wordlist() -> dict:
    return _read_wordlist()


@app.post("/api/wordlist")
async def update_wordlist(request: Request) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object.")
    cleaned: dict[str, list[str]] = {}
    for key in ("stems", "extra", "allow"):
        value = payload.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a list of words.")
        cleaned[key] = [word.strip() for word in value if word.strip()]
    WORDLIST_PATH.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    return cleaned


@app.post("/api/upload")
async def upload(file: UploadFile = Form(...)) -> dict:
    name = Path(file.filename or "upload.mp4").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or name}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )
    target = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is larger than {settings.max_upload_mb} MB.",
                    )
                handle.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")
    job = jobs.create(name, target)
    return job.snapshot()


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return job.snapshot()


@app.post("/api/job/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    if not jobs.cancel(job_id):
        raise HTTPException(status_code=409, detail="Job is not running.")
    return {"ok": True}


@app.get("/api/job/{job_id}/download")
def job_download(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    if job.output is None or not job.output.exists():
        raise HTTPException(status_code=409, detail="Output is not ready yet.")
    stem = Path(job.name).stem or "video"
    return FileResponse(
        job.output, media_type="video/mp4", filename=f"{stem}-beeped.mp4"
    )


@app.get("/api/job/{job_id}/video")
def job_video(job_id: str):
    job = jobs.get(job_id)
    if job is None or job.output is None or not job.output.exists():
        raise HTTPException(status_code=404, detail="Output is not ready yet.")
    return FileResponse(
        job.output,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-store"},
    )


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "jobs": len(jobs.jobs)})


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
