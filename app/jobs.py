from __future__ import annotations

import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import WORK_DIR, Settings
from .pipeline import Report, run, write_report

KEEP_JOBS = 40


@dataclass
class Job:
    id: str
    name: str
    folder: Path
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting in line"
    error: str = ""
    report: Report | None = None
    output: Path | None = None
    created: float = field(default_factory=time.time)
    cancel_requested: bool = False
    settings: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "created": self.created,
            "report": self.report.__dict__ if self.report else None,
            "has_output": self.output is not None and self.output.exists(),
        }


class JobManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="beepclean")

    def create(self, name: str, source: Path) -> Job:
        job_id = uuid.uuid4().hex[:12]
        folder = WORK_DIR / job_id
        folder.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, name=name, folder=folder)
        with self.lock:
            self.jobs[job_id] = job
            self._prune()
        self.pool.submit(self._execute, job, source)
        return job

    def _prune(self) -> None:
        if len(self.jobs) <= KEEP_JOBS:
            return
        finished = sorted(
            (job for job in self.jobs.values() if job.status in ("done", "error", "cancelled")),
            key=lambda job: job.created,
        )
        while len(self.jobs) > KEEP_JOBS and finished:
            self._discard(finished.pop(0))

    def _discard(self, job: Job) -> None:
        self.jobs.pop(job.id, None)
        shutil.rmtree(job.folder, ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in ("done", "error", "cancelled"):
            return False
        job.cancel_requested = True
        return True

    def _execute(self, job: Job, source: Path) -> None:
        def cancelled() -> bool:
            return job.cancel_requested

        def on_progress(fraction: float, message: str) -> None:
            job.progress = max(0.0, min(1.0, fraction))
            job.message = message
            job.stage = message

        job.status = "running"
        job.stage = "probe"
        job.message = "Starting"
        destination = job.folder / "output.mp4"
        try:
            report = run(source, destination, self.settings, on_progress, cancelled)
            if cancelled():
                raise InterruptedError("Cancelled")
            job.report = report
            job.output = destination
            job.status = "done"
            job.stage = "done"
            job.progress = 1.0
            job.message = "Finished"
            write_report(job.folder / "report.json", report)
        except InterruptedError:
            job.status = "cancelled"
            job.stage = "cancelled"
            job.message = "Cancelled"
            if destination.exists():
                destination.unlink()
        except Exception as error:  # noqa: BLE001
            job.status = "error"
            job.stage = "error"
            job.error = f"{type(error).__name__}: {error}"
            job.message = "Failed"
            (job.folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if destination.exists():
                destination.unlink()
        finally:
            source.unlink(missing_ok=True)
