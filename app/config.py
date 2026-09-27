from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
WORK_DIR = ROOT / "work"

for _directory in (DATA_DIR, WORK_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

SETTINGS_PATH = DATA_DIR / "settings.json"
WORDLIST_PATH = DATA_DIR / "wordlist.json"

BLUR_MODES = ("blur", "pixelate", "blackout")
MODELS = ("tiny.en", "base.en", "small.en")


@dataclass
class Settings:
    model: str = "tiny.en"
    compute_type: str = "int8"
    cpu_threads: int = 4
    vad_threshold: float = 0.2
    beam_size: int = 5
    condition_on_previous_text: bool = True
    beep_freq: float = 1000.0
    beep_amplitude: float = 0.55
    beep_min_ms: int = 120
    beep_max_ms: int = 600
    edge_pad_ms: int = 25
    fade_ms: int = 8
    min_confidence: float = 0.0
    blur_mode: str = "blur"
    ocr_enabled: bool = True
    ocr_fps: float = 3.0
    ocr_band: float = 0.30
    ocr_max_width: int = 640
    ocr_min_conf: float = 0.5
    ocr_merge_gap: float = 1.0
    ocr_min_duration: float = 0.30
    max_resolution: int = 1080
    crf: int = 23
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    report_only: bool = False
    max_upload_mb: int = 2048

    def clamped(self) -> "Settings":
        self.blur_mode = self.blur_mode if self.blur_mode in BLUR_MODES else "blur"
        if self.model not in MODELS:
            self.model = "tiny.en"
        self.cpu_threads = max(1, min(16, int(self.cpu_threads)))
        self.vad_threshold = max(0.0, min(0.9, float(self.vad_threshold)))
        self.beam_size = max(1, min(10, int(self.beam_size)))
        self.beep_freq = max(100.0, min(8000.0, float(self.beep_freq)))
        self.beep_amplitude = max(0.0, min(1.0, float(self.beep_amplitude)))
        self.beep_min_ms = max(20, min(2000, int(self.beep_min_ms)))
        self.beep_max_ms = max(self.beep_min_ms, min(4000, int(self.beep_max_ms)))
        self.edge_pad_ms = max(0, min(300, int(self.edge_pad_ms)))
        self.fade_ms = max(0, min(50, int(self.fade_ms)))
        self.min_confidence = max(0.0, min(1.0, float(self.min_confidence)))
        self.ocr_fps = max(0.5, min(15.0, float(self.ocr_fps)))
        self.ocr_band = max(0.05, min(1.0, float(self.ocr_band)))
        self.ocr_max_width = max(320, min(1920, int(self.ocr_max_width)))
        self.ocr_min_conf = max(0.0, min(1.0, float(self.ocr_min_conf)))
        self.ocr_merge_gap = max(0.0, min(5.0, float(self.ocr_merge_gap)))
        self.ocr_min_duration = max(0.0, min(5.0, float(self.ocr_min_duration)))
        self.max_resolution = max(240, min(4320, int(self.max_resolution)))
        self.crf = max(0, min(51, int(self.crf)))
        self.max_upload_mb = max(1, min(16384, int(self.max_upload_mb)))
        return self


def load_settings() -> Settings:
    if not SETTINGS_PATH.exists():
        return Settings()
    try:
        stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    known = {f.name: f.type for f in fields(Settings)}
    clean: dict[str, object] = {}
    for key, value in stored.items():
        if key not in known:
            continue
        try:
            if isinstance(value, bool):
                clean[key] = bool(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                clean[key] = int(value)
            elif isinstance(value, float):
                clean[key] = float(value)
            else:
                clean[key] = str(value)
        except (TypeError, ValueError):
            continue
    return Settings(**clean).clamped()


def save_settings(settings: Settings) -> Settings:
    settings.clamped()
    SETTINGS_PATH.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    return settings
