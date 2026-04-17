from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
SETTINGS_FILE = LOGS_DIR / "genesis_dia_settings.json"
SECRETS_FILE = LOGS_DIR / "genesis_dia_secrets.json"
LOG_FILE = LOGS_DIR / "diarization_log.jsonl"

DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
HISTORY_MAX_LEN = 100


def resolve_model_cache_path(cache_path: str) -> str:
    normalized = str(cache_path or "").strip()
    if not normalized:
        return ""

    path = Path(normalized).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve(strict=False))


settings_lock = threading.Lock()
model_load_lock = threading.Lock()
history_lock = threading.Lock()
task_status_lock = threading.Lock()
task_runtime_lock = threading.Lock()

current_settings: Dict[str, Any] = {}
current_task_status: Dict[str, Any] = {
    "task_name": "Idle",
    "progress": 0.0,
    "details": "Server is ready.",
}
diarization_pipeline: Dict[str, Any] = {"pipeline": None, "model_identifier": None}
diarization_history = deque(maxlen=HISTORY_MAX_LEN)
task_runtime_state: Dict[str, Any] = {
    "worker_running": False,
    "pending_requests": 0,
    "active_request_id": None,
    "active_started_at": None,
    "active_audio_seconds": 0.0,
    "last_completed_at": None,
    "last_duration_ms": None,
    "last_error": None,
    "total_requests_processed": 0,
}
