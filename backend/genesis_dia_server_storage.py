from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .genesis_dia_server_globals import (
    DEFAULT_DIARIZATION_MODEL_ID,
    LOG_FILE,
    LOGS_DIR,
    SETTINGS_FILE,
)


DEFAULT_SETTINGS: Dict[str, Any] = {
    "diarization_model_id": DEFAULT_DIARIZATION_MODEL_ID,
    "model_cache_path": ".\\models",
    "huggingface_token": "",
    "gpu_device": "auto",
    # Load + warm the pipeline right after start instead of inside the first request (~13 s).
    "preload_model_on_startup": True,
    # Run the embedding network only for speakers active in a chunk (identical output, ~1.5x faster).
    "skip_inactive_embeddings": True,
    # pyannote's config ships 32; 16 halves the embedding activations (~1 GB less peak VRAM on
    # the shared GPU) while the ResNet34 is already ~90 % busy at 32, so throughput barely moves.
    "embedding_batch_size": 16,
    # Return the segmentation phase's cached blocks before the embedding phase allocates its own.
    "release_between_phases": True,
}

EMBEDDING_BATCH_SIZE_RANGE = (1, 128)


def normalize_int_setting(value: Any, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return min(max(number, minimum), maximum)


def normalize_bool_setting(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_device_setting(value: Any) -> str:
    """Accept ``auto``, ``cpu``, ``cuda`` and ``cuda:N``; anything else means ``auto``."""

    text = str(value or "").strip().lower()
    if text == "cpu":
        return "cpu"
    if text == "cuda":
        return "cuda:0"
    match = re.fullmatch(r"cuda:(\d+)", text)
    if match:
        return f"cuda:{int(match.group(1))}"
    return "auto"


def normalize_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    source = settings or {}
    normalized: Dict[str, Any] = {}

    model_id = str(source.get("diarization_model_id", DEFAULT_SETTINGS["diarization_model_id"])).strip()
    normalized["diarization_model_id"] = model_id or DEFAULT_SETTINGS["diarization_model_id"]
    normalized["model_cache_path"] = str(source.get("model_cache_path", DEFAULT_SETTINGS["model_cache_path"])).strip()
    normalized["huggingface_token"] = str(source.get("huggingface_token", "")).strip()
    normalized["gpu_device"] = normalize_device_setting(source.get("gpu_device", DEFAULT_SETTINGS["gpu_device"]))
    for key in ("preload_model_on_startup", "skip_inactive_embeddings", "release_between_phases"):
        normalized[key] = normalize_bool_setting(source.get(key), DEFAULT_SETTINGS[key])
    normalized["embedding_batch_size"] = normalize_int_setting(
        source.get("embedding_batch_size"), DEFAULT_SETTINGS["embedding_batch_size"], *EMBEDDING_BATCH_SIZE_RANGE
    )
    return normalized


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file in the same directory + os.replace, so a crash never leaves a half-written file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_settings() -> Dict[str, Any]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS.copy()

    try:
        return normalize_settings(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[FEHLER] Konnte Einstellungsdatei nicht laden: {exc}", file=sys.stderr)
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_settings(settings)
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"[INFO] Einstellungen in '{SETTINGS_FILE}' gespeichert.")
    except OSError as exc:
        print(f"[FEHLER] Konnte Einstellungen nicht speichern: {exc}", file=sys.stderr)
    return normalized


def log_diarization(log_data: Dict[str, Any]) -> None:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(log_data, ensure_ascii=True) + "\n")
    except OSError as exc:
        print(f"[FEHLER] Konnte Log-Eintrag nicht schreiben: {exc}", file=sys.stderr)
