from __future__ import annotations

import json
import sys
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
}


def normalize_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    source = settings or {}
    normalized: Dict[str, Any] = {}

    model_id = str(source.get("diarization_model_id", DEFAULT_SETTINGS["diarization_model_id"])).strip()
    normalized["diarization_model_id"] = model_id or DEFAULT_SETTINGS["diarization_model_id"]
    normalized["model_cache_path"] = str(source.get("model_cache_path", DEFAULT_SETTINGS["model_cache_path"])).strip()
    normalized["huggingface_token"] = str(source.get("huggingface_token", "")).strip()
    return normalized


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
