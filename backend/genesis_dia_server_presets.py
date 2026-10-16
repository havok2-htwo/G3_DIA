"""Named settings presets: built-in GPU tiers plus presets the operator saves from the dashboard.

Same model as G3_OmniVoice and G3_RERANK: a preset is a bag of settings values. Loading one fills the settings form;
the operator reviews it and saves through the normal validated PUT /api/admin/settings, so a preset never bypasses
validation.

Built-in tiers (one name per target machine, identical across the G3 services):
- "32GB VRAM"  RTX 5090, every G3 service on one card, whole-card peak <= 30 GB.
- "24GB VRAM"  RTX 4090, every G3 service on one card, whole-card peak <= 23 GB.
- "8+24 VRAM"  3090 (24 GB) + 2070S (8 GB) box, DIA on the 8 GB card.
The number in front names the card the service runs on. Presets with a target card size pick gpu_device on load:
the CUDA device whose memory matches (machines number their cards differently), see resolve_presets.
There is no "24+8 VRAM" tier: in the two-card box DIA is the only service on the 2070S, so the 3090 stays free for
OmniVoice, Whisper, EMBED and RERANK and DIA no longer shares the GPU lease with Whisper.

DIA always runs fp32 (fp16 gives NaN, bf16 changes the diarization), so there is no precision knob here.

User presets live in logs/genesis_dia_presets.json; they cannot reuse a built-in name.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import genesis_dia_server_storage as storage
from .genesis_dia_server_globals import LOGS_DIR
from .genesis_dia_server_storage import atomic_write_text, normalize_settings

PRESETS_FILE = LOGS_DIR / "genesis_dia_presets.json"
# A fresh install (no saved settings yet) starts with this tier; the same in every G3 service.
FIRST_START_PRESET = "24GB VRAM"
PRESET_NAME_MAX_LENGTH = 64

# Keys a preset may set: device and VRAM behaviour. The model id, the model cache path and the Hugging Face token
# stay out, so loading a preset never touches them.
PRESET_KEYS = (
    "gpu_device",
    "embedding_batch_size",
    "release_between_phases",
    "skip_inactive_embeddings",
    "preload_model_on_startup",
)

# Tolerance when matching a preset's card size to a detected card (a "24 GB" card reports ~23.6-24.0 GiB).
CARD_MATCH_TOLERANCE_MIB = 2048


# Every tier runs the same values: DIA's peak is small and flat in audio length, so the cards differ only in speed.
_MEASURED_VALUES: Dict[str, Any] = {
    "embedding_batch_size": 16,
    "skip_inactive_embeddings": True,
    "release_between_phases": False,
    "preload_model_on_startup": True,
}

# Expected process VRAM with these values, measured in-process on the RTX 5090 (fp32, pyannote community-1).
_VRAM_NOTE = (
    "Expected process VRAM (measured on the RTX 5090): ~0.64 GB idle, ~2.05 GB request peak (independent of audio "
    "length, 60 min checked), ~2.5 GB startup warm-up."
)

# ======================================================================================================================
# BUILT-IN TIER VALUES
# Measured on the RTX 5090 (5 and 60 min audio, batch 8/16/32/64, release on/off, skip on/off): batch 16 is the knee
# (241x realtime; 32 adds +1.95 GB for no speed), release_between_phases=False peaks ~0.5 GB lower at the same speed,
# skip_inactive_embeddings is 1.65x faster for the same peak. The 4090 and the 2070S were not measured.
# ======================================================================================================================
BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "32GB VRAM": {
        "description": "RTX 5090, all G3 services on one card (whole-card peak <= 30 GB). Batch 16, ~240x realtime. "
        + _VRAM_NOTE,
        "target_vram_gb": None,
        "values": dict(_MEASURED_VALUES),
    },
    "24GB VRAM": {
        "description": "RTX 4090, all G3 services on one card (whole-card peak <= 23 GB). Same values as 32GB VRAM; "
        "batch 8 would save ~270 MiB for ~18 % less speed. " + _VRAM_NOTE,
        "target_vram_gb": None,
        "values": dict(_MEASURED_VALUES),
    },
    "8+24 VRAM": {
        "description": "3090 + 2070S box, DIA alone on the 2070S (8 GB). " + _VRAM_NOTE + " Speed on the 2070S is an "
        "estimate, not measured: ~25-40x realtime (60 min audio in ~2-3 min).",
        "target_vram_gb": 8,
        "values": dict(_MEASURED_VALUES),
    },
}

_lock = threading.RLock()


def detect_cuda_memory_mib() -> List[int]:
    """Total memory of every visible CUDA device in MiB, in cuda:N order (empty without CUDA)."""

    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [
            int(torch.cuda.get_device_properties(index).total_memory // (1024 * 1024))
            for index in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # a broken driver must not break the settings page
        print(f"[WARNUNG-DIA] GPU-Erkennung fuer Presets fehlgeschlagen: {exc}", file=sys.stderr, flush=True)
        return []


def _clean_name(name: str) -> str:
    clean = " ".join(str(name or "").split())
    if not clean:
        raise ValueError("Preset name is required.")
    if len(clean) > PRESET_NAME_MAX_LENGTH:
        raise ValueError(f"Preset names have at most {PRESET_NAME_MAX_LENGTH} characters.")
    return clean


def preset_values(values: Dict[str, Any]) -> Dict[str, Any]:
    """Only PRESET_KEYS, each normalised exactly like a saved setting (unknown keys are dropped)."""

    sent = {key: values[key] for key in PRESET_KEYS if key in values}
    normalized = normalize_settings(sent)
    return {key: normalized[key] for key in sent}


def _load_user_presets() -> Dict[str, Dict[str, Any]]:
    if not PRESETS_FILE.exists():
        return {}
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[FEHLER] Konnte Presets nicht laden: {exc}", file=sys.stderr, flush=True)
        return {}
    presets: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for name, entry in data.items():
            if isinstance(entry, dict) and isinstance(entry.get("values"), dict) and name not in BUILTIN_PRESETS:
                presets[str(name)] = {
                    "name": str(name),
                    "values": preset_values(entry["values"]),
                    "created_at": entry.get("created_at"),
                }
    return presets


def _write_user_presets(presets: Dict[str, Dict[str, Any]]) -> None:
    atomic_write_text(PRESETS_FILE, json.dumps(presets, indent=2, ensure_ascii=True))


def _match_card(target_gb: Optional[int], cards_mib: List[int]) -> Tuple[Optional[str], Optional[str]]:
    """gpu_device for a preset's card size on this machine, plus a note when no card matches."""

    if not target_gb:
        return None, None
    target_mib = target_gb * 1024
    for index, total_mib in enumerate(cards_mib):
        if abs(total_mib - target_mib) <= CARD_MATCH_TOLERANCE_MIB:
            return f"cuda:{index}", None
    detected = ", ".join(f"GPU {index}: {total / 1024:.0f} GB" for index, total in enumerate(cards_mib)) or "none"
    return None, f"No {target_gb} GB card detected (cards: {detected}); gpu_device is left unchanged."


def resolve_presets() -> List[Dict[str, Any]]:
    """Built-ins first (in tier order), then user presets by name. Built-ins with a card size get gpu_device."""

    cards_mib = detect_cuda_memory_mib()
    result: List[Dict[str, Any]] = []
    for name, preset in BUILTIN_PRESETS.items():
        values = dict(preset["values"])
        device, note = _match_card(preset.get("target_vram_gb"), cards_mib)
        if device:
            values["gpu_device"] = device
        result.append(
            {
                "name": name,
                "built_in": True,
                "description": preset.get("description", ""),
                "target_vram_gb": preset.get("target_vram_gb"),
                "created_at": None,
                "note": note,
                "values": values,
            }
        )
    with _lock:
        user = _load_user_presets()
    for name in sorted(user, key=str.lower):
        entry = user[name]
        result.append(
            {
                "name": name,
                "built_in": False,
                "description": "",
                "target_vram_gb": None,
                "created_at": entry.get("created_at"),
                "note": None,
                "values": entry["values"],
            }
        )
    return result


def save_user_preset(name: str, values: Dict[str, Any]) -> Dict[str, Any]:
    clean = _clean_name(name)
    if clean in BUILTIN_PRESETS:
        raise ValueError(f"'{clean}' is a built-in preset; save under another name.")
    cleaned = preset_values(values)
    if not cleaned:
        raise ValueError("The preset contains no settings.")
    with _lock:
        presets = _load_user_presets()
        created = (presets.get(clean) or {}).get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S")
        presets[clean] = {"name": clean, "values": cleaned, "created_at": created}
        _write_user_presets(presets)
    return {
        "name": clean,
        "built_in": False,
        "description": "",
        "target_vram_gb": None,
        "created_at": created,
        "note": None,
        "values": cleaned,
    }


def delete_user_preset(name: str) -> bool:
    clean = " ".join(str(name or "").split())
    if clean in BUILTIN_PRESETS:
        raise ValueError(f"'{clean}' is a built-in preset and cannot be deleted.")
    with _lock:
        presets = _load_user_presets()
        if clean not in presets:
            return False
        presets.pop(clean)
        _write_user_presets(presets)
    return True


def load_settings_or_first_start_preset() -> Tuple[Dict[str, Any], bool]:
    """The saved settings; without a settings file (first start) the defaults, the entrypoint seed (container paths)
    and FIRST_START_PRESET on top, saved so the dashboard shows what runs. Returns (settings, applied)."""
    if storage.SETTINGS_FILE.exists():
        return storage.load_settings(), False
    values = dict(BUILTIN_PRESETS[FIRST_START_PRESET]["values"])
    return storage.save_settings({**storage.DEFAULT_SETTINGS, **storage.load_settings_seed(), **values}), True


__all__ = [
    "BUILTIN_PRESETS",
    "FIRST_START_PRESET",
    "PRESETS_FILE",
    "PRESET_KEYS",
    "delete_user_preset",
    "detect_cuda_memory_mib",
    "load_settings_or_first_start_preset",
    "preset_values",
    "resolve_presets",
    "save_user_preset",
]
