from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from dotenv import load_dotenv

from .genesis_dia_server_globals import (
    current_settings,
    current_task_status,
    diarization_pipeline,
    model_load_lock,
    resolve_model_cache_path,
    settings_lock,
    task_status_lock,
)

HUGGING_FACE_TOKEN = None


def _resolve_huggingface_token() -> str | None:
    with settings_lock:
        settings_token = str(current_settings.get("huggingface_token", "")).strip()
    if settings_token:
        return settings_token

    load_dotenv()
    env_token = str(
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    return env_token or None


def _repo_dir_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def _resolve_diarization_pretrained_source(model_id: str, cache_path: str) -> tuple[str, str | None]:
    if not cache_path:
        return model_id, None

    repo_cache_dir = Path(cache_path) / _repo_dir_name(model_id)
    refs_main_path = repo_cache_dir / "refs" / "main"
    snapshots_dir = repo_cache_dir / "snapshots"
    snapshot_candidates: list[Path] = []

    if refs_main_path.is_file():
        try:
            revision = refs_main_path.read_text(encoding="utf-8").strip()
        except OSError:
            revision = ""
        if revision:
            snapshot_candidates.append(snapshots_dir / revision)

    if snapshots_dir.is_dir():
        try:
            snapshot_candidates.extend(path for path in snapshots_dir.iterdir() if path.is_dir())
        except OSError:
            pass

    for snapshot_path in snapshot_candidates:
        if (snapshot_path / "config.yaml").is_file():
            return str(snapshot_path), cache_path

    return model_id, cache_path


def _from_pretrained_any(model_id: str, token: Optional[str], cache_dir: Optional[str]):
    args = {"cache_dir": cache_dir} if cache_dir else {}
    from pyannote.audio import Pipeline

    try:
        return Pipeline.from_pretrained(model_id, token=token, **args)
    except TypeError:
        return Pipeline.from_pretrained(model_id, use_auth_token=token, **args)


def load_diarization_model() -> bool:
    global HUGGING_FACE_TOKEN

    with settings_lock:
        model_id = str(current_settings.get("diarization_model_id", "")).strip()
        resolved_cache_path = resolve_model_cache_path(str(current_settings.get("model_cache_path", "")).strip())
    target_identifier = (model_id, resolved_cache_path)

    with model_load_lock:
        if (
            diarization_pipeline.get("pipeline") is not None
            and diarization_pipeline.get("model_identifier") == target_identifier
        ):
            return True

        if diarization_pipeline.get("pipeline") is not None:
            print("[INFO-DIA] Entlade altes Diarisierungs-Modell...", file=sys.stderr)
            diarization_pipeline["pipeline"] = None
            diarization_pipeline["model_identifier"] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[INFO-DIA] Lade Sprecher-Diarisierungs-Modell ({model_id})...", file=sys.stderr)

        HUGGING_FACE_TOKEN = _resolve_huggingface_token()
        if HUGGING_FACE_TOKEN:
            print("[INFO-DIA] Hugging Face Token aus Settings/.env geladen.", file=sys.stderr)
        else:
            print(
                "[WARNUNG-DIA] Kein Hugging Face Token in Settings/.env gefunden. Versuche Cache-/Hub-Load trotzdem.",
                file=sys.stderr,
            )

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained_source, cache_dir = _resolve_diarization_pretrained_source(model_id, resolved_cache_path)
            if pretrained_source != model_id:
                print(f"[INFO-DIA] Verwende lokales Cache-Modell fuer Diarisierung: {pretrained_source}", file=sys.stderr)
            elif cache_dir:
                print(f"[INFO-DIA] Verwende Hugging-Face-Cache fuer Diarisierung: {cache_dir}", file=sys.stderr)

            pipeline = _from_pretrained_any(pretrained_source, HUGGING_FACE_TOKEN, cache_dir)
            pipeline.to(device)

            diarization_pipeline["pipeline"] = pipeline
            diarization_pipeline["model_identifier"] = target_identifier
            print(f"[INFO-DIA] Diarisierungs-Modell erfolgreich auf '{device}' geladen.", file=sys.stderr)
            return True
        except Exception as exc:
            error_message = str(exc)
            if "No module named 'omegaconf'" in error_message:
                error_message = (
                    "Fehlende Python-Abhaengigkeit 'omegaconf'. "
                    "Bitte die Server-venv mit requirements.txt aktualisieren."
                )
            print(f"[FEHLER-DIA] Kritisches Problem beim Laden des Diarisierungs-Modells: {error_message}", file=sys.stderr)
            diarization_pipeline["pipeline"] = None
            diarization_pipeline["model_identifier"] = None
            return False


def _format_result(result_obj: Any) -> Dict[str, list]:
    speaker_turns: Dict[str, list] = {}

    if hasattr(result_obj, "speaker_diarization"):
        iterator = result_obj.speaker_diarization
        for turn, speaker in iterator:
            speaker_turns.setdefault(str(speaker), []).append(
                {"start": round(float(turn.start), 3), "end": round(float(turn.end), 3)}
            )
        return speaker_turns

    if hasattr(result_obj, "itertracks"):
        for turn, _, speaker in result_obj.itertracks(yield_label=True):
            speaker_turns.setdefault(str(speaker), []).append(
                {"start": round(float(turn.start), 3), "end": round(float(turn.end), 3)}
            )
        return speaker_turns

    return speaker_turns


def diarize_audio(
    audio_data_np: np.ndarray,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Dict[str, list]:
    if not load_diarization_model():
        raise RuntimeError(
            "Das Diarisierungs-Modell konnte nicht geladen werden. Pruefen Sie die Server-Logs und den Hugging Face Token."
        )

    pipeline = diarization_pipeline.get("pipeline")
    if pipeline is None:
        raise RuntimeError("Diarisierungs-Pipeline nicht initialisiert.")

    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        class LiveStatusProgressHook(ProgressHook):
            def __call__(
                self,
                step_name: str,
                step_artifact: Any,
                file: Optional[Dict] = None,
                total: Optional[int] = None,
                completed: Optional[int] = None,
            ):
                super().__call__(step_name, step_artifact, file=file, total=total, completed=completed)
                with task_status_lock:
                    safe_total = total if total is not None else 1
                    safe_comp = completed if completed is not None else 1
                    progress_percent = (safe_comp / safe_total * 100) if safe_total > 0 else 0
                    current_task_status["task_name"] = "Diarization"
                    current_task_status["progress"] = round(progress_percent, 2)
                    current_task_status["details"] = f"Step: {step_name} ({safe_comp}/{safe_total})"

        waveform = torch.from_numpy(audio_data_np).unsqueeze(0)
        audio_dict = {"waveform": waveform, "sample_rate": 16000}

        pipeline_kwargs: Dict[str, int] = {}
        if num_speakers is not None:
            pipeline_kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            pipeline_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            pipeline_kwargs["max_speakers"] = max_speakers

        with LiveStatusProgressHook() as hook:
            diarization_result = pipeline(audio_dict, hook=hook, **pipeline_kwargs)

        return _format_result(diarization_result)
    except Exception as exc:
        print(f"[FEHLER-DIA] Bei der Diarisierung ist ein Fehler aufgetreten: {exc}", file=sys.stderr)
        raise RuntimeError(f"Fehler bei der Diarisierung: {exc}") from exc
