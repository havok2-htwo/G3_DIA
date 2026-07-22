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
from .genesis_dia_server_gpu_lease import acquire_gpu_lease

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
    """Load the pipeline, taking the optional cross-process CUDA lease."""

    with acquire_gpu_lease():
        return _load_diarization_model_unleased()


def _load_diarization_model_unleased() -> bool:
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


def _annotation_tracks(annotation: Any) -> list[tuple[float, float, str]]:
    tracks: list[tuple[float, float, str]] = []
    if annotation is None:
        return tracks

    if hasattr(annotation, "itertracks"):
        iterator = annotation.itertracks(yield_label=True)
        for turn, _, speaker in iterator:
            tracks.append((float(turn.start), float(turn.end), str(speaker)))
    else:
        for turn, speaker in annotation:
            tracks.append((float(turn.start), float(turn.end), str(speaker)))

    tracks.sort(key=lambda item: (item[0], item[1], item[2]))
    return tracks


def _standard_annotation(result_obj: Any) -> Any:
    return getattr(result_obj, "speaker_diarization", result_obj)


def _format_result(result_obj: Any) -> Dict[str, list]:
    """Keep the historical /diarize/ speaker mapping byte-for-byte compatible."""

    speaker_turns: Dict[str, list] = {}
    for start, end, speaker in _annotation_tracks(_standard_annotation(result_obj)):
        speaker_turns.setdefault(speaker, []).append({"start": round(start, 3), "end": round(end, 3)})
    return speaker_turns


def _format_segments_ms(annotation: Any) -> list[Dict[str, Any]]:
    segments: list[Dict[str, Any]] = []
    for start, end, speaker in _annotation_tracks(annotation):
        start_ms = round(start * 1000.0)
        end_ms = round(end * 1000.0)
        if end_ms <= start_ms:
            continue
        segments.append({"start_ms": start_ms, "end_ms": end_ms, "speaker_id": speaker})
    return segments


def _format_overlap_regions(annotation: Any) -> list[Dict[str, Any]]:
    """Return maximal standard-diarization regions with two or more speakers."""

    events: Dict[float, list[tuple[str, int]]] = {}
    for start, end, speaker in _annotation_tracks(annotation):
        if end <= start:
            continue
        events.setdefault(start, []).append((speaker, 1))
        events.setdefault(end, []).append((speaker, -1))

    active_counts: Dict[str, int] = {}
    overlap_regions: list[Dict[str, Any]] = []
    previous_time: float | None = None

    for event_time in sorted(events):
        active_speakers = sorted(speaker for speaker, count in active_counts.items() if count > 0)
        if previous_time is not None and event_time > previous_time and len(active_speakers) >= 2:
            start_ms = round(previous_time * 1000.0)
            end_ms = round(event_time * 1000.0)
            if end_ms > start_ms:
                current = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "speaker_ids": active_speakers,
                }
                if (
                    overlap_regions
                    and overlap_regions[-1]["end_ms"] == start_ms
                    and overlap_regions[-1]["speaker_ids"] == active_speakers
                ):
                    overlap_regions[-1]["end_ms"] = end_ms
                else:
                    overlap_regions.append(current)

        for speaker, delta in events[event_time]:
            next_count = active_counts.get(speaker, 0) + delta
            if next_count > 0:
                active_counts[speaker] = next_count
            else:
                active_counts.pop(speaker, None)
        previous_time = event_time

    return overlap_regions


def format_diarization_v2(result_obj: Any) -> Dict[str, list]:
    """Format pyannote 4 output without exposing its native speaker centroids."""

    standard_annotation = _standard_annotation(result_obj)
    exclusive_annotation = getattr(result_obj, "exclusive_speaker_diarization", None)
    if exclusive_annotation is None:
        raise RuntimeError(
            "Das konfigurierte Diarisierungs-Modell liefert keine Exclusive-Diarization. "
            "Fuer /v2/diarize ist pyannote/speaker-diarization-community-1 erforderlich."
        )

    return {
        "diarization": _format_segments_ms(standard_annotation),
        "exclusive_diarization": _format_segments_ms(exclusive_annotation),
        "overlaps": _format_overlap_regions(standard_annotation),
    }


def _run_diarization_pipeline(
    audio_data_np: np.ndarray,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Any:
    # One lease spans model loading and inference.  Calling the unleased model
    # loader here avoids nested locks while the public load helper still
    # protects admin-triggered standalone model loads.
    with acquire_gpu_lease():
        return _run_diarization_pipeline_unleased(audio_data_np, num_speakers, min_speakers, max_speakers)


def _run_diarization_pipeline_unleased(
    audio_data_np: np.ndarray,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Any:
    if not _load_diarization_model_unleased():
        raise RuntimeError(
            "Das Diarisierungs-Modell konnte nicht geladen werden. Pruefen Sie die Server-Logs und den Hugging Face Token."
        )

    pipeline = diarization_pipeline.get("pipeline")
    if pipeline is None:
        raise RuntimeError("Diarisierungs-Pipeline nicht initialisiert.")

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
        return pipeline(audio_dict, hook=hook, **pipeline_kwargs)


def diarize_audio(
    audio_data_np: np.ndarray,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Dict[str, list]:
    try:
        diarization_result = _run_diarization_pipeline(audio_data_np, num_speakers, min_speakers, max_speakers)
        return _format_result(diarization_result)
    except Exception as exc:
        print(f"[FEHLER-DIA] Bei der Diarisierung ist ein Fehler aufgetreten: {exc}", file=sys.stderr)
        raise RuntimeError(f"Fehler bei der Diarisierung: {exc}") from exc


def diarize_audio_v2(
    audio_data_np: np.ndarray,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Dict[str, list]:
    try:
        diarization_result = _run_diarization_pipeline(audio_data_np, num_speakers, min_speakers, max_speakers)
        return format_diarization_v2(diarization_result)
    except Exception as exc:
        print(f"[FEHLER-DIA] Bei der v2-Diarisierung ist ein Fehler aufgetreten: {exc}", file=sys.stderr)
        raise RuntimeError(f"Fehler bei der Diarisierung: {exc}") from exc
