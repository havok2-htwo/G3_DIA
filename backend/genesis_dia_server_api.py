from __future__ import annotations

import asyncio
import datetime
import sys
import time
import uuid
from typing import Any, Callable, Dict, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from .genesis_dia_server_audio import get_audio_duration_seconds, load_audio_file
from .genesis_dia_server_auth import authorize_api_key, get_auth_store
from .genesis_dia_server_engine import diarize_audio, diarize_audio_v2
from .genesis_dia_server_gpu_lease import run_blocking_gpu_phase
from .genesis_dia_server_globals import (
    current_settings,
    current_task_status,
    diarization_history,
    diarization_pipeline,
    history_lock,
    settings_lock,
    task_runtime_lock,
    task_runtime_state,
    task_status_lock,
)
from .genesis_dia_server_storage import log_diarization


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _speaker_count(result: Dict[str, list]) -> int:
    return len(result.keys())


def _segment_count(result: Dict[str, list]) -> int:
    return sum(len(segments) for segments in result.values())


def _v2_speaker_count(result: Dict[str, list]) -> int:
    return len({segment["speaker_id"] for segment in result.get("diarization", [])})


def _register_waiting_request() -> None:
    with task_runtime_lock:
        task_runtime_state["pending_requests"] = int(task_runtime_state.get("pending_requests", 0)) + 1


def _cancel_waiting_request() -> None:
    with task_runtime_lock:
        task_runtime_state["pending_requests"] = max(0, int(task_runtime_state.get("pending_requests", 0)) - 1)


def _activate_request(request_id: str, audio_seconds: float) -> None:
    with task_runtime_lock:
        task_runtime_state["pending_requests"] = max(0, int(task_runtime_state.get("pending_requests", 0)) - 1)
        task_runtime_state["worker_running"] = True
        task_runtime_state["active_request_id"] = request_id
        task_runtime_state["active_started_at"] = _now_iso()
        task_runtime_state["active_audio_seconds"] = round(audio_seconds, 3)
        task_runtime_state["last_error"] = None


def _complete_request(total_duration_ms: int, error_message: str | None = None) -> None:
    with task_runtime_lock:
        task_runtime_state["worker_running"] = False
        task_runtime_state["active_request_id"] = None
        task_runtime_state["active_started_at"] = None
        task_runtime_state["active_audio_seconds"] = 0.0
        task_runtime_state["last_completed_at"] = _now_iso()
        task_runtime_state["last_duration_ms"] = total_duration_ms
        task_runtime_state["last_error"] = error_message
        if error_message is None:
            task_runtime_state["total_requests_processed"] = int(task_runtime_state.get("total_requests_processed", 0)) + 1


def _validate_v2_speaker_counts(
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> None:
    values = {
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }
    for name, value in values.items():
        if value is not None and not 1 <= value <= 64:
            raise HTTPException(status_code=422, detail=f"{name} muss zwischen 1 und 64 liegen.")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise HTTPException(status_code=422, detail="min_speakers darf nicht groesser als max_speakers sein.")
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise HTTPException(
            status_code=422,
            detail="num_speakers darf nicht mit min_speakers oder max_speakers kombiniert werden.",
        )


async def _decode_upload(file: UploadFile) -> tuple[Any, str, float]:
    filename = file.filename or "upload-audio"
    try:
        audio_data = await asyncio.to_thread(load_audio_file, file.file, filename)
        audio_seconds = get_audio_duration_seconds(audio_data)
        return audio_data, filename, audio_seconds
    except (HTTPException, asyncio.CancelledError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Konnte Audiodatei nicht verarbeiten: {exc}") from exc


async def _run_request(
    request: Request,
    file: UploadFile,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    *,
    endpoint_path: str,
    engine: Callable[..., Dict[str, list]],
    result_counts: Callable[[Dict[str, list]], tuple[int, int]],
) -> tuple[Dict[str, list], Dict[str, Any]]:
    api_key_id = authorize_api_key(request)
    request_start_time = time.monotonic()
    request_id = uuid.uuid4().hex[:10]
    source_ip = request.client.host if request.client else "unknown"
    audio_data, filename, audio_seconds = await _decode_upload(file)
    activated = False
    waiting_registered = False

    local_gpu_lock = request.app.state.local_gpu_lock
    _register_waiting_request()
    waiting_registered = True

    try:
        async with local_gpu_lock:
            _activate_request(request_id, audio_seconds)
            activated = True
            with task_status_lock:
                current_task_status["task_name"] = "Diarization"
                current_task_status["progress"] = 0.0
                current_task_status["details"] = "Preparing diarization."
            try:
                print("[API-DIA] Lokaler GPU-Lock fuer Diarisierung erworben.", file=sys.stderr)
                result = await run_blocking_gpu_phase(
                    engine,
                    audio_data,
                    num_speakers,
                    min_speakers,
                    max_speakers,
                )
                print("[API-DIA] Lokaler GPU-Lock fuer Diarisierung freigegeben.", file=sys.stderr)
            finally:
                # Reset progress before releasing the local lock. Otherwise a
                # queued request could acquire it, publish its progress and be
                # overwritten by this request's late cleanup.
                with task_status_lock:
                    current_task_status["task_name"] = "Idle"
                    current_task_status["progress"] = 0.0
                    current_task_status["details"] = "Server is ready."
    except (HTTPException, asyncio.CancelledError) as exc:
        total_duration_ms = round((time.monotonic() - request_start_time) * 1000)
        if activated:
            _complete_request(total_duration_ms, str(exc))
        elif waiting_registered:
            _cancel_waiting_request()
        raise
    except Exception as exc:
        total_duration_ms = round((time.monotonic() - request_start_time) * 1000)
        if activated:
            _complete_request(total_duration_ms, str(exc))
        elif waiting_registered:
            _cancel_waiting_request()
        print(f"[API-DIA-FEHLER] bei {endpoint_path}: {exc}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    total_duration_ms = round((time.monotonic() - request_start_time) * 1000)
    _complete_request(total_duration_ms)
    speakers_found, segments_found = result_counts(result)

    with settings_lock:
        model_id = str(current_settings.get("diarization_model_id", ""))

    log_entry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_ip": source_ip,
        "engine": "diarization",
        "api_version": "2.0" if endpoint_path.startswith("/v2/") else "legacy",
        "model_id": model_id,
        "audio_seconds": round(audio_seconds, 3),
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "speakers_found": speakers_found,
        "segments_found": segments_found,
        "total_duration_ms": total_duration_ms,
        "summary": f"Diarization successful. {speakers_found} speakers, {segments_found} segments.",
    }

    with history_lock:
        diarization_history.appendleft(log_entry)
    log_diarization(log_entry)

    if api_key_id:
        get_auth_store().record_api_key_usage(api_key_id, audio_seconds)

    if await request.is_disconnected():
        print("[API-DIA-WARNUNG] Client hat die Verbindung getrennt.", file=sys.stderr)

    metadata = {
        "request_id": request_id,
        "model_id": model_id,
        "duration_ms": round(audio_seconds * 1000.0),
        "total_duration_ms": total_duration_ms,
        "speakers_found": speakers_found,
        "segments_found": segments_found,
        "filename": filename,
    }
    return result, metadata


def _legacy_counts(result: Dict[str, list]) -> tuple[int, int]:
    return _speaker_count(result), _segment_count(result)


def _v2_counts(result: Dict[str, list]) -> tuple[int, int]:
    return _v2_speaker_count(result), len(result.get("diarization", []))


def _build_v2_response(
    result: Dict[str, list],
    metadata: Dict[str, Any],
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Dict[str, Any]:
    return {
        "schema_version": "2.0",
        "request_id": metadata["request_id"],
        "status": "completed",
        "model": {"id": metadata["model_id"]},
        "input": {
            "duration_ms": metadata["duration_ms"],
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        },
        "counts": {
            "speakers": metadata["speakers_found"],
            "diarization_segments": len(result["diarization"]),
            "exclusive_segments": len(result["exclusive_diarization"]),
            "overlaps": len(result["overlaps"]),
        },
        "diarization": result["diarization"],
        "exclusive_diarization": result["exclusive_diarization"],
        "overlaps": result["overlaps"],
        "total_duration_ms": metadata["total_duration_ms"],
    }


def create_api(app: FastAPI) -> FastAPI:
    @app.post("/diarize/")
    async def diarize_endpoint(
        request: Request,
        file: UploadFile = File(..., description="The audio or video file to diarize."),
        num_speakers: Optional[int] = Form(None, description="Exact number of speakers."),
        min_speakers: Optional[int] = Form(None, description="Minimum number of speakers."),
        max_speakers: Optional[int] = Form(None, description="Maximum number of speakers."),
    ):
        result, metadata = await _run_request(
            request,
            file,
            num_speakers,
            min_speakers,
            max_speakers,
            endpoint_path="/diarize/",
            engine=diarize_audio,
            result_counts=_legacy_counts,
        )
        # Deliberately preserve the legacy response shape.
        return {
            "diarization": result,
            "total_duration_ms": metadata["total_duration_ms"],
            "speakers_found": metadata["speakers_found"],
            "segments_found": metadata["segments_found"],
        }

    @app.post("/v2/diarize")
    async def diarize_v2_endpoint(
        request: Request,
        file: UploadFile = File(..., description="The audio or video file to diarize."),
        num_speakers: Optional[int] = Form(None, description="Exact number of speakers (1..64)."),
        min_speakers: Optional[int] = Form(None, description="Minimum number of speakers (1..64)."),
        max_speakers: Optional[int] = Form(None, description="Maximum number of speakers (1..64)."),
    ):
        _validate_v2_speaker_counts(num_speakers, min_speakers, max_speakers)
        result, metadata = await _run_request(
            request,
            file,
            num_speakers,
            min_speakers,
            max_speakers,
            endpoint_path="/v2/diarize",
            engine=diarize_audio_v2,
            result_counts=_v2_counts,
        )

        return _build_v2_response(result, metadata, num_speakers, min_speakers, max_speakers)

    @app.get("/v2/capabilities")
    async def capabilities_v2_endpoint(request: Request):
        authorize_api_key(request)
        with settings_lock:
            model_id = str(current_settings.get("diarization_model_id", ""))
        model_loaded = diarization_pipeline.get("pipeline") is not None
        return {
            "api_version": "2.0",
            "exclusive_diarization": True,
            "overlap_regions": True,
            "native_speaker_embeddings": False,
            "model": {
                "id": model_id,
                "status": "loaded" if model_loaded else "not_loaded",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
        }

    return app
