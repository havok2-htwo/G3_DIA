from __future__ import annotations

import asyncio
import datetime
import sys
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from .genesis_dia_server_audio import get_audio_duration_seconds, load_audio_bytes
from .genesis_dia_server_engine import diarize_audio
from .genesis_dia_server_globals import (
    current_settings,
    current_task_status,
    diarization_history,
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


def _summary_text(result: Dict[str, list]) -> str:
    return f"Diarization successful. {_speaker_count(result)} speakers, {_segment_count(result)} segments."


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


def create_api(app: FastAPI) -> FastAPI:
    @app.post("/diarize/")
    async def diarize_endpoint(
        request: Request,
        file: UploadFile = File(..., description="The audio or video file to diarize."),
        num_speakers: Optional[int] = Form(None, description="Exact number of speakers."),
        min_speakers: Optional[int] = Form(None, description="Minimum number of speakers."),
        max_speakers: Optional[int] = Form(None, description="Maximum number of speakers."),
    ):
        request_start_time = time.monotonic()
        request_id = uuid.uuid4().hex[:10]
        source_ip = request.client.host if request.client else "unknown"
        filename = file.filename or "upload-audio"
        diarization_result: Dict[str, list] = {}
        activated = False
        waiting_registered = False

        try:
            audio_bytes = await file.read()
            audio_data = load_audio_bytes(audio_bytes, filename)
            audio_seconds = round(get_audio_duration_seconds(audio_data), 3)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Konnte Audiodatei nicht verarbeiten: {exc}") from exc

        with task_status_lock:
            current_task_status["task_name"] = "Queued"
            current_task_status["progress"] = 0.0
            current_task_status["details"] = "Waiting for the local diarization worker."

        local_gpu_lock = request.app.state.local_gpu_lock
        _register_waiting_request()
        waiting_registered = True

        try:
            async with local_gpu_lock:
                _activate_request(request_id, audio_seconds)
                activated = True
                print("[API-DIA] Lokaler GPU-Lock fuer Diarisierung erworben.", file=sys.stderr)
                diarization_result = await asyncio.to_thread(
                    diarize_audio,
                    audio_data,
                    num_speakers,
                    min_speakers,
                    max_speakers,
                )
                print("[API-DIA] Lokaler GPU-Lock fuer Diarisierung freigegeben.", file=sys.stderr)
        except HTTPException:
            raise
        except Exception as exc:
            total_duration_ms = round((time.monotonic() - request_start_time) * 1000)
            if activated:
                _complete_request(total_duration_ms, str(exc))
            elif waiting_registered:
                _cancel_waiting_request()
            print(f"[API-DIA-FEHLER] bei /diarize/: {exc}", file=sys.stderr)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            with task_status_lock:
                current_task_status["task_name"] = "Idle"
                current_task_status["progress"] = 0.0
                current_task_status["details"] = "Server is ready."

        total_duration_ms = round((time.monotonic() - request_start_time) * 1000)
        _complete_request(total_duration_ms)

        speakers_found = _speaker_count(diarization_result)
        segments_found = _segment_count(diarization_result)
        with settings_lock:
            model_id = current_settings.get("diarization_model_id", "")

        log_entry: Dict[str, Any] = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": source_ip,
            "engine": "diarization",
            "model_id": model_id,
            "audio_seconds": audio_seconds,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "speakers_found": speakers_found,
            "segments_found": segments_found,
            "total_duration_ms": total_duration_ms,
            "summary": _summary_text(diarization_result),
        }

        with history_lock:
            diarization_history.appendleft(log_entry)
        log_diarization(log_entry)

        if await request.is_disconnected():
            print("[API-DIA-WARNUNG] Client hat die Verbindung getrennt.", file=sys.stderr)
            return

        return {
            "diarization": diarization_result,
            "total_duration_ms": total_duration_ms,
            "speakers_found": speakers_found,
            "segments_found": segments_found,
        }

    return app
