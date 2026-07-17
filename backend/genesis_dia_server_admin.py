from __future__ import annotations

import asyncio
import json
import time
import uuid
from statistics import mean
from typing import Any, Dict, List

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from .genesis_dia_server_audio import get_audio_duration_seconds, load_audio_bytes
from .genesis_dia_server_auth import (
    SESSION_COOKIE_NAME,
    clear_session_cookie,
    get_auth_store,
    require_admin,
    require_session,
    set_session_cookie,
)
from .genesis_dia_server_engine import load_diarization_model, diarize_audio
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
from .genesis_dia_server_storage import normalize_settings, save_settings


class AdminSettingsPayload(BaseModel):
    diarization_model_id: str
    model_cache_path: str
    huggingface_token: str


class LoginPayload(BaseModel):
    username: str
    password: str


class ChangePasswordPayload(BaseModel):
    current_password: str
    new_password: str


class CreateApiKeyPayload(BaseModel):
    alias: str = ""


def _serialize_settings() -> Dict[str, Any]:
    with settings_lock:
        settings_copy = current_settings.copy()
    return normalize_settings(settings_copy)


def _reset_peak_vram_tracking(cuda_index: int | None) -> None:
    if cuda_index is None:
        return
    torch.cuda.synchronize(cuda_index)
    torch.cuda.reset_peak_memory_stats(cuda_index)


def _read_peak_vram_metrics(cuda_index: int | None) -> Dict[str, float | None]:
    if cuda_index is None:
        return {
            "peak_vram_reserved_mb": None,
            "peak_vram_allocated_mb": None,
        }

    torch.cuda.synchronize(cuda_index)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(cuda_index)
    peak_allocated_bytes = torch.cuda.max_memory_allocated(cuda_index)
    return {
        "peak_vram_reserved_mb": round(peak_reserved_bytes / (1024 * 1024), 2),
        "peak_vram_allocated_mb": round(peak_allocated_bytes / (1024 * 1024), 2),
    }


def _normalize_result_for_compare(result: Dict[str, list]) -> str:
    return json.dumps(result, sort_keys=True, ensure_ascii=True)


async def _run_admin_benchmark(request: Request, audio_data, repeat_count: int) -> Dict[str, Any]:
    if not await asyncio.to_thread(load_diarization_model):
        raise HTTPException(status_code=500, detail="Diarisierungs-Modell konnte fuer den Benchmark nicht geladen werden.")

    cuda_index = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
    _reset_peak_vram_tracking(cuda_index)

    audio_seconds = round(get_audio_duration_seconds(audio_data), 3)
    total_audio_seconds = round(audio_seconds * repeat_count, 3)
    run_durations: List[int] = []
    results: List[Dict[str, list]] = []
    local_gpu_lock = request.app.state.local_gpu_lock
    benchmark_request_id = f"benchmark-{uuid.uuid4().hex[:10]}"
    batch_started_at = time.perf_counter()

    with task_status_lock:
        current_task_status["task_name"] = "Benchmark"
        current_task_status["progress"] = 0.0
        current_task_status["details"] = "Preparing repeated diarization runs."

    async with local_gpu_lock:
        with task_runtime_lock:
            task_runtime_state["worker_running"] = True
            task_runtime_state["active_request_id"] = benchmark_request_id
            task_runtime_state["active_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            task_runtime_state["active_audio_seconds"] = audio_seconds
            task_runtime_state["last_error"] = None

        try:
            for repeat_index in range(repeat_count):
                with task_status_lock:
                    current_task_status["task_name"] = "Benchmark"
                    current_task_status["progress"] = round((repeat_index / repeat_count) * 100, 2)
                    current_task_status["details"] = f"Run {repeat_index + 1} / {repeat_count}"

                run_started_at = time.perf_counter()
                result = await asyncio.to_thread(diarize_audio, audio_data)
                run_durations.append(round((time.perf_counter() - run_started_at) * 1000))
                results.append(result)
        except Exception as exc:
            with task_runtime_lock:
                task_runtime_state["worker_running"] = False
                task_runtime_state["active_request_id"] = None
                task_runtime_state["active_started_at"] = None
                task_runtime_state["active_audio_seconds"] = 0.0
                task_runtime_state["last_error"] = str(exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            with task_status_lock:
                current_task_status["task_name"] = "Idle"
                current_task_status["progress"] = 0.0
                current_task_status["details"] = "Server is ready."
            with task_runtime_lock:
                task_runtime_state["worker_running"] = False
                task_runtime_state["active_request_id"] = None
                task_runtime_state["active_started_at"] = None
                task_runtime_state["active_audio_seconds"] = 0.0

    total_wall_time_ms = round((time.perf_counter() - batch_started_at) * 1000)
    wall_seconds = total_wall_time_ms / 1000 if total_wall_time_ms > 0 else 0.0
    first_result = results[0] if results else {}
    speakers_found = len(first_result.keys())
    segments_found = sum(len(segments) for segments in first_result.values())
    peak_metrics = _read_peak_vram_metrics(cuda_index)
    normalized_results = {_normalize_result_for_compare(result) for result in results}
    with settings_lock:
        model_id = current_settings.get("diarization_model_id", "")

    return {
        "ok": True,
        "workflow": "serial_diarization",
        "model_id": model_id,
        "repeat_count": repeat_count,
        "audio_seconds": audio_seconds,
        "total_audio_seconds": total_audio_seconds,
        "total_wall_time_ms": total_wall_time_ms,
        "avg_wall_time_per_run_ms": round(total_wall_time_ms / repeat_count, 2),
        "avg_single_run_ms": round(mean(run_durations), 2) if run_durations else None,
        "rtf": round(total_audio_seconds / wall_seconds, 3) if wall_seconds > 0 else None,
        "results_match": len(normalized_results) <= 1,
        "speakers_found": speakers_found,
        "segments_found": segments_found,
        "sample_result": first_result,
        **peak_metrics,
    }


def create_admin_api(app: FastAPI) -> FastAPI:
    # --- Auth: username/password login backed by an httpOnly session cookie ---
    @app.post("/api/admin/auth/login")
    async def admin_login(payload: LoginPayload, request: Request, response: Response):
        store = get_auth_store()
        user = store.verify_user(payload.username, payload.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        token = store.create_session(user["username"])
        store.touch_login(user["username"])
        set_session_cookie(response, request, token)
        return {"username": user["username"], "must_change_password": bool(user.get("must_change_password"))}

    @app.post("/api/admin/auth/logout")
    async def admin_logout(request: Request, response: Response, _: dict = Depends(require_session)):
        get_auth_store().delete_session(request.cookies.get(SESSION_COOKIE_NAME))
        clear_session_cookie(response)
        return {"ok": True}

    @app.get("/api/admin/auth/whoami")
    async def admin_whoami(ctx: dict = Depends(require_session)):
        return {"username": ctx["username"], "must_change_password": ctx["must_change_password"]}

    @app.post("/api/admin/auth/change-password")
    async def admin_change_password(
        payload: ChangePasswordPayload,
        request: Request,
        response: Response,
        ctx: dict = Depends(require_session),
    ):
        store = get_auth_store()
        if not store.verify_user(ctx["username"], payload.current_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect.")
        store.set_password(ctx["username"], payload.new_password)
        token = store.create_session(ctx["username"])
        set_session_cookie(response, request, token)
        return {"ok": True, "must_change_password": False}

    # --- Client API keys (admin-managed; used to authorize the public /diarize/ API) ---
    @app.get("/api/admin/api-keys")
    async def admin_list_api_keys(_: dict = Depends(require_admin)):
        return {"keys": get_auth_store().list_api_keys()}

    @app.post("/api/admin/api-keys")
    async def admin_create_api_key(payload: CreateApiKeyPayload, _: dict = Depends(require_admin)):
        return get_auth_store().create_api_key(payload.alias)

    @app.delete("/api/admin/api-keys/{key_id}")
    async def admin_delete_api_key(key_id: str, _: dict = Depends(require_admin)):
        if not get_auth_store().delete_api_key(key_id):
            raise HTTPException(status_code=404, detail="API key not found.")
        return {"ok": True}

    @app.get("/api/admin/settings")
    async def admin_get_settings(_: dict[str, str] = Depends(require_admin)):
        model_identifier = diarization_pipeline.get("model_identifier")
        return {
            "settings": _serialize_settings(),
            "loaded_model_identifier": list(model_identifier) if model_identifier else None,
        }

    @app.put("/api/admin/settings")
    async def admin_update_settings(payload: AdminSettingsPayload, _: dict[str, str] = Depends(require_admin)):
        payload_data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        normalized = normalize_settings(payload_data)
        with settings_lock:
            previous_settings = current_settings.copy()
            current_settings.update(normalized)
            saved_settings = save_settings(current_settings.copy())
            current_settings.clear()
            current_settings.update(saved_settings)

        model_settings_changed = any(
            previous_settings.get(key) != saved_settings.get(key)
            for key in ("diarization_model_id", "model_cache_path", "huggingface_token")
        )
        model_loaded = None
        if model_settings_changed:
            model_loaded = await asyncio.to_thread(load_diarization_model)

        model_identifier = diarization_pipeline.get("model_identifier")
        return {
            "ok": True,
            "settings": _serialize_settings(),
            "model_reloaded": model_settings_changed,
            "model_loaded": model_loaded,
            "loaded_model_identifier": list(model_identifier) if model_identifier else None,
        }

    @app.get("/api/admin/stats")
    async def admin_stats(_: dict[str, str] = Depends(require_admin)):
        with history_lock:
            history_items = list(diarization_history)

        recent_history = history_items[:25]
        total_duration_values = [entry.get("total_duration_ms", 0) for entry in history_items if entry.get("total_duration_ms") is not None]
        speaker_values = [entry.get("speakers_found", 0) for entry in history_items if entry.get("speakers_found") is not None]
        segment_values = [entry.get("segments_found", 0) for entry in history_items if entry.get("segments_found") is not None]

        return {
            "summary": {
                "total_requests": len(history_items),
                "avg_total_duration_ms": round(mean(total_duration_values), 2) if total_duration_values else None,
                "avg_speakers_found": round(mean(speaker_values), 2) if speaker_values else None,
                "avg_segments_found": round(mean(segment_values), 2) if segment_values else None,
            },
            "history": recent_history,
        }

    @app.get("/api/admin/task")
    async def admin_task(_: dict[str, str] = Depends(require_admin)):
        with task_runtime_lock:
            runtime_snapshot = dict(task_runtime_state)
        with task_status_lock:
            task_snapshot = dict(current_task_status)
        model_identifier = diarization_pipeline.get("model_identifier")
        return {
            **runtime_snapshot,
            "current_task": task_snapshot,
            "loaded_model_identifier": list(model_identifier) if model_identifier else None,
        }

    @app.post("/api/admin/benchmark")
    async def admin_benchmark(
        request: Request,
        file: UploadFile = File(..., description="Audio- oder Video-Datei fuer den Benchmark."),
        repeat_count: int = Form(1),
        _: dict[str, str] = Depends(require_admin),
    ):
        if repeat_count < 1 or repeat_count > 32:
            raise HTTPException(status_code=400, detail="Wiederholungen muessen zwischen 1 und 32 liegen.")

        filename = file.filename or "benchmark-audio"
        try:
            audio_bytes = await file.read()
            audio_data = load_audio_bytes(audio_bytes, filename)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Konnte Audiodatei nicht verarbeiten: {exc}") from exc

        benchmark_result = await _run_admin_benchmark(request, audio_data, repeat_count)
        benchmark_result["file_name"] = filename
        return benchmark_result

    return app
