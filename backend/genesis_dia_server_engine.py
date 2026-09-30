from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from dotenv import load_dotenv

from .genesis_dia_server_devices import DeviceUnavailableError, resolve_device
from .genesis_dia_server_globals import (
    current_settings,
    current_task_status,
    diarization_pipeline,
    model_load_lock,
    model_status,
    model_status_lock,
    resolve_model_cache_path,
    settings_lock,
    task_status_lock,
)
from .genesis_dia_server_gpu_lease import acquire_gpu_lease

HUGGING_FACE_TOKEN = None


def _resolve_huggingface_token_with_source() -> tuple[str | None, str | None]:
    with settings_lock:
        settings_token = str(current_settings.get("huggingface_token", "")).strip()
    if settings_token:
        return settings_token, "settings"

    load_dotenv()
    env_token = str(
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    return (env_token, "env") if env_token else (None, None)


def _resolve_huggingface_token() -> str | None:
    return _resolve_huggingface_token_with_source()[0]


def _set_model_status(state: str, *, error: str | None = None, device: str | None = None, token_source: str | None = None) -> None:
    with model_status_lock:
        model_status.update(
            {
                "state": state,
                "error": error,
                "device": device,
                "token_source": token_source,
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        )


def get_model_status() -> Dict[str, Any]:
    with model_status_lock:
        return dict(model_status)


def get_token_source() -> str | None:
    """Where the effective Hugging Face token comes from right now (settings, env or none)."""

    return _resolve_huggingface_token_with_source()[1]


def _gated_access_hint(model_id: str) -> str:
    return (
        f"Bitte pruefen: (1) Auf https://hf.co/{model_id} mit dem Konto, dem der Token gehoert, die "
        "Nutzungsbedingungen akzeptieren (ein anderes, im Browser eingeloggtes Konto zaehlt nicht). "
        "(2) Ein 'fine-grained' Token braucht die Berechtigung "
        "'Read access to contents of all public gated repos you can access' (oder einen 'Read'-Token verwenden)."
    )


_NETWORK_HINT = (
    "Hugging Face ist vom Server/Container aus nicht erreichbar und das Modell liegt noch nicht vollstaendig im "
    "Cache. DNS, Proxy und Firewall des Containers pruefen: neben huggingface.co muessen auch die Download-Hosts "
    "cdn-lfs.hf.co und *.xethub.hf.co erreichbar sein. Ein Browser auf dem Host nutzt evtl. einen anderen Proxy."
)
_NETWORK_ERROR_NAMES = {
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "Timeout",
    "ProxyError",
    "SSLError",
    "ConnectError",
    "NameResolutionError",
    "NewConnectionError",
    "MaxRetryError",
    "OfflineModeIsEnabled",
    "LocalEntryNotFoundError",
}
_NETWORK_MARKERS = ("name resolution", "max retries", "connection refused", "network is unreachable", "timed out")


def _hf_error_type(name: str) -> type | None:
    try:
        import huggingface_hub.errors as hf_errors
    except ImportError:  # pragma: no cover - very old huggingface_hub
        return None
    return getattr(hf_errors, name, None)


def _is_instance(exc: BaseException, name: str) -> bool:
    error_type = _hf_error_type(name)
    return error_type is not None and isinstance(exc, error_type)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """The exception plus its causes: hub errors often wrap the real HTTP failure."""

    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 10 and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _http_status(exc: BaseException) -> int | None:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _response_header(exc: BaseException, name: str) -> str:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        return str(headers.get(name) or "") if headers is not None else ""
    except Exception:
        return ""


def _server_message(exc: BaseException) -> str:
    return str(getattr(exc, "server_message", None) or _response_header(exc, "X-Error-Message"))


def _describe_load_error(exc: BaseException, model_id: str, token: str | None) -> str:
    """Turn a pipeline load failure into an actionable admin-UI message.

    Classification walks the whole cause chain: huggingface_hub turns e.g. a 403
    for a fine-grained token into a generic LocalEntryNotFoundError ("check your
    connection"), and only the cause still carries the real HTTP answer.
    """

    chain = _exception_chain(exc)
    raw = f"{type(exc).__name__}: {exc}".strip()
    root = chain[-1]
    details = raw if root is exc else f"{raw} | Ursache: {type(root).__name__}: {root}"
    details = details[:1200]
    joined = " ".join(f"{type(item).__name__}: {item}" for item in chain).lower()
    token_note = (
        " Hinweis: Der Token beginnt nicht mit 'hf_'; pyannote verwirft solche Tokens und laedt dann ohne Token."
        if token and not token.startswith("hf_")
        else ""
    )

    if "no module named 'omegaconf'" in joined:
        return "Fehlende Python-Abhaengigkeit 'omegaconf'. Bitte die Server-venv mit requirements.txt aktualisieren."

    if any(_is_instance(item, "GatedRepoError") for item in chain):
        return (
            f"Zugriff auf das zugangsbeschraenkte Modell '{model_id}' verweigert. "
            f"{_gated_access_hint(model_id)}{token_note} Details: {details}"
        )

    for item in chain:
        status = _http_status(item)
        if status == 401:
            if not token:
                return (
                    f"Hugging Face verlangt fuer '{model_id}' einen Token (HTTP 401), es ist aber keiner gesetzt. "
                    f"Token in den Einstellungen speichern oder HUGGINGFACE_TOKEN setzen. Details: {details}"
                )
            return (
                "Hugging Face hat den Token abgelehnt (HTTP 401: ungueltig, abgelaufen, widerrufen oder "
                "unvollstaendig kopiert). Unter https://hf.co/settings/tokens neu erzeugen bzw. vollstaendig "
                f"kopieren.{token_note} Details: {details}"
            )
        if status == 403:
            message = _server_message(item).lower()
            if "fine-grained" in message or "fine grained" in message:
                return (
                    "Der fine-grained Token darf keine zugangsbeschraenkten (gated) Repos lesen. Unter "
                    "https://hf.co/settings/tokens beim Token 'Read access to contents of all public gated repos "
                    f"you can access' aktivieren oder einen klassischen 'Read'-Token verwenden. Details: {details}"
                )
            if message or _response_header(item, "X-Error-Code") or _response_header(item, "X-Request-Id"):
                return (
                    f"Hugging Face hat den Zugriff auf '{model_id}' verweigert (HTTP 403). "
                    f"{_gated_access_hint(model_id)}{token_note} Details: {details}"
                )
            return (
                "HTTP 403 ohne Hugging-Face-Fehlerkennung: vermutlich blockiert ein Proxy oder eine Firewall "
                f"zwischen Container und Hugging Face. {_NETWORK_HINT} Details: {details}"
            )

    if any(_is_instance(item, "RepositoryNotFoundError") for item in chain):
        return (
            f"Das Modell '{model_id}' wurde auf Hugging Face nicht gefunden oder ist fuer diesen Token nicht "
            f"sichtbar. Modell-ID und Token pruefen.{token_note} Details: {details}"
        )
    if any(_is_instance(item, "RevisionNotFoundError") or _is_instance(item, "EntryNotFoundError") for item in chain):
        return (
            f"Eine Datei oder Revision von '{model_id}' fehlt auf Hugging Face (Modell-ID oder pyannote-Version "
            f"passt nicht). Details: {details}"
        )

    for item in chain:
        status = _http_status(item)
        if status == 429:
            return f"Hugging Face drosselt die Anfragen (HTTP 429). Spaeter erneut laden. Details: {details}"
        if status is not None and status >= 500:
            return f"Hugging Face meldet einen Serverfehler (HTTP {status}). Spaeter erneut laden. Details: {details}"

    if any(type(item).__name__ in _NETWORK_ERROR_NAMES for item in chain) or any(
        marker in joined for marker in _NETWORK_MARKERS
    ):
        return f"{_NETWORK_HINT} Details: {details}"

    if any(isinstance(item, FileNotFoundError) for item in chain):
        return (
            "Eine Modelldatei fehlt im lokalen Cache, vermutlich nach einem abgebrochenen Download. Den Ordner "
            f"'{_repo_dir_name(model_id)}' im Model Cache Path loeschen und das Modell neu laden. Details: {details}"
        )

    return f"{details}{token_note}"


def _repo_dir_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


_PIPELINE_SUBFOLDERS = ("segmentation", "embedding", "plda")


def _snapshot_is_complete(snapshot_path: Path) -> bool:
    """A snapshot only counts once every sub-model its config references is on disk.

    config.yaml is downloaded first and the weights afterwards (from other CDN
    hosts), so an interrupted first download would otherwise pin every later load
    to a local snapshot without weights instead of resuming from the Hub.
    """

    config_path = snapshot_path / "config.yaml"
    if not config_path.is_file():
        return False
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError:
        return False
    for subfolder in _PIPELINE_SUBFOLDERS:
        if f"subfolder: {subfolder}" not in config_text:
            continue
        folder = snapshot_path / subfolder
        try:
            if not any(entry.is_file() and entry.stat().st_size > 0 for entry in folder.iterdir()):
                return False
        except OSError:
            return False
    return True


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
        if _snapshot_is_complete(snapshot_path):
            return str(snapshot_path), cache_path

    return model_id, cache_path


def _from_pretrained_any(model_id: str, token: Optional[str], cache_dir: Optional[str]):
    args = {"cache_dir": cache_dir} if cache_dir else {}
    from pyannote.audio import Pipeline

    try:
        return Pipeline.from_pretrained(model_id, token=token, **args)
    except TypeError as exc:
        # Only pyannote 3.x lacks the ``token`` keyword; any other TypeError is a real
        # load failure and must not be masked by a second, misleading one.
        message = str(exc)
        if "unexpected keyword argument" not in message or "'token'" not in message:
            raise
        return Pipeline.from_pretrained(model_id, use_auth_token=token, **args)


def _uses_gpu_lease() -> bool:
    """CPU inference must not hold the cross-container GPU lease (it would stall Whisper)."""

    with settings_lock:
        device_setting = str(current_settings.get("gpu_device", "auto"))
    try:
        return resolve_device(device_setting).type != "cpu"
    except Exception:
        return True  # the loader reports the device problem itself


def load_diarization_model() -> bool:
    """Load the pipeline, taking the optional cross-process CUDA lease."""

    try:
        with acquire_gpu_lease(enabled=_uses_gpu_lease()):
            return _load_diarization_model_unleased()
    except Exception as exc:
        # E.g. the shared lease file is not writable; keep the reason visible in the UI.
        message = f"Modell-Load abgebrochen: {type(exc).__name__}: {exc}"
        print(f"[FEHLER-DIA] {message}", file=sys.stderr)
        _set_model_status("error", error=message, token_source=get_token_source())
        return False


def _unload_pipeline_locked() -> None:
    if diarization_pipeline.get("pipeline") is None:
        return
    print("[INFO-DIA] Entlade altes Diarisierungs-Modell...", file=sys.stderr)
    diarization_pipeline["pipeline"] = None
    diarization_pipeline["model_identifier"] = None
    diarization_pipeline["device"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_diarization_model_unleased() -> bool:
    global HUGGING_FACE_TOKEN

    with settings_lock:
        model_id = str(current_settings.get("diarization_model_id", "")).strip()
        resolved_cache_path = resolve_model_cache_path(str(current_settings.get("model_cache_path", "")).strip())
        device_setting = str(current_settings.get("gpu_device", "auto"))
    token, token_source = _resolve_huggingface_token_with_source()

    try:
        device = resolve_device(device_setting)
    except Exception as exc:
        message = str(exc) if isinstance(exc, DeviceUnavailableError) else (
            f"Geraet '{device_setting}' ist nicht nutzbar: {type(exc).__name__}: {exc}"
        )
        print(f"[FEHLER-DIA] {message}", file=sys.stderr)
        # Do not keep serving from the previous device: the saved choice is what
        # jobs will try to load, so status and capabilities must agree with it.
        with model_load_lock:
            try:
                _unload_pipeline_locked()
            except Exception:
                pass
        _set_model_status("error", error=message, token_source=token_source)
        return False
    target_identifier = (model_id, resolved_cache_path, str(device))

    with model_load_lock:
        if (
            diarization_pipeline.get("pipeline") is not None
            and diarization_pipeline.get("model_identifier") == target_identifier
        ):
            current = get_model_status()
            if current.get("state") != "loaded" or current.get("token_source") != token_source:
                _set_model_status("loaded", device=str(device), token_source=token_source)
            return True

        _set_model_status("loading", device=str(device), token_source=token_source)
        print(f"[INFO-DIA] Lade Sprecher-Diarisierungs-Modell ({model_id}) auf '{device}'...", file=sys.stderr)

        HUGGING_FACE_TOKEN = token
        if HUGGING_FACE_TOKEN:
            print(f"[INFO-DIA] Hugging Face Token aus {'Settings' if token_source == 'settings' else '.env/Umgebung'} geladen.", file=sys.stderr)
            if not HUGGING_FACE_TOKEN.startswith("hf_"):
                # pyannote drops tokens without the hf_ prefix (it takes them for
                # pyannoteAI keys); a failing load then gets a hint in its message.
                print("[WARNUNG-DIA] Hugging Face Token beginnt nicht mit 'hf_'; pyannote ignoriert ihn.", file=sys.stderr)
        else:
            print(
                "[WARNUNG-DIA] Kein Hugging Face Token in Settings/.env gefunden. Versuche Cache-/Hub-Load trotzdem.",
                file=sys.stderr,
            )

        try:
            _unload_pipeline_locked()
            pretrained_source, cache_dir = _resolve_diarization_pretrained_source(model_id, resolved_cache_path)
            if pretrained_source != model_id:
                print(f"[INFO-DIA] Verwende lokales Cache-Modell fuer Diarisierung: {pretrained_source}", file=sys.stderr)
            elif cache_dir:
                print(f"[INFO-DIA] Verwende Hugging-Face-Cache fuer Diarisierung: {cache_dir}", file=sys.stderr)

            pipeline = _from_pretrained_any(pretrained_source, HUGGING_FACE_TOKEN, cache_dir)
            if pipeline is None:
                raise RuntimeError(f"pyannote konnte keine Pipeline-Konfiguration fuer '{model_id}' laden.")
            pipeline.to(device)

            diarization_pipeline["pipeline"] = pipeline
            diarization_pipeline["model_identifier"] = target_identifier
            diarization_pipeline["device"] = device
            _set_model_status("loaded", device=str(device), token_source=token_source)
            print(f"[INFO-DIA] Diarisierungs-Modell erfolgreich auf '{device}' geladen.", file=sys.stderr)
            return True
        except Exception as exc:
            error_message = _describe_load_error(exc, model_id, HUGGING_FACE_TOKEN)
            print(f"[FEHLER-DIA] Kritisches Problem beim Laden des Diarisierungs-Modells: {error_message}", file=sys.stderr)
            diarization_pipeline["pipeline"] = None
            diarization_pipeline["model_identifier"] = None
            diarization_pipeline["device"] = None
            _set_model_status("error", error=error_message, device=str(device), token_source=token_source)
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
    with acquire_gpu_lease(enabled=_uses_gpu_lease()):
        return _run_diarization_pipeline_unleased(audio_data_np, num_speakers, min_speakers, max_speakers)


def _run_diarization_pipeline_unleased(
    audio_data_np: np.ndarray,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Any:
    if not _load_diarization_model_unleased():
        reason = get_model_status().get("error") or "Pruefen Sie die Server-Logs und den Hugging Face Token."
        raise RuntimeError(f"Das Diarisierungs-Modell konnte nicht geladen werden: {reason}")

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
