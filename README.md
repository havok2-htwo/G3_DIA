# G3_DIA

## Purpose

`G3_DIA` is the G3-grade local speaker diarization server.
It combines the diarization core from the older `genesis2_dia_server_project` with the protected FastAPI + React admin experience used by `G3_WHISPER`.

The project exposes:

- a public diarization API on `POST /diarize/`
- a protected admin dashboard on `GET /admin`
- OpenAPI docs on `GET /docs`
- a landing page on `GET /`

## Core Features

- speaker diarization via `pyannote/speaker-diarization-community-1`
- local FastAPI backend with React/Vite frontend
- `X-Admin-Key` protection for all admin routes
- persistent hashed admin key plus temporary startup admin key
- admin key rotation from the browser
- persisted Hugging Face token and cache-path settings
- live task progress and worker status for the active diarization job
- request history with speaker/segment summaries
- benchmark workflow for repeated diarization runs
- audio loading with `soundfile` and `ffmpeg` fallback

## Architecture

Active backend files live in [`backend`](x:/dev/G3_DIA/backend):

- `backend/genesis_dia_server.py`
  - FastAPI app, startup, frontend delivery
- `backend/genesis_dia_server_api.py`
  - public diarization endpoint on `POST /diarize/`
- `backend/genesis_dia_server_admin.py`
  - protected admin endpoints for keys, settings, stats, task state, benchmark
- `backend/genesis_dia_server_engine.py`
  - pyannote model loading and diarization runtime
- `backend/genesis_dia_server_audio.py`
  - audio/video decode, resampling, mono normalization
- `backend/genesis_dia_server_auth.py`
  - persistent admin key storage, startup key support, header verification
- `backend/genesis_dia_server_storage.py`
  - settings persistence and JSONL logging
- `backend/genesis_dia_server_globals.py`
  - shared runtime state, locks, log paths, defaults

Frontend files live in [`frontend`](x:/dev/G3_DIA/frontend) and follow the same G3 visual language as `G3_WHISPER`, but with DIA-specific sections instead of ASR model routing and batch queue management.

## Public API

### `POST /diarize/`

Multipart form fields:

- `file`: audio or video file
- `num_speakers`: optional exact speaker count
- `min_speakers`: optional lower bound
- `max_speakers`: optional upper bound

Response fields:

- `diarization`: speaker-to-segment mapping
- `total_duration_ms`
- `speakers_found`
- `segments_found`

The API is intentionally public. Admin credentials are only required for `/api/admin/...`.

## Admin Dashboard

Protected routes:

- `GET /api/admin/keys`
- `POST /api/admin/keys`
- `GET /api/admin/settings`
- `PUT /api/admin/settings`
- `GET /api/admin/stats`
- `GET /api/admin/task`
- `POST /api/admin/benchmark`

The dashboard offers:

- admin key metadata and rotation
- current task progress from the pyannote progress hook
- worker status such as pending requests and last error
- runtime settings for cache path and Hugging Face token
- benchmark results including runtime, RTF, speaker count, and sample output
- latest request history

## Settings and Tokens

Persisted settings are stored in `logs/genesis_dia_settings.json`.
The most important runtime values are:

- `diarization_model_id`
- `model_cache_path`
- `huggingface_token`

Important token rule:

- A token entered only ad hoc for a one-time download is not sufficient for later runtime loading.
- The token must be saved in the admin settings or provided through `.env`.

Recognized environment variables:

- `HUGGINGFACE_TOKEN`
- `HF_TOKEN`
- `HUGGING_FACE_HUB_TOKEN`
- `GENESIS_ADMIN_KEY`
- `GENESIS_STARTUP_ADMIN_KEY`
- `GENESIS_STARTUP_ADMIN_KEY_TTL_SECONDS`
- `GENESIS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS`

## Logs and Persistent Data

Runtime files are stored under [`logs`](x:/dev/G3_DIA/logs):

- `logs/genesis_dia_settings.json`
- `logs/genesis_dia_secrets.json`
- `logs/diarization_log.jsonl`

## Startup

Windows:

```bat
start.bat
```

Linux / Unix:

```bash
bash ./install.sh
bash ./start.sh
```

The launch/install flow handles:

- local `venv` creation
- `pip`, `setuptools`, `wheel` updates
- PyTorch installation
- dependency installation from `requirements.txt`
- frontend dependency installation and build
- startup admin key generation
- server launch via `python -m backend.genesis_dia_server`

## Notes

- `omegaconf` is intentionally part of `requirements.txt` because pyannote can fail at runtime without it even when `pyannote.audio` is already installed.
- `ffmpeg` is optional but strongly recommended for broader media-format support.
- The current G3-DIA runtime is intentionally built around a single diarization model, so there is no model picker in the admin UI.
