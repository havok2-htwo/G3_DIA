# API Documentation

## Public Routes

The public diarization route is open until at least one client API key is configured. Once keys exist, callers must provide:

```http
X-API-Key: g3_dia_...
```

### `POST /diarize/`

Legacy endpoint. Its response contract remains unchanged.

Multipart form-data:

- `file`: required audio or video upload
- `num_speakers`: optional exact speaker count
- `min_speakers`: optional lower speaker bound
- `max_speakers`: optional upper speaker bound

Example response:

```json
{
  "diarization": {
    "SPEAKER_00": [
      { "start": 0.12, "end": 2.46 }
    ],
    "SPEAKER_01": [
      { "start": 2.61, "end": 5.14 }
    ]
  },
  "total_duration_ms": 1842,
  "speakers_found": 2,
  "segments_found": 2
}
```

### `POST /v2/diarize`

Multipart form-data:

- `file`: required audio or video upload
- `num_speakers`: optional exact speaker count (`1..64`)
- `min_speakers`: optional lower speaker bound (`1..64`)
- `max_speakers`: optional upper speaker bound (`1..64`)

`num_speakers` cannot be combined with the bounds. `min_speakers` must not be
greater than `max_speakers`. Timecodes are integer milliseconds. Standard
diarization retains overlapping speakers; exclusive diarization contains at
most one speaker at a time for downstream transcription.

```json
{
  "schema_version": "2.0",
  "request_id": "3fdba5b82a",
  "status": "completed",
  "model": {
    "id": "pyannote/speaker-diarization-community-1"
  },
  "input": {
    "duration_ms": 5140,
    "num_speakers": 2,
    "min_speakers": null,
    "max_speakers": null
  },
  "counts": {
    "speakers": 2,
    "diarization_segments": 3,
    "exclusive_segments": 2,
    "overlaps": 1
  },
  "diarization": [
    { "start_ms": 120, "end_ms": 2460, "speaker_id": "SPEAKER_00" },
    { "start_ms": 2200, "end_ms": 5140, "speaker_id": "SPEAKER_01" }
  ],
  "exclusive_diarization": [
    { "start_ms": 120, "end_ms": 2380, "speaker_id": "SPEAKER_00" },
    { "start_ms": 2380, "end_ms": 5140, "speaker_id": "SPEAKER_01" }
  ],
  "overlaps": [
    {
      "start_ms": 2200,
      "end_ms": 2460,
      "speaker_ids": ["SPEAKER_00", "SPEAKER_01"]
    }
  ],
  "total_duration_ms": 1842
}
```

Native pyannote speaker centroids are deliberately not exposed by either
endpoint.

### `GET /v2/capabilities`

Returns the configured model ID, model load status, runtime device and support
flags for exclusive diarization and overlap regions. It uses the same
`X-API-Key` policy as the upload endpoints.

```json
{
  "api_version": "2.0",
  "exclusive_diarization": true,
  "overlap_regions": true,
  "native_speaker_embeddings": false,
  "model": {
    "id": "pyannote/speaker-diarization-community-1",
    "status": "loaded",
    "device": "cuda"
  }
}
```

### cURL

```bash
curl -X POST "http://127.0.0.1:7864/v2/diarize" \
  -H "X-API-Key: g3_dia_..." \
  -F "file=@Sprache.m4a" \
  -F "num_speakers=2"
```

## Protected Admin Routes

Admin routes use username/password login and an httpOnly session cookie. The default first-run login is `admin` / `admin`, and the password must be changed before protected admin actions are available.

### `POST /api/admin/auth/login`

Creates a browser session.

```json
{
  "username": "admin",
  "password": "admin"
}
```

### `POST /api/admin/auth/change-password`

Changes the current admin password and refreshes the session.

```json
{
  "current_password": "admin",
  "new_password": "new-password"
}
```

### `GET /api/admin/auth/whoami`

Returns the current session user and password-change state.

### `POST /api/admin/auth/logout`

Clears the current browser session.

### `GET /api/admin/api-keys`

Returns metadata for configured client API keys.

### `POST /api/admin/api-keys`

Creates a client API key and returns the plaintext token once.

```json
{
  "alias": "local-client"
}
```

### `DELETE /api/admin/api-keys/{key_id}`

Deletes a client API key.

### `GET /api/admin/settings`

Returns the persisted DIA settings and the currently loaded model identifier.

### `PUT /api/admin/settings`

Expected JSON body:

```json
{
  "diarization_model_id": "pyannote/speaker-diarization-community-1",
  "model_cache_path": ".\\models",
  "huggingface_token": ""
}
```

### `GET /api/admin/stats`

Returns summary metrics and recent diarization history.

Summary fields:

- `total_requests`
- `avg_total_duration_ms`
- `avg_speakers_found`
- `avg_segments_found`

### `GET /api/admin/task`

Returns live worker state and the current task-progress snapshot.

Fields include:

- `worker_running`
- `pending_requests`
- `active_request_id`
- `active_started_at`
- `active_audio_seconds`
- `last_completed_at`
- `last_duration_ms`
- `last_error`
- `total_requests_processed`
- `current_task`

### `POST /api/admin/benchmark`

Multipart form-data:

- `file`: required audio or video upload
- `repeat_count`: integer from `1` to `32`

Example response fields:

- `workflow`
- `model_id`
- `repeat_count`
- `audio_seconds`
- `total_audio_seconds`
- `total_wall_time_ms`
- `avg_wall_time_per_run_ms`
- `avg_single_run_ms`
- `rtf`
- `results_match`
- `speakers_found`
- `segments_found`
- `sample_result`
- `peak_vram_reserved_mb`
- `peak_vram_allocated_mb`

## Shared GPU lease

When DIA and Whisper share one CUDA device, set `GENESIS_GPU_LEASE_PATH` to
the same path in both processes, backed by a shared volume when containers are
used. DIA then holds a cross-process file lock during model load and inference,
in addition to its local request lock. If the variable is unset or CUDA is not
available, this feature is a no-op.

```env
GENESIS_GPU_LEASE_PATH=/shared-locks/genesis-gpu.lock
```
