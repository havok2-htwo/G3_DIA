# API Documentation

## Public Route

The public diarization route is open until at least one client API key is configured. Once keys exist, callers must provide:

```http
X-API-Key: g3_dia_...
```

### `POST /diarize/`

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
