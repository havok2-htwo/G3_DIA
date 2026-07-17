#!/bin/sh
set -e

# The DIA server's built-in default model cache path is the Windows literal ".\\models",
# which is meaningless on Linux. On first boot (before any settings file exists) seed one
# that pins the cache to the mounted /app/models volume. Existing settings (written by the
# admin dashboard) are left untouched.

SETTINGS_DIR="/app/logs"
SETTINGS_FILE="${SETTINGS_DIR}/genesis_dia_settings.json"

mkdir -p "${SETTINGS_DIR}" /app/models

if [ ! -f "${SETTINGS_FILE}" ]; then
  echo "[entrypoint] Seeding ${SETTINGS_FILE} (cache -> /app/models)"
  cat > "${SETTINGS_FILE}" <<'JSON'
{
    "diarization_model_id": "pyannote/speaker-diarization-community-1",
    "model_cache_path": "/app/models",
    "huggingface_token": ""
}
JSON
fi

# Admin access is username/password (default admin/admin, forced change on first login),
# so no startup admin key is generated here.
exec "$@"
