#!/bin/sh
set -e

# The DIA server's built-in default model cache path is the Windows literal ".\models",
# which is meaningless on Linux. On first boot (before any settings file exists) write a
# seed file that pins the cache to the mounted /app/models volume. The server itself
# writes the settings file on its first start: defaults, then this seed, then the
# "24GB VRAM" tier, so a fresh container gets the same tier as a fresh local install.
# Existing settings (written by the server or the admin dashboard) are left untouched.

SETTINGS_DIR="/app/logs"
SETTINGS_FILE="${SETTINGS_DIR}/genesis_dia_settings.json"
SEED_FILE="${SETTINGS_DIR}/genesis_dia_settings_seed.json"

mkdir -p "${SETTINGS_DIR}" /app/models

if [ ! -f "${SETTINGS_FILE}" ]; then
  echo "[entrypoint] Seeding ${SEED_FILE} (cache -> /app/models)"
  cat > "${SEED_FILE}" <<'JSON'
{
    "model_cache_path": "/app/models"
}
JSON
fi

# Admin access is username/password (default admin/admin, forced change on first login),
# so no startup admin key is generated here.
exec "$@"
