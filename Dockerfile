# syntax=docker/dockerfile:1.7
#
# GENESIS DIA Server — GPU speaker-diarization service (pyannote).
# Target host: Linux + NVIDIA driver + nvidia-container-toolkit (RTX 5090 / sm_120 OK).
# No models are baked in: pyannote/speaker-diarization-community-1 is GATED and needs a
# HUGGINGFACE_TOKEN; it downloads into the mounted /app/models volume on first use.

############################  1) Frontend build  ############################
FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


############################  2) Python runtime  ############################
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/.hf

# System libraries:
#   ffmpeg         -> decode mp3/m4a/video; also brings the libav* libs pyannote's torchcodec needs
#   libsndfile1    -> soundfile
#   libsamplerate0 -> the `samplerate` resampler
#   build-essential-> g++ toolchain in case torch.compile / Inductor kernels are ever built
#   curl           -> container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libsamplerate0 \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CUDA 12.8 wheels (Blackwell / sm_120 OK), pinned to the verified pyannote stack.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Application code + built frontend.
COPY backend/ ./backend/
COPY --from=frontend /build/frontend/dist ./frontend/dist

# Entrypoint seeds a Linux-correct settings file on first boot (built-in default cache
# path is the Windows literal ".\\models").
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Non-root; pre-own the volume mountpoints so named volumes inherit ownership.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/models /app/logs /app/frontend \
    && chown -R app:app /app
USER app

EXPOSE 7864
VOLUME ["/app/models", "/app/logs"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=1200s --retries=5 \
    CMD curl -fsS http://localhost:7864/openapi.json >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "backend.genesis_dia_server"]
