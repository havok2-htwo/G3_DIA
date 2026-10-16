"""Skip speaker-embedding inference for inactive (chunk, local speaker) slots.

pyannote's ``SpeakerDiarization.get_embeddings`` runs the WeSpeaker ResNet34 on
every (10 s chunk, local speaker) pair, three per chunk, including local
speakers that are silent in that chunk.  Their mask is all zeros, so weighted
statistics pooling sees ``sum(x * 0) / (0 + 1e-8) == 0`` and the network
returns the same constant vector for every such slot, whatever the audio.
Clustering still consumes those vectors (constrained assignment per chunk),
so they cannot simply be dropped -- but they do not need the GPU.

The embedding phase is ~80 % of diarization time on this box, and in normal
conversations roughly half of all slots are inactive.  This module computes
the constant once, verifies it really is input- and batch-size independent,
then only runs the network on slots with a non-zero mask (repacked into full
batches) and fills the rest with the constant.  Same masks and the same
constant as upstream; only slots that end up in a differently sized (last)
batch than upstream may differ by fp32 rounding (~1e-6, GPU kernels are chosen per
batch size), exactly like the rows in upstream's own partial last batch.

If the self-check fails (other pyannote/model version, NaN, ...), the pipeline
keeps the upstream implementation.  Training mode always uses upstream.
"""

from __future__ import annotations

import math
import sys
import threading
import types
from typing import Any, Callable, Optional

import numpy as np
import torch

_INSTALLED_ATTR = "_genesis_inactive_skip"
_check_lock = threading.Lock()


def _batches(iterable, batch_size: int):
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_inactive_embedding(embedding: Callable, num_samples: int, num_frames: int, batch_size: int) -> Optional[np.ndarray]:
    """Embedding the model returns for an all-zero mask, or None if it is not a constant.

    Runs one full batch of different waveforms (and a single-sample batch) with
    all-zero masks; every row must be bit-identical and finite.
    """

    batch_size = max(2, int(batch_size))
    generator = torch.Generator().manual_seed(1234)
    waveforms = torch.randn((batch_size, 1, num_samples), generator=generator) * 0.1
    waveforms[0].zero_()
    masks = torch.zeros((batch_size, num_frames), dtype=torch.float32)
    rows = np.asarray(embedding(waveforms, masks=masks))
    single = np.asarray(embedding(waveforms[1:2] * 0.5, masks=masks[:1]))
    if rows.ndim != 2 or rows.shape[0] != batch_size or single.shape != (1, rows.shape[1]):
        return None
    constant = rows[0]
    if not np.all(np.isfinite(constant)):
        return None
    if not np.array_equal(rows, np.broadcast_to(constant, rows.shape)) or not np.array_equal(single[0], constant):
        return None
    return constant.copy()


def _get_inactive_embedding(pipeline: Any, num_samples: int, num_frames: int) -> Optional[np.ndarray]:
    state = getattr(pipeline, _INSTALLED_ATTR)
    key = (id(pipeline._embedding), int(num_samples), int(num_frames))
    with _check_lock:
        if key not in state["constants"]:
            try:
                constant = compute_inactive_embedding(
                    pipeline._embedding, int(num_samples), int(num_frames), pipeline.embedding_batch_size
                )
            except Exception as exc:  # never let the shortcut break diarization
                print(f"[WARNUNG-DIA] Inaktive-Sprecher-Optimierung deaktiviert: {type(exc).__name__}: {exc}", file=sys.stderr)
                constant = None
            if constant is None:
                print(
                    "[WARNUNG-DIA] Embedding fuer inaktive Sprecher ist nicht konstant; nutze pyannote-Standardpfad.",
                    file=sys.stderr,
                )
            state["constants"][key] = constant
        return state["constants"][key]


def release_segmentation_cache(state: dict) -> None:
    """Hand the segmentation phase's cached CUDA blocks back before the embedding phase (off by default).

    pyannote calls ``get_embeddings`` right after the segmentation finished. The idea was
    that the embedding phase would otherwise grow the pool on top of the segmentation's
    cached blocks. Measured on the RTX 5090 (batch 16, 5 and 60 min audio) it is the
    other way round: with the release the request peaks at 2560 MiB process / 1994 MiB
    reserved, without it at 2052 / 1486 MiB, at the same speed. The embedding phase
    reuses the cached segments; after ``empty_cache`` it allocates fresh ones and ends
    up ~0.5 GB higher. Kept as a setting until the RTX 4090 is re-checked.
    """

    release = state.get("release_between_phases")
    if release is None or not release():
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        pass


def fast_get_embeddings(self, file, binary_segmentations, exclude_overlap: bool = False, hook: Optional[Callable] = None):
    """Drop-in for ``SpeakerDiarization.get_embeddings`` (pyannote.audio 4.0)."""

    state = getattr(self, _INSTALLED_ATTR)
    original = state["original"]
    release_segmentation_cache(state)
    if self.training or not state["enabled"]():
        return original(file, binary_segmentations, exclude_overlap=exclude_overlap, hook=hook)

    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape
    num_samples = int(round(duration * self._embedding.sample_rate))

    inactive_embedding = _get_inactive_embedding(self, num_samples, num_frames)
    if inactive_embedding is None:
        return original(file, binary_segmentations, exclude_overlap=exclude_overlap, hook=hook)

    # Mask selection identical to upstream get_embeddings.
    if exclude_overlap:
        min_num_samples = self._embedding.min_num_samples
        min_num_frames = math.ceil(num_frames * min_num_samples / (duration * self._embedding.sample_rate))
        clean_frames = 1.0 * (np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2)
        clean_data = binary_segmentations.data * clean_frames
    else:
        min_num_frames = -1
        clean_data = binary_segmentations.data

    all_masks = np.nan_to_num(binary_segmentations.data, nan=0.0).astype(np.float32)
    all_clean = np.nan_to_num(clean_data, nan=0.0).astype(np.float32)
    # (num_chunks, num_frames, num_speakers)
    use_clean = np.sum(all_clean, axis=1) > min_num_frames
    used_masks = np.where(use_clean[:, None, :], all_clean, all_masks)
    active = np.any(used_masks != 0.0, axis=1)
    # (num_chunks, num_speakers)

    sliding_window = binary_segmentations.sliding_window
    batch_size = self.embedding_batch_size
    num_active = int(active.sum())
    batch_count = math.ceil(num_active / batch_size)

    def iter_active():
        for c in np.flatnonzero(active.any(axis=1)):
            waveform, _ = self._audio.crop(file, sliding_window[int(c)], mode="pad")
            for s in np.flatnonzero(active[c]):
                yield int(c) * num_speakers + int(s), waveform[None], torch.from_numpy(np.ascontiguousarray(used_masks[c, :, s]))[None]

    embeddings = np.empty((num_chunks * num_speakers, inactive_embedding.shape[0]), dtype=inactive_embedding.dtype)
    embeddings[:] = inactive_embedding

    if hook is not None:
        hook("embeddings", None, total=batch_count, completed=0)

    for i, batch in enumerate(_batches(iter_active(), batch_size), 1):
        indices, waveforms, masks = zip(*batch)
        embedding_batch = self._embedding(torch.vstack(waveforms), masks=torch.vstack(masks))
        embeddings[list(indices)] = embedding_batch
        if hook is not None:
            hook("embeddings", embedding_batch, total=batch_count, completed=i)

    state["stats"] = {"slots": num_chunks * num_speakers, "embedded": num_active}
    return embeddings.reshape(num_chunks, num_speakers, -1)


def install(
    pipeline: Any,
    enabled: Callable[[], bool],
    release_between_phases: Optional[Callable[[], bool]] = None,
) -> bool:
    """Route ``pipeline.get_embeddings`` through :func:`fast_get_embeddings`.

    ``enabled`` and ``release_between_phases`` are read on every call so the admin
    settings apply without a reload. Returns False when the pipeline does not look
    like pyannote's SpeakerDiarization.
    """

    state = getattr(pipeline, "__dict__", {}).get(_INSTALLED_ATTR)
    if isinstance(state, dict):
        state["enabled"] = enabled
        state["release_between_phases"] = release_between_phases
        return True
    required = ("get_embeddings", "_embedding", "_audio", "embedding_batch_size", "training")
    if not all(hasattr(pipeline, name) for name in required):
        return False
    setattr(
        pipeline,
        _INSTALLED_ATTR,
        {
            "original": pipeline.get_embeddings,
            "enabled": enabled,
            "release_between_phases": release_between_phases,
            "constants": {},
            "stats": None,
        },
    )
    pipeline.get_embeddings = types.MethodType(fast_get_embeddings, pipeline)
    return True


def last_stats(pipeline: Any) -> Optional[dict]:
    state = getattr(pipeline, "__dict__", {}).get(_INSTALLED_ATTR)
    return dict(state["stats"]) if isinstance(state, dict) and state.get("stats") else None
