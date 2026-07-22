from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

import torch
from filelock import FileLock

GPU_LEASE_ENV = "GENESIS_GPU_LEASE_PATH"
T = TypeVar("T")


def configured_gpu_lease_path() -> str | None:
    value = str(os.getenv(GPU_LEASE_ENV) or "").strip()
    if not value:
        return None
    return str(Path(value).expanduser().resolve(strict=False))


@contextmanager
def acquire_gpu_lease() -> Iterator[None]:
    """Serialize CUDA work with other GENESIS processes when configured.

    The local FastAPI asyncio lock protects this DIA process.  This optional
    file lock additionally coordinates DIA and Whisper containers that share a
    GPU and a mounted lock-file path.  CPU-only processes and an unset env var
    intentionally remain no-ops.
    """

    lock_path = configured_gpu_lease_path()
    if lock_path is None or not torch.cuda.is_available():
        yield
        return

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)):
        yield


async def run_blocking_gpu_phase(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Keep the caller's local async lock held until a CUDA thread really exits."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            pass
        raise
