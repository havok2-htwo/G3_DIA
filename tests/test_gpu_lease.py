import asyncio
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from backend import genesis_dia_server_gpu_lease as gpu_lease


class GpuLeaseTests(unittest.TestCase):
    def test_unset_path_is_noop(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(gpu_lease.FileLock, "__init__") as init:
            with gpu_lease.acquire_gpu_lease():
                pass
        init.assert_not_called()

    def test_configured_cuda_path_uses_file_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="genesis-dia-lease-") as temp_dir:
            lock_path = Path(temp_dir) / "gpu.lock"
            with mock.patch.dict(os.environ, {gpu_lease.GPU_LEASE_ENV: str(lock_path)}, clear=True), mock.patch.object(
                gpu_lease.torch.cuda, "is_available", return_value=True
            ):
                with gpu_lease.acquire_gpu_lease():
                    self.assertTrue(lock_path.exists())


class BlockingGpuPhaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_waits_until_worker_thread_has_exited(self) -> None:
        worker_started = threading.Event()
        release_worker = threading.Event()
        worker_finished = threading.Event()

        def blocking_worker() -> None:
            worker_started.set()
            release_worker.wait(timeout=5)
            worker_finished.set()

        phase_task = asyncio.create_task(gpu_lease.run_blocking_gpu_phase(blocking_worker))
        self.assertTrue(await asyncio.to_thread(worker_started.wait, 2))

        phase_task.cancel()
        await asyncio.sleep(0.02)
        self.assertFalse(phase_task.done())
        self.assertFalse(worker_finished.is_set())

        release_worker.set()
        with self.assertRaises(asyncio.CancelledError):
            await phase_task
        self.assertTrue(worker_finished.is_set())

    async def test_worker_exception_is_propagated(self) -> None:
        def failing_worker() -> None:
            raise RuntimeError("worker failed")

        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            await gpu_lease.run_blocking_gpu_phase(failing_worker)


if __name__ == "__main__":
    unittest.main()
