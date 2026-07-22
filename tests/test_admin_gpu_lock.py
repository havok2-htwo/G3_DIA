import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from backend import genesis_dia_server_admin as admin


class AdminGpuLockTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        with admin.task_runtime_lock:
            self.runtime_snapshot = dict(admin.task_runtime_state)
        with admin.task_status_lock:
            self.status_snapshot = dict(admin.current_task_status)

    async def asyncTearDown(self) -> None:
        with admin.task_runtime_lock:
            admin.task_runtime_state.clear()
            admin.task_runtime_state.update(self.runtime_snapshot)
        with admin.task_status_lock:
            admin.current_task_status.clear()
            admin.current_task_status.update(self.status_snapshot)

    async def test_cancelled_model_load_keeps_local_lock_until_worker_exits(self) -> None:
        local_lock = asyncio.Lock()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(local_gpu_lock=local_lock)))
        worker_started = threading.Event()
        release_worker = threading.Event()

        def blocking_model_load() -> bool:
            worker_started.set()
            release_worker.wait(timeout=5)
            return True

        with mock.patch.object(admin, "load_diarization_model", side_effect=blocking_model_load):
            benchmark_task = asyncio.create_task(
                admin._run_admin_benchmark(request, np.zeros(1600, dtype=np.float32), 1)
            )
            self.assertTrue(await asyncio.to_thread(worker_started.wait, 2))
            self.assertTrue(local_lock.locked())

            benchmark_task.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(benchmark_task.done())
            self.assertTrue(local_lock.locked())

            release_worker.set()
            with self.assertRaises(asyncio.CancelledError):
                await benchmark_task

        self.assertFalse(local_lock.locked())
        with admin.task_status_lock:
            self.assertEqual(admin.current_task_status["task_name"], "Idle")
        with admin.task_runtime_lock:
            self.assertFalse(admin.task_runtime_state["worker_running"])
            self.assertEqual(admin.task_runtime_state["last_error"], "Benchmark request was cancelled.")

    async def test_cancelled_queued_benchmark_does_not_overwrite_active_status(self) -> None:
        local_lock = asyncio.Lock()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(local_gpu_lock=local_lock)))
        await local_lock.acquire()
        with admin.task_status_lock:
            admin.current_task_status.update(
                {"task_name": "Diarization", "progress": 42.0, "details": "Active API request"}
            )

        benchmark_task = asyncio.create_task(
            admin._run_admin_benchmark(request, np.zeros(1600, dtype=np.float32), 1)
        )
        await asyncio.sleep(0.02)
        benchmark_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await benchmark_task
        local_lock.release()

        with admin.task_status_lock:
            self.assertEqual(
                dict(admin.current_task_status),
                {"task_name": "Diarization", "progress": 42.0, "details": "Active API request"},
            )


if __name__ == "__main__":
    unittest.main()
