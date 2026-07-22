import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from backend import genesis_dia_server_api as api


class ApiGpuLockTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        with api.task_runtime_lock:
            self.runtime_snapshot = dict(api.task_runtime_state)
        with api.task_status_lock:
            self.status_snapshot = dict(api.current_task_status)

    async def asyncTearDown(self) -> None:
        with api.task_runtime_lock:
            api.task_runtime_state.clear()
            api.task_runtime_state.update(self.runtime_snapshot)
        with api.task_status_lock:
            api.current_task_status.clear()
            api.current_task_status.update(self.status_snapshot)

    async def test_cancelled_waiter_does_not_clobber_active_request_status(self) -> None:
        local_lock = asyncio.Lock()
        await local_lock.acquire()
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(local_gpu_lock=local_lock)),
            client=SimpleNamespace(host="127.0.0.1"),
        )
        with api.task_runtime_lock:
            api.task_runtime_state["pending_requests"] = 0
            api.task_runtime_state["worker_running"] = True
        with api.task_status_lock:
            api.current_task_status.update(
                {"task_name": "Diarization", "progress": 42.0, "details": "Active API request"}
            )

        with mock.patch.object(api, "authorize_api_key", return_value=None), mock.patch.object(
            api,
            "_decode_upload",
            new=mock.AsyncMock(return_value=(np.zeros(1600, dtype=np.float32), "queued.wav", 0.1)),
        ):
            queued_task = asyncio.create_task(
                api._run_request(
                    request,
                    None,
                    None,
                    None,
                    None,
                    endpoint_path="/v2/diarize",
                    engine=lambda *_: {},
                    result_counts=lambda _: (0, 0),
                )
            )
            await asyncio.sleep(0.02)
            queued_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued_task

        local_lock.release()
        with api.task_status_lock:
            self.assertEqual(
                dict(api.current_task_status),
                {"task_name": "Diarization", "progress": 42.0, "details": "Active API request"},
            )
        with api.task_runtime_lock:
            self.assertEqual(api.task_runtime_state["pending_requests"], 0)
            self.assertTrue(api.task_runtime_state["worker_running"])


if __name__ == "__main__":
    unittest.main()
