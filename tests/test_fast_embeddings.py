import asyncio
import types
import unittest
from unittest import mock

import numpy as np
import torch
from fastapi import FastAPI
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.core import SlidingWindow, SlidingWindowFeature

from backend import genesis_dia_server as server
from backend import genesis_dia_server_engine as engine
from backend import genesis_dia_server_fast_embeddings as fast
from backend.genesis_dia_server_globals import current_settings, diarization_pipeline, model_status
from backend.genesis_dia_server_storage import DEFAULT_SETTINGS, normalize_settings

SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.0
NUM_FRAMES = 20
DIM = 6


class FakeEmbedding:
    """Mimics WeSpeaker + weighted stats pooling: an all-zero mask yields the bias only."""

    sample_rate = SAMPLE_RATE
    min_num_samples = 400

    def __init__(self, constant_for_zero_mask: bool = True) -> None:
        self.constant_for_zero_mask = constant_for_zero_mask
        self.rows_seen = 0
        self.calls = 0
        rng = np.random.default_rng(7)
        self.weight = rng.standard_normal((2, DIM)).astype(np.float32)
        self.bias = rng.standard_normal(DIM).astype(np.float32)

    def __call__(self, waveforms: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
        self.calls += 1
        self.rows_seen += waveforms.shape[0]
        wav = waveforms[:, 0, :].numpy()
        frames = wav[:, : (wav.shape[1] // masks.shape[1]) * masks.shape[1]].reshape(wav.shape[0], masks.shape[1], -1).mean(axis=2)
        weights = masks.numpy()
        mean = (frames * weights).sum(axis=1) / (weights.sum(axis=1) + 1e-8)
        energy = (np.abs(frames) * weights).sum(axis=1) / (weights.sum(axis=1) + 1e-8)
        stats = np.stack([mean, energy], axis=1).astype(np.float32)
        if not self.constant_for_zero_mask:
            stats = stats + wav[:, :1]  # leaks the waveform into every row
        return stats @ self.weight + self.bias


def make_pipeline(embedding: FakeEmbedding, batch_size: int = 4):
    fake = types.SimpleNamespace(
        training=False,
        _embedding=embedding,
        _audio=Audio(sample_rate=SAMPLE_RATE, mono="downmix"),
        embedding_batch_size=batch_size,
        _segmentation=types.SimpleNamespace(model=types.SimpleNamespace(specifications=types.SimpleNamespace(powerset=True))),
    )
    fake.get_embeddings = types.MethodType(SpeakerDiarization.get_embeddings, fake)
    return fake


def make_inputs(num_chunks: int = 9, num_speakers: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    duration = CHUNK_SECONDS * (num_chunks + 1)
    waveform = torch.from_numpy(rng.standard_normal((1, int(duration * SAMPLE_RATE))).astype(np.float32))
    file = {"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": "test"}
    data = (rng.random((num_chunks, NUM_FRAMES, num_speakers)) > 0.55).astype(np.float32)
    data[1, :, :] = 0.0  # chunk without any speaker
    data[2, :, 1:] = 0.0  # only speaker 0 active
    data[3, 4, 2] = np.nan  # partial stitching leaves NaNs
    data[4, :, 0] = 1.0  # speaker 0 always on -> overlap everywhere
    window = SlidingWindow(start=0.0, duration=CHUNK_SECONDS, step=CHUNK_SECONDS)
    return file, SlidingWindowFeature(data, window)


class FastEmbeddingsTests(unittest.TestCase):
    def _compare(self, exclude_overlap: bool, batch_size: int, seed: int) -> None:
        file, segmentations = make_inputs(seed=seed)
        reference_embedding = FakeEmbedding()
        expected = make_pipeline(reference_embedding, batch_size).get_embeddings(file, segmentations, exclude_overlap=exclude_overlap)

        fast_embedding = FakeEmbedding()
        pipeline = make_pipeline(fast_embedding, batch_size)
        self.assertTrue(fast.install(pipeline, lambda: True))
        actual = pipeline.get_embeddings(file, segmentations, exclude_overlap=exclude_overlap)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)
        stats = fast.last_stats(pipeline)
        self.assertEqual(stats["slots"], 27)
        self.assertLess(stats["embedded"], stats["slots"])
        # constant check (batch + single sample) plus only the active slots
        self.assertEqual(fast_embedding.rows_seen - max(2, batch_size) - 1, stats["embedded"])

    def test_output_identical_to_pyannote(self) -> None:
        for exclude_overlap in (True, False):
            for batch_size in (1, 4, 32):
                for seed in (0, 1):
                    with self.subTest(exclude_overlap=exclude_overlap, batch_size=batch_size, seed=seed):
                        self._compare(exclude_overlap, batch_size, seed)

    def test_falls_back_when_zero_mask_embedding_is_not_constant(self) -> None:
        file, segmentations = make_inputs()
        expected = make_pipeline(FakeEmbedding(False)).get_embeddings(file, segmentations, exclude_overlap=True)
        pipeline = make_pipeline(FakeEmbedding(False))
        fast.install(pipeline, lambda: True)
        with mock.patch("builtins.print"):
            actual = pipeline.get_embeddings(file, segmentations, exclude_overlap=True)
        np.testing.assert_array_equal(actual, expected)
        self.assertIsNone(fast.last_stats(pipeline))

    def test_disabled_setting_and_training_use_upstream(self) -> None:
        file, segmentations = make_inputs()
        for enabled, training in ((False, False), (True, True)):
            embedding = FakeEmbedding()
            pipeline = make_pipeline(embedding)
            fast.install(pipeline, lambda enabled=enabled: enabled)
            pipeline.training = training
            if training:
                file = dict(file)
            pipeline.get_embeddings(file, segmentations, exclude_overlap=True)
            self.assertEqual(embedding.rows_seen, 27, (enabled, training))

    def test_hook_reports_batches_of_active_slots(self) -> None:
        file, segmentations = make_inputs()
        pipeline = make_pipeline(FakeEmbedding(), batch_size=4)
        fast.install(pipeline, lambda: True)
        hook = mock.Mock()
        pipeline.get_embeddings(file, segmentations, exclude_overlap=True, hook=hook)
        embedded = fast.last_stats(pipeline)["embedded"]
        totals = {call.kwargs["total"] for call in hook.call_args_list}
        self.assertEqual(totals, {-(-embedded // 4)})
        self.assertEqual(hook.call_args_list[-1].kwargs["completed"], -(-embedded // 4))

    def test_install_is_idempotent_and_rejects_foreign_pipelines(self) -> None:
        pipeline = make_pipeline(FakeEmbedding())
        self.assertTrue(fast.install(pipeline, lambda: True))
        bound = pipeline.get_embeddings
        self.assertTrue(fast.install(pipeline, lambda: False))
        self.assertIs(pipeline.get_embeddings, bound)
        self.assertFalse(fast.install(types.SimpleNamespace(), lambda: True))


class PreloadSettingsTests(unittest.TestCase):
    def test_new_settings_default_on_and_parse_strings(self) -> None:
        self.assertTrue(DEFAULT_SETTINGS["preload_model_on_startup"])
        self.assertTrue(DEFAULT_SETTINGS["skip_inactive_embeddings"])
        old = normalize_settings({"diarization_model_id": "x"})
        self.assertTrue(old["preload_model_on_startup"])
        self.assertTrue(old["skip_inactive_embeddings"])
        parsed = normalize_settings({"preload_model_on_startup": "off", "skip_inactive_embeddings": False})
        self.assertFalse(parsed["preload_model_on_startup"])
        self.assertFalse(parsed["skip_inactive_embeddings"])


class PreloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._settings_backup = dict(current_settings)
        self._pipeline_backup = dict(diarization_pipeline)
        self._status_backup = dict(model_status)
        current_settings.clear()
        current_settings.update(normalize_settings({"huggingface_token": "hf_test", "gpu_device": "cpu", "model_cache_path": ""}))
        diarization_pipeline.update({"pipeline": None, "model_identifier": None, "device": None})

    def tearDown(self) -> None:
        current_settings.clear()
        current_settings.update(self._settings_backup)
        diarization_pipeline.clear()
        diarization_pipeline.update(self._pipeline_backup)
        model_status.clear()
        model_status.update(self._status_backup)

    def test_cpu_preload_loads_without_warmup(self) -> None:
        with mock.patch.object(engine, "_from_pretrained_any", return_value=mock.Mock()), mock.patch.object(
            engine, "_warmup_pipeline_unleased"
        ) as warmup:
            self.assertTrue(engine.preload_and_warmup_model())
        warmup.assert_not_called()
        self.assertEqual(engine.get_model_status()["state"], "loaded")

    def test_gpu_warmup_failure_keeps_the_model_loaded(self) -> None:
        with mock.patch.object(engine, "_from_pretrained_any", return_value=mock.Mock()), mock.patch.object(
            engine, "resolve_device", return_value=torch.device("cuda:0")
        ), mock.patch.object(engine, "_warmup_pipeline_unleased", side_effect=RuntimeError("boom")), mock.patch.object(
            engine.torch.cuda, "empty_cache"
        ) as empty_cache, mock.patch.object(engine, "acquire_gpu_lease", return_value=mock.MagicMock()), mock.patch("builtins.print"):
            self.assertTrue(engine.preload_and_warmup_model())
        empty_cache.assert_called_once()
        self.assertEqual(engine.get_model_status()["state"], "loaded")

    def test_failed_load_is_reported_not_raised(self) -> None:
        with mock.patch.object(engine, "_from_pretrained_any", side_effect=OSError("disk")), mock.patch("builtins.print"):
            self.assertFalse(engine.preload_and_warmup_model())
        self.assertEqual(engine.get_model_status()["state"], "error")

    def test_startup_schedules_preload_only_when_enabled(self) -> None:
        async def scenario(enabled: bool):
            app = FastAPI()
            settings = normalize_settings({"preload_model_on_startup": enabled})
            with mock.patch.object(server, "load_settings", return_value=settings), mock.patch.object(
                server, "preload_and_warmup_model", return_value=True
            ) as preload, mock.patch("builtins.print"):
                await server.startup_server(app)
                if app.state.preload_task is not None:
                    self.assertTrue(await app.state.preload_task)
                await server.shutdown_server(app)
            return preload.call_count

        self.assertEqual(asyncio.run(scenario(True)), 1)
        self.assertEqual(asyncio.run(scenario(False)), 0)


class CudaCacheReleaseTests(unittest.TestCase):
    def test_cache_is_released_after_success_and_failure(self) -> None:
        for side_effect in (None, RuntimeError("boom")):
            with mock.patch.object(engine, "_run_diarization_pipeline_unleased", side_effect=side_effect, return_value="ok"), mock.patch.object(
                engine, "_release_cuda_cache"
            ) as release, mock.patch.object(engine, "_uses_gpu_lease", return_value=False):
                if side_effect is None:
                    self.assertEqual(engine._run_diarization_pipeline(np.zeros(16000, dtype=np.float32)), "ok")
                else:
                    with self.assertRaises(RuntimeError):
                        engine._run_diarization_pipeline(np.zeros(16000, dtype=np.float32))
            release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
