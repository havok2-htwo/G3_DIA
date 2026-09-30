import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from backend import genesis_dia_server_devices as devices
from backend import genesis_dia_server_engine as engine
from backend.genesis_dia_server_globals import current_settings, diarization_pipeline, model_status
from backend.genesis_dia_server_storage import normalize_device_setting, normalize_settings


class DeviceSettingTests(unittest.TestCase):
    def test_normalizes_device_values(self) -> None:
        self.assertEqual(normalize_device_setting("CUDA:1"), "cuda:1")
        self.assertEqual(normalize_device_setting("cuda"), "cuda:0")
        self.assertEqual(normalize_device_setting(" cpu "), "cpu")
        self.assertEqual(normalize_device_setting("gpu-7"), "auto")
        self.assertEqual(normalize_device_setting(None), "auto")

    def test_old_settings_file_defaults_to_auto(self) -> None:
        settings = normalize_settings({"diarization_model_id": "m", "model_cache_path": "/x", "huggingface_token": ""})
        self.assertEqual(settings["gpu_device"], "auto")


class ResolveDeviceTests(unittest.TestCase):
    def _cuda(self, available: bool, count: int = 0, current: int = 0):
        return mock.patch.multiple(
            devices.torch.cuda,
            is_available=mock.Mock(return_value=available),
            device_count=mock.Mock(return_value=count),
            current_device=mock.Mock(return_value=current),
        )

    def test_auto_prefers_current_gpu_and_falls_back_to_cpu(self) -> None:
        with self._cuda(True, count=2, current=1):
            self.assertEqual(devices.resolve_device("auto"), torch.device("cuda:1"))
        with self._cuda(False):
            self.assertEqual(devices.resolve_device("auto"), torch.device("cpu"))

    def test_explicit_gpu_and_cpu(self) -> None:
        with self._cuda(True, count=2):
            self.assertEqual(devices.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(devices.resolve_device("cpu"), torch.device("cpu"))

    def test_missing_gpu_fails_loudly(self) -> None:
        with self._cuda(True, count=1):
            with self.assertRaisesRegex(devices.DeviceUnavailableError, "1 GPU"):
                devices.resolve_device("cuda:3")
        with self._cuda(False):
            with self.assertRaisesRegex(devices.DeviceUnavailableError, "CUDA ist nicht verfuegbar"):
                devices.resolve_device("cuda:0")

    def test_lists_every_gpu_after_auto_and_cpu(self) -> None:
        properties = SimpleNamespace(name="RTX Test", total_memory=24 * 1024 ** 3)
        with self._cuda(True, count=2), mock.patch.object(
            devices.torch.cuda, "get_device_properties", return_value=properties
        ):
            options = devices.list_device_options()
        self.assertEqual([option["value"] for option in options], ["auto", "cpu", "cuda:0", "cuda:1"])
        self.assertEqual(options[2]["label"], "GPU 0: RTX Test (24 GB)")


class ModelLoadStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._settings_backup = dict(current_settings)
        self._pipeline_backup = dict(diarization_pipeline)
        self._status_backup = dict(model_status)
        current_settings.clear()
        current_settings.update(
            {
                "diarization_model_id": "pyannote/speaker-diarization-community-1",
                "model_cache_path": "",
                "huggingface_token": "hf_test",
                "gpu_device": "cpu",
            }
        )
        diarization_pipeline.update({"pipeline": None, "model_identifier": None, "device": None})

    def tearDown(self) -> None:
        current_settings.clear()
        current_settings.update(self._settings_backup)
        diarization_pipeline.clear()
        diarization_pipeline.update(self._pipeline_backup)
        model_status.clear()
        model_status.update(self._status_backup)

    def test_successful_load_records_device_and_token_source(self) -> None:
        fake_pipeline = mock.Mock()
        with mock.patch.object(engine, "_from_pretrained_any", return_value=fake_pipeline):
            self.assertTrue(engine._load_diarization_model_unleased())

        fake_pipeline.to.assert_called_once_with(torch.device("cpu"))
        status = engine.get_model_status()
        self.assertEqual(status["state"], "loaded")
        self.assertEqual(status["device"], "cpu")
        self.assertEqual(status["token_source"], "settings")
        self.assertIsNone(status["error"])
        self.assertEqual(diarization_pipeline["device"], torch.device("cpu"))

    def test_none_pipeline_becomes_a_visible_error(self) -> None:
        with mock.patch.object(engine, "_from_pretrained_any", return_value=None):
            self.assertFalse(engine._load_diarization_model_unleased())
        status = engine.get_model_status()
        self.assertEqual(status["state"], "error")
        self.assertIn("keine Pipeline-Konfiguration", status["error"])

    def test_http_401_explains_gated_access(self) -> None:
        error = RuntimeError("401 Client Error: Unauthorized for url: https://huggingface.co/...")
        with mock.patch.object(engine, "_from_pretrained_any", side_effect=error):
            self.assertFalse(engine._load_diarization_model_unleased())
        message = engine.get_model_status()["error"]
        self.assertIn("abgelehnt", message)
        self.assertIn("https://hf.co/pyannote/speaker-diarization-community-1", message)
        self.assertIn("gated repos", message)

    def test_network_failure_is_reported_as_unreachable_hub(self) -> None:
        error = OSError("Max retries exceeded with url ... Temporary failure in name resolution")
        with mock.patch.object(engine, "_from_pretrained_any", side_effect=error):
            self.assertFalse(engine._load_diarization_model_unleased())
        self.assertIn("nicht erreichbar", engine.get_model_status()["error"])

    def test_token_without_hf_prefix_is_rejected_before_download(self) -> None:
        current_settings["huggingface_token"] = "abc123"
        with mock.patch.object(engine, "_from_pretrained_any") as from_pretrained:
            self.assertFalse(engine._load_diarization_model_unleased())
        from_pretrained.assert_not_called()
        self.assertIn("hf_", engine.get_model_status()["error"])

    def test_missing_gpu_is_reported_without_loading(self) -> None:
        current_settings["gpu_device"] = "cuda:5"
        with (
            mock.patch.object(engine, "resolve_device", side_effect=devices.DeviceUnavailableError("GPU 'cuda:5' existiert nicht")),
            mock.patch.object(engine, "_from_pretrained_any") as from_pretrained,
        ):
            self.assertFalse(engine._load_diarization_model_unleased())
        from_pretrained.assert_not_called()
        self.assertIn("cuda:5", engine.get_model_status()["error"])

    def test_device_change_triggers_a_reload(self) -> None:
        first, second = mock.Mock(), mock.Mock()
        with mock.patch.object(engine, "_from_pretrained_any", side_effect=[first, second]) as from_pretrained, mock.patch.object(
            engine, "resolve_device", side_effect=[torch.device("cpu"), torch.device("cpu"), torch.device("cuda:1")]
        ):
            self.assertTrue(engine._load_diarization_model_unleased())
            self.assertTrue(engine._load_diarization_model_unleased())  # same device: cached
            current_settings["gpu_device"] = "cuda:1"
            self.assertTrue(engine._load_diarization_model_unleased())
        self.assertEqual(from_pretrained.call_count, 2)
        second.to.assert_called_once_with(torch.device("cuda:1"))


if __name__ == "__main__":
    unittest.main()
