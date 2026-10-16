"""Settings presets: built-in GPU tiers, card matching for the two-GPU tier, saved presets and the admin routes.

No GPU: the detected cards are faked, and every test writes its presets file into a temporary directory.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI, HTTPException

from backend import genesis_dia_server_admin as admin
from backend import genesis_dia_server_presets as presets
from backend.genesis_dia_server_storage import DEFAULT_SETTINGS, normalize_settings

TIER_NAMES = ["32GB VRAM", "24GB VRAM", "8+24 VRAM"]


class PresetTestCase(unittest.TestCase):
    """Redirects the presets file to a temp dir (never logs/) and fakes the detected CUDA cards."""

    cards_mib: tuple = ()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(presets, "PRESETS_FILE", Path(self._tmp.name) / "genesis_dia_presets.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.set_cards(*self.cards_mib)

    def set_cards(self, *cards_mib: int) -> None:
        patcher = mock.patch.object(presets, "detect_cuda_memory_mib", return_value=list(cards_mib))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tier(self, name: str) -> dict:
        return next(preset for preset in presets.resolve_presets() if preset["name"] == name)


class BuiltinTierTests(PresetTestCase):
    def test_builtin_tiers_come_first_in_tier_order(self) -> None:
        self.set_cards(32607)
        presets.save_user_preset("aaa first by name", {"embedding_batch_size": 8})
        names = [preset["name"] for preset in presets.resolve_presets()]
        self.assertEqual(names[: len(TIER_NAMES)], TIER_NAMES)
        self.assertEqual(names[len(TIER_NAMES):], ["aaa first by name"])

    def test_dia_has_no_24_plus_8_tier(self) -> None:
        self.assertNotIn("24+8 VRAM", presets.BUILTIN_PRESETS)

    def test_builtin_values_are_valid_settings(self) -> None:
        for name in TIER_NAMES:
            with self.subTest(name=name):
                values = presets.BUILTIN_PRESETS[name]["values"]
                self.assertTrue(values)
                self.assertLessEqual(set(values), set(presets.PRESET_KEYS))
                normalized = normalize_settings(values)
                self.assertEqual({key: normalized[key] for key in values}, values, "a built-in value would be clamped")

    def test_builtin_values_are_the_defaults_until_measured(self) -> None:
        for name in TIER_NAMES:
            for key, value in presets.BUILTIN_PRESETS[name]["values"].items():
                self.assertEqual(value, DEFAULT_SETTINGS[key], f"{name}.{key}")

    def test_preset_keys_hold_no_secrets_or_paths(self) -> None:
        for key in ("huggingface_token", "model_cache_path", "diarization_model_id"):
            self.assertNotIn(key, presets.PRESET_KEYS)
        self.assertLessEqual(set(presets.PRESET_KEYS), set(DEFAULT_SETTINGS))

    def test_two_gpu_tier_picks_the_8gb_card_wherever_it_sits(self) -> None:
        for cards, expected in (((24576, 8192), "cuda:1"), ((8192, 24576), "cuda:0"), ((24576, 7979), "cuda:1")):
            with self.subTest(cards=cards):
                self.set_cards(*cards)
                tier = self.tier("8+24 VRAM")
                self.assertEqual(tier["values"]["gpu_device"], expected)
                self.assertIsNone(tier["note"])

    def test_two_gpu_tier_without_a_matching_card_leaves_the_gpu_alone(self) -> None:
        self.set_cards(32607)
        tier = self.tier("8+24 VRAM")
        self.assertNotIn("gpu_device", tier["values"])
        self.assertIn("No 8 GB card", tier["note"])
        self.assertIn("GPU 0: 32 GB", tier["note"])

    def test_two_gpu_tier_without_cuda_says_so(self) -> None:
        tier = self.tier("8+24 VRAM")
        self.assertNotIn("gpu_device", tier["values"])
        self.assertIn("cards: none", tier["note"])

    def test_single_card_tiers_never_set_the_gpu(self) -> None:
        self.set_cards(24576, 8192)
        for preset in presets.resolve_presets():
            if preset["name"] in ("32GB VRAM", "24GB VRAM"):
                self.assertNotIn("gpu_device", preset["values"])
                self.assertIsNone(preset["note"])

    def test_resolving_does_not_mutate_the_builtin_dict(self) -> None:
        self.set_cards(8192)
        self.assertEqual(self.tier("8+24 VRAM")["values"]["gpu_device"], "cuda:0")
        self.assertNotIn("gpu_device", presets.BUILTIN_PRESETS["8+24 VRAM"]["values"])


class UserPresetTests(PresetTestCase):
    def test_saved_preset_keeps_only_preset_keys_normalised(self) -> None:
        saved = presets.save_user_preset(
            "  my   tier ",
            {
                "embedding_batch_size": 999,
                "huggingface_token": "hf_secret",
                "model_cache_path": "D:\\models",
                "diarization_model_id": "someone/else",
                "nonsense": 1,
                "release_between_phases": "off",
                "gpu_device": "CUDA",
            },
        )
        self.assertEqual(saved["name"], "my tier")
        self.assertEqual(
            saved["values"], {"gpu_device": "cuda:0", "embedding_batch_size": 128, "release_between_phases": False}
        )
        listed = [preset for preset in presets.resolve_presets() if not preset["built_in"]]
        self.assertEqual([(preset["name"], preset["values"]) for preset in listed], [("my tier", saved["values"])])
        text = presets.PRESETS_FILE.read_text(encoding="utf-8")
        self.assertNotIn("hf_secret", text)
        self.assertNotIn("D:\\\\models", text)

    def test_overwrite_keeps_created_at(self) -> None:
        first = presets.save_user_preset("lab", {"embedding_batch_size": 8})
        second = presets.save_user_preset("lab", {"embedding_batch_size": 32})
        self.assertEqual(first["created_at"], second["created_at"])
        self.assertEqual(self.tier("lab")["values"], {"embedding_batch_size": 32})

    def test_builtin_names_are_protected(self) -> None:
        with self.assertRaises(ValueError):
            presets.save_user_preset("32GB VRAM", {"embedding_batch_size": 8})
        with self.assertRaises(ValueError):
            presets.save_user_preset(" 8+24   VRAM ", {"embedding_batch_size": 8})
        with self.assertRaises(ValueError):
            presets.delete_user_preset("24GB VRAM")

    def test_builtin_name_in_the_file_is_ignored(self) -> None:
        presets.PRESETS_FILE.write_text('{"32GB VRAM": {"values": {"embedding_batch_size": 1}}}', encoding="utf-8")
        self.assertEqual(self.tier("32GB VRAM")["values"]["embedding_batch_size"], DEFAULT_SETTINGS["embedding_batch_size"])
        self.assertTrue(all(preset["built_in"] for preset in presets.resolve_presets()))

    def test_corrupt_file_yields_only_builtins(self) -> None:
        presets.PRESETS_FILE.write_text("{not json", encoding="utf-8")
        self.assertEqual([preset["name"] for preset in presets.resolve_presets()], TIER_NAMES)

    def test_delete_saved_preset(self) -> None:
        presets.save_user_preset("tmp", {"embedding_batch_size": 8})
        self.assertTrue(presets.delete_user_preset("tmp"))
        self.assertFalse(presets.delete_user_preset("tmp"))
        self.assertTrue(all(preset["built_in"] for preset in presets.resolve_presets()))

    def test_empty_or_long_names_and_empty_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            presets.save_user_preset("   ", {"embedding_batch_size": 8})
        with self.assertRaises(ValueError):
            presets.save_user_preset("x" * 65, {"embedding_batch_size": 8})
        with self.assertRaises(ValueError):
            presets.save_user_preset("only unknown keys", {"nonsense": 1, "huggingface_token": "hf_x"})

    def test_no_temp_file_is_left_behind(self) -> None:
        presets.save_user_preset("lab", {"embedding_batch_size": 8})
        self.assertEqual([path.name for path in presets.PRESETS_FILE.parent.iterdir()], ["genesis_dia_presets.json"])


class AdminPresetRouteTests(PresetTestCase):
    cards_mib = (32607,)

    def setUp(self) -> None:
        super().setUp()
        app = admin.create_admin_api(FastAPI())
        self.endpoints = {
            (method, route.path): route.endpoint
            for route in app.routes
            for method in getattr(route, "methods", set())
        }

    def call(self, method: str, path: str, *args):
        return asyncio.run(self.endpoints[(method, path)](*args, {"username": "admin"}))

    def test_admin_preset_routes(self) -> None:
        listed = self.call("GET", "/api/admin/settings/presets")["presets"]
        self.assertEqual([preset["name"] for preset in listed], TIER_NAMES)

        path = "/api/admin/settings/presets/{name}"
        payload = admin.PresetPayload(values={"embedding_batch_size": 8, "gpu_device": "cuda:1", "huggingface_token": "x"})
        saved = self.call("PUT", path, "lab", payload)
        self.assertEqual(saved["preset"]["values"], {"gpu_device": "cuda:1", "embedding_batch_size": 8})

        for method, args in (("PUT", ("32GB VRAM", admin.PresetPayload(values={"embedding_batch_size": 8}))), ("DELETE", ("24GB VRAM",))):
            with self.assertRaises(HTTPException) as raised:
                self.call(method, path, *args)
            self.assertEqual(raised.exception.status_code, 400)

        self.assertEqual(self.call("DELETE", path, "lab"), {"ok": True})
        with self.assertRaises(HTTPException) as raised:
            self.call("DELETE", path, "lab")
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
