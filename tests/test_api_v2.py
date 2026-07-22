import asyncio
import unittest
from unittest import mock

from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from backend import genesis_dia_server_api as api
from backend.genesis_dia_server_api import _build_v2_response, _validate_v2_speaker_counts, create_api


class ApiV2ContractTests(unittest.TestCase):
    def test_v2_routes_are_additive(self) -> None:
        app = create_api(FastAPI())
        routes = {(method, route.path) for route in app.routes for method in getattr(route, "methods", set())}

        self.assertIn(("POST", "/diarize/"), routes)
        self.assertIn(("POST", "/v2/diarize"), routes)
        self.assertIn(("GET", "/v2/capabilities"), routes)

    def test_capabilities_uses_public_api_key_gate(self) -> None:
        app = create_api(FastAPI())
        endpoint = next(route.endpoint for route in app.routes if route.path == "/v2/capabilities")
        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/v2/capabilities",
                "raw_path": b"/v2/capabilities",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 50000),
                "server": ("127.0.0.1", 7864),
                "root_path": "",
                "app": app,
            }
        )

        with mock.patch.object(api, "authorize_api_key", side_effect=HTTPException(status_code=401)) as gate:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(endpoint(request))

        gate.assert_called_once_with(request)
        self.assertEqual(raised.exception.status_code, 401)

        with mock.patch.object(api, "authorize_api_key", return_value="key_test"), mock.patch.dict(
            api.current_settings,
            {"diarization_model_id": "pyannote/speaker-diarization-community-1"},
            clear=True,
        ), mock.patch.object(api.torch.cuda, "is_available", return_value=True):
            response = asyncio.run(endpoint(request))

        self.assertEqual(response["api_version"], "2.0")
        self.assertTrue(response["exclusive_diarization"])
        self.assertTrue(response["overlap_regions"])
        self.assertFalse(response["native_speaker_embeddings"])
        self.assertEqual(response["model"]["device"], "cuda")

    def test_v2_response_contract_and_counts(self) -> None:
        result = {
            "diarization": [
                {"start_ms": 0, "end_ms": 2000, "speaker_id": "SPEAKER_00"},
                {"start_ms": 1000, "end_ms": 3000, "speaker_id": "SPEAKER_01"},
            ],
            "exclusive_diarization": [
                {"start_ms": 0, "end_ms": 1000, "speaker_id": "SPEAKER_00"},
                {"start_ms": 1000, "end_ms": 3000, "speaker_id": "SPEAKER_01"},
            ],
            "overlaps": [
                {"start_ms": 1000, "end_ms": 2000, "speaker_ids": ["SPEAKER_00", "SPEAKER_01"]}
            ],
        }
        metadata = {
            "request_id": "req123",
            "model_id": "pyannote/speaker-diarization-community-1",
            "duration_ms": 3000,
            "total_duration_ms": 800,
            "speakers_found": 2,
        }

        response = _build_v2_response(result, metadata, 2, None, None)

        self.assertEqual(response["schema_version"], "2.0")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["input"]["num_speakers"], 2)
        self.assertEqual(
            response["counts"],
            {"speakers": 2, "diarization_segments": 2, "exclusive_segments": 2, "overlaps": 1},
        )
        self.assertNotIn("speaker_embeddings", response)

    def test_v2_validates_speaker_bounds(self) -> None:
        _validate_v2_speaker_counts(None, 2, 5)

        invalid = [(0, None, None), (65, None, None), (None, 5, 2), (2, 2, None)]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(HTTPException) as raised:
                _validate_v2_speaker_counts(*args)
            self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
