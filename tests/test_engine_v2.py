import unittest

from backend.genesis_dia_server_engine import _format_result, format_diarization_v2


class _Segment:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _Annotation:
    def __init__(self, tracks: list[tuple[float, float, str]]):
        self.tracks = tracks

    def itertracks(self, yield_label: bool = False):
        assert yield_label
        for index, (start, end, speaker) in enumerate(self.tracks):
            yield _Segment(start, end), index, speaker


class _DiarizeOutput:
    def __init__(self):
        self.speaker_diarization = _Annotation(
            [
                (3.0, 4.0, "SPEAKER_00"),
                (0.0, 2.0, "SPEAKER_00"),
                (1.0, 3.0, "SPEAKER_01"),
            ]
        )
        self.exclusive_speaker_diarization = _Annotation(
            [
                (0.0, 1.0, "SPEAKER_00"),
                (1.0, 3.0, "SPEAKER_01"),
                (3.0, 4.0, "SPEAKER_00"),
            ]
        )
        # Native pyannote centroids must never appear in v2 output.
        self.speaker_embeddings = [[0.1] * 256, [0.2] * 256]


class EngineV2FormattingTests(unittest.TestCase):
    def test_formats_chronological_ms_segments_and_overlap_regions(self) -> None:
        result = format_diarization_v2(_DiarizeOutput())

        self.assertEqual(
            result["diarization"],
            [
                {"start_ms": 0, "end_ms": 2000, "speaker_id": "SPEAKER_00"},
                {"start_ms": 1000, "end_ms": 3000, "speaker_id": "SPEAKER_01"},
                {"start_ms": 3000, "end_ms": 4000, "speaker_id": "SPEAKER_00"},
            ],
        )
        self.assertEqual(
            result["exclusive_diarization"],
            [
                {"start_ms": 0, "end_ms": 1000, "speaker_id": "SPEAKER_00"},
                {"start_ms": 1000, "end_ms": 3000, "speaker_id": "SPEAKER_01"},
                {"start_ms": 3000, "end_ms": 4000, "speaker_id": "SPEAKER_00"},
            ],
        )
        self.assertEqual(
            result["overlaps"],
            [{"start_ms": 1000, "end_ms": 2000, "speaker_ids": ["SPEAKER_00", "SPEAKER_01"]}],
        )
        self.assertNotIn("speaker_embeddings", result)

    def test_legacy_mapping_shape_stays_unchanged(self) -> None:
        result = _format_result(_DiarizeOutput())

        self.assertEqual(
            result,
            {
                "SPEAKER_00": [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 4.0}],
                "SPEAKER_01": [{"start": 1.0, "end": 3.0}],
            },
        )

    def test_v2_rejects_pipeline_without_exclusive_output(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Exclusive-Diarization"):
            format_diarization_v2(_Annotation([(0.0, 1.0, "SPEAKER_00")]))


if __name__ == "__main__":
    unittest.main()
