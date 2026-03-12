import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.realtime.metrics import (
    _ensure_stereo,
    _evaluate_thresholds,
    _write_report,
)


class MetricsTests(unittest.TestCase):
    def test_ensure_stereo_from_mono_inputs(self):
        mono_1d = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        out_1d = _ensure_stereo(mono_1d)
        self.assertEqual(out_1d.shape, (3, 2))
        self.assertTrue(np.allclose(out_1d[:, 0], out_1d[:, 1]))

        mono_2d = mono_1d[:, np.newaxis]
        out_2d = _ensure_stereo(mono_2d)
        self.assertEqual(out_2d.shape, (3, 2))
        self.assertTrue(np.allclose(out_2d[:, 0], out_2d[:, 1]))

    def test_evaluate_thresholds_excludes_file_mode_stability_metrics(self):
        stats = {
            "drops_input": 10,
            "drops_output": 5,
            "underruns": 1,
            "rtf_avg": 0.7,
        }
        thresholds = {
            "drops_input": 0,
            "drops_output": 0,
            "underruns": 0,
            "rtf_avg": 1.0,
        }
        failed = _evaluate_thresholds(stats, thresholds, mode="file")
        self.assertEqual(failed, [])

    def test_write_report_preserves_stats_and_adds_status(self):
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "report.json"
            stats = {
                "mode": "file",
                "sample_rate": 16000,
                "chunk_size": 128,
                "rtf_avg": 0.6,
                "latency_ms_avg": 4.1,
                "latency_ms_p50": 4.0,
                "latency_ms_p95": 5.1,
                "latency_ms_p99": 5.5,
                "latency_ms_max": 6.0,
                "drops_input": 0,
                "drops_output": 0,
                "underruns": 0,
                "nan_count": 0,
                "clip_ratio": 0.0,
                "rms_in": 0.1,
                "rms_out": 0.1,
                "cosine_similarity_before": 0.1,
                "cosine_similarity_after": 0.2,
                "cosine_similarity_delta": 0.1,
            }
            _write_report(stats, report_path, failed_thresholds=[])
            data = json.loads(report_path.read_text())

            for key, value in stats.items():
                self.assertIn(key, data)
                self.assertEqual(data[key], value)
            self.assertEqual(data["status"], "pass")
            self.assertEqual(data["failed_thresholds"], [])


if __name__ == "__main__":
    unittest.main()
