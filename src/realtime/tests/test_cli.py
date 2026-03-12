import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.realtime import cli


class CliTests(unittest.TestCase):
    def _run_main(self, argv: list[str]):
        with patch.object(sys, "argv", ["realtime_inference.py", *argv]):
            cli.main()

    def test_dispatches_to_file_mode_tester(self):
        fake_tester = MagicMock()
        fake_tester.process_file.return_value = {
            "mode": "file",
            "rtf_avg": 0.5,
            "drops_input": 0,
            "drops_output": 0,
            "underruns": 0,
        }

        with (
            patch("src.realtime.cli.multiprocessing.set_start_method"),
            patch("src.realtime.cli.DEFAULT_YAML_CONFIG_PATH", Path("/__missing__/config.yaml")),
            patch("src.realtime.cli.FileBasedTest", return_value=fake_tester) as file_cls,
            patch("src.realtime.cli.RealtimeInference") as live_cls,
        ):
            self._run_main([
                "--test-file",
                "media/si_sdr_fixture/mixture.wav",
                "--embedding",
                "media/si_sdr_fixture/enrollment.npy",
            ])

        file_cls.assert_called_once()
        live_cls.assert_not_called()
        fake_tester.process_file.assert_called_once()

    def test_exits_non_zero_when_thresholds_fail(self):
        fake_tester = MagicMock()
        fake_tester.process_file.return_value = {
            "mode": "file",
            "rtf_avg": 2.0,
            "drops_input": 0,
            "drops_output": 0,
            "underruns": 0,
        }

        with (
            patch("src.realtime.cli.multiprocessing.set_start_method"),
            patch("src.realtime.cli.DEFAULT_YAML_CONFIG_PATH", Path("/__missing__/config.yaml")),
            patch("src.realtime.cli.FileBasedTest", return_value=fake_tester),
            patch("src.realtime.cli._load_threshold_profile", return_value={"rtf_avg": 1.0}),
            patch("src.realtime.cli._evaluate_thresholds", return_value=["rtf_avg"]),
        ):
            with self.assertRaises(SystemExit) as cm:
                self._run_main([
                    "--test-file",
                    "media/si_sdr_fixture/mixture.wav",
                    "--embedding",
                    "media/si_sdr_fixture/enrollment.npy",
                    "--threshold-profile",
                    "dev",
                ])

        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
