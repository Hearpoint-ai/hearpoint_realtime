import subprocess
import sys
import unittest

import src.realtime.realtime_inference as ri


class CompatibilityTests(unittest.TestCase):
    def test_legacy_import_surface(self):
        for name in ["Config", "RealtimeInference", "FileBasedTest", "_ensure_stereo", "main"]:
            self.assertTrue(hasattr(ri, name), f"Missing symbol: {name}")

    def test_entrypoint_help_still_works(self):
        proc = subprocess.run(
            [sys.executable, "src/realtime/realtime_inference.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--embedding", proc.stdout)
        self.assertIn("--test-file", proc.stdout)


if __name__ == "__main__":
    unittest.main()
