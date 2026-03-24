import unittest
from pathlib import Path

from src.realtime.config import Config, REPO_ROOT


class ConfigParsingTests(unittest.TestCase):
    def test_invalid_embedding_model_raises(self):
        with self.assertRaises(ValueError):
            Config._from_dict({"embedding_model": "unknown-model"})

    def test_relative_paths_resolve_to_repo_root(self):
        cfg = Config._from_dict(
            {
                "embedding": "media/enrollments/test.npy",
                "audio": {"output_gain": 2.5},
                "name_detection": {"model_path": "src/models/vosk-model-small-en-us-0.15"},
            }
        )

        self.assertEqual(cfg.audio.output_gain, 2.5)
        self.assertEqual(cfg.model.embedding, REPO_ROOT / "media/enrollments/test.npy")
        self.assertEqual(
            cfg.name_detection.model_path,
            REPO_ROOT / "src/models/vosk-model-small-en-us-0.15",
        )

    def test_defaults_are_applied(self):
        cfg = Config._from_dict({})
        self.assertEqual(cfg.model.embedding_model, "resemblyzer")
        self.assertEqual(cfg.audio.output_gain, 1.0)
        self.assertEqual(
            cfg.name_detection.model_path,
            REPO_ROOT / "src/models/vosk-model-small-en-us-0.15",
        )
        self.assertIsNone(cfg.model.embedding)
        self.assertIsInstance(cfg.get_checkpoint_path(), Path)
        self.assertFalse(cfg.spectral_subtraction.enabled)


if __name__ == "__main__":
    unittest.main()
