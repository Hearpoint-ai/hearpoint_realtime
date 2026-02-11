import numpy as np
from pathlib import Path

from .interfaces import SpeakerEmbeddingModel


class MockSpeakerEmbeddingModel(SpeakerEmbeddingModel):
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim

    def compute_embedding(self, audio_path: Path) -> np.ndarray:
        seed = sum(audio_path.name.encode("utf-8"))
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.embedding_dim).astype(np.float32)

    def compute_embedding_from_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG002
        # Deterministic pseudo-embedding based on audio content to keep tests repeatable.
        seed = int(np.sum(np.abs(audio)) * 1e6) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.embedding_dim).astype(np.float32)
