from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

from .interfaces import SpeakerEmbeddingModel
from ..utils import get_torch_device


class ResemblyzerSpeakerEmbeddingModel(SpeakerEmbeddingModel):
    def __init__(self, device: str | None = None):
        self._device = device or get_torch_device()
        self._encoder = VoiceEncoder(device=self._device)

    def compute_embedding(self, audio_path: Path) -> np.ndarray:
        if not audio_path.exists():
            raise FileNotFoundError(f"Enrollment audio not found: {audio_path}")
        waveform = preprocess_wav(str(audio_path))
        embedding = self._encoder.embed_utterance(waveform)
        return embedding.astype(np.float32)

    def compute_embedding_from_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute an embedding directly from an in-memory waveform.
        """
        waveform = preprocess_wav(audio, source_sr=sample_rate)
        embedding = self._encoder.embed_utterance(waveform)
        return embedding.astype(np.float32)
