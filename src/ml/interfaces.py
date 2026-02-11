from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np


class SpeakerEmbeddingModel(ABC):
    @abstractmethod
    def compute_embedding(self, audio_path: Path) -> np.ndarray:
        ...

    def compute_embedding_from_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Optional fast path to compute an embedding directly from an in-memory waveform.
        Defaults to writing to disk via compute_embedding when not overridden.
        """
        import soundfile as sf  # Lazy import to avoid adding a hard dependency for mocks.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            return self.compute_embedding(Path(tmp.name))


class TargetSpeechExtractionModel(ABC):
    @abstractmethod
    def separate(
        self,
        mixture_audio_path: Path,
        speaker_embedding_paths: List[Path],
        output_dir: Path,
        output_name_prefix: str,
    ) -> List[Path]:
        ...
