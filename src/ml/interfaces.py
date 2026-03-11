from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np


class SpeakerEmbeddingModel(ABC):
    @abstractmethod
    def compute_embedding(self, audio_path: Path) -> np.ndarray:
        ...

    def compute_embedding_from_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute embedding from an in-memory stereo waveform.

        Default implementation writes to a temp file and delegates to
        ``compute_embedding``.  Override for a faster in-memory path.

        Args:
            audio: Stereo audio, channels-first ``[2, N]``.
            sample_rate: Sample rate of the input audio.
        """
        import soundfile as sf  # Lazy import to avoid adding a hard dependency for mocks.
        import tempfile

        # Contract: audio is [2, N] channels-first; soundfile expects [N, C]
        wav = audio.T if audio.ndim == 2 else audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, wav, sample_rate)
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
