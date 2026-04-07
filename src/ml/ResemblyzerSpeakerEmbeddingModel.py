from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav

from .interfaces import SpeakerEmbeddingModel
from ..utils import get_torch_device


class ResemblyzerSpeakerEmbeddingModel(SpeakerEmbeddingModel):
    def __init__(self, device: str | None = None):
        try:
            from df.enhance import enhance, init_df
        except ImportError as exc:
            raise ImportError(
                "DeepFilterNet is required for enrollment denoising. "
                "Install with `pip install deepfilternet`."
            ) from exc

        self._device = device or get_torch_device()
        self._encoder = VoiceEncoder(device=self._device)
        self._df_enhance = enhance
        self._df_model, self._df_state, _ = init_df()

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio.astype(np.float32, copy=False)
        if audio.ndim != 2:
            raise ValueError(f"Expected mono or stereo audio, got shape {audio.shape}")
        # Accept either [N, C] or [C, N].
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            return audio.mean(axis=0).astype(np.float32, copy=False)
        return audio.mean(axis=1).astype(np.float32, copy=False)

    def _enhance_audio(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        mono_audio = self._to_mono(audio)
        df_sample_rate = int(self._df_state.sr())

        if sample_rate != df_sample_rate:
            mono_audio = resampy.resample(mono_audio, sample_rate, df_sample_rate, axis=-1)

        enhanced = self._df_enhance(self._df_model, self._df_state, mono_audio)
        if hasattr(enhanced, "detach"):
            enhanced = enhanced.detach().cpu().numpy()
        return np.asarray(enhanced, dtype=np.float32), df_sample_rate

    def compute_embedding(self, audio_path: Path) -> np.ndarray:
        if not audio_path.exists():
            raise FileNotFoundError(f"Enrollment audio not found: {audio_path}")
        audio, sample_rate = sf.read(str(audio_path), dtype="float32")
        enhanced_audio, enhanced_sr = self._enhance_audio(audio, int(sample_rate))
        waveform = preprocess_wav(enhanced_audio, source_sr=enhanced_sr)
        embedding = self._encoder.embed_utterance(waveform)
        return embedding.astype(np.float32)

    def compute_embedding_from_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute an embedding directly from an in-memory waveform.

        Args:
            audio: Stereo audio [2, N] channels-first. Averaged to mono internally.
            sample_rate: Sample rate of the input audio.
        """
        enhanced_audio, enhanced_sr = self._enhance_audio(audio, sample_rate)
        waveform = preprocess_wav(enhanced_audio, source_sr=enhanced_sr)
        embedding = self._encoder.embed_utterance(waveform)
        return embedding.astype(np.float32)
