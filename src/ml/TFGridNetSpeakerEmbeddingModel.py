from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import resampy
import torch
import json

from .interfaces import SpeakerEmbeddingModel
from ..models.tfgridnet_enrollment.tfgridnet import EmbedTFGridNet
from ..utils import get_torch_device

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BACKEND_DIR / "src" / "configs" / "tfgridnet_enroll_cipic.json"
DEFAULT_CHECKPOINT_PATH = BACKEND_DIR / "weights" / "tfgridnet_enroll.ckpt"

class TFGridNetSpeakerEmbeddingModel(SpeakerEmbeddingModel):
    """
    Speaker embedding model that uses a pretrained TFGridNet architecture
    to generate enrollment embeddings directly from raw waveforms.

    Implements SpeakerEmbeddingModel, to conform by the contract used in services.
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        config_path: Path | None = None,
        sample_rate: int = 16000,
        device: Optional[str] = None,
    ):

        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self.sample_rate = sample_rate
        self.device = torch.device(device) if device else get_torch_device()

        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Place tfgridnet_enroll.ckpt in backend/weights/."
            )
        
        # Load model config
        with open(self.config_path, "r") as f:
            config = json.load(f)

        model_params = (
            config.get("pl_module_args", {})
            .get("model_params", {})
        )

        self.model = EmbedTFGridNet(**model_params).to(self.device)
        self.model.eval()

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )

        # Strip training prefixes like "model." or "module." if present
        prefixes = ("model.model.", "model.", "module.")
        for pref in prefixes:
            if all(k.startswith(pref) for k in state_dict.keys()):
                state_dict = {k[len(pref):]: v for k, v in state_dict.items()}
                break

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[TfGridNetEnrollmentModel] Missing keys: {missing}")
        if unexpected:
            print(f"[TfGridNetEnrollmentModel] Unexpected keys: {unexpected}")

        print(
            f"[TfGridNetEnrollmentModel] Loaded checkpoint={self.checkpoint_path} "
            f"config={self.config_path} device={self.device}"
        )

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """
        Loads a binaural audio file and resamples to self.sample_rate.

        Returns shape [2, N] channels-first.
        """
        waveform, sr = sf.read(str(audio_path))
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1 or (waveform.ndim == 2 and waveform.shape[1] != 2):
            raise ValueError(
                f"Expected binaural (2-channel) audio, got shape {waveform.shape}: {audio_path}"
            )
        # [N, 2] → [2, N]
        waveform = waveform.T
        if sr != self.sample_rate:
            waveform = resampy.resample(waveform, sr, self.sample_rate, axis=-1)
        return waveform.astype(np.float32)

    def compute_embedding(self, audio_path: Path) -> np.ndarray:
        """
        Computes a speaker embedding from a binaural audio file.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Enrollment audio not found: {audio_path}")

        audio = self._load_audio(audio_path)  # [2, N]

        with torch.no_grad():
            tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # [1, 2, N]
            embedding = self.model(tensor)
            embedding = embedding.squeeze(0).cpu().numpy()

        return embedding.astype(np.float32)

    def compute_embedding_from_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute embedding directly from a binaural numpy array [2, N] (for streaming/similarity checks).
        """
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError(f"Expected binaural audio with shape [2, N], got {audio.shape}")

        audio = audio.astype(np.float32)
        if sample_rate != self.sample_rate:
            audio = resampy.resample(audio, sample_rate, self.sample_rate, axis=-1)

        with torch.no_grad():
            tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # [1, 2, N]
            embedding = self.model(tensor)
            embedding = embedding.squeeze(0).cpu().numpy().astype(np.float32)
        return embedding