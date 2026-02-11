##################################################
#### This file is simply a cleaned up version of src/ml/TFGridNetSpeakerEmbeddingModel.py
#### For some reason (likely the mono, binaural randomness), this cleaned up version works worse than the original #####
#### The original has a lot of random binaural --> mono --> binaural arbitrary conversions that I will prob need to spend some more time on later 
##################################################

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
DEFAULT_CONFIG_PATH = BACKEND_DIR / "configs" / "tfgridnet_enroll_cipic.json"
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
        Loads an audio file and ensures correct sample rate.
        """
        waveform, sr = sf.read(str(audio_path))
        waveform = np.asarray(waveform, dtype=np.float32)

        if sr != self.sample_rate:
            waveform = resampy.resample(waveform.T, sr, self.sample_rate).T
        return waveform

    ### IMPORTANT: TfGridNet Enrollment network expects binaural audio
    def _ensure_channel_layout(self, audio: np.ndarray) -> np.ndarray:
        """
        Ensure consistent [2, N] channel layout for TFGridNet,
        duplicating mono if needed and averaging down if more than 2 channels
        """
        # convert mono -> binaural
        if audio.ndim == 1:
            return np.stack([audio, audio], axis=0)  # [2, N] TEMPORARLIY DUPLICATING MONO AUDIO TO STERO
        
        # Handle 2D cases
        if audio.ndim == 2:
            # If [2, N] -> [N, 2]
            if audio.shape[1] == 2:
                return audio.T
            # If already [2, N], return as is
            if audio.shape[0] == 2:
                return audio
            # More than 2 channels -> average down to stereo
            return np.stack([audio.mean(axis=0), audio.mean(axis=0)], axis=0)
        raise RuntimeError(f"Unsupported audio shape: {audio.shape}")

    def compute_embedding(self, audio_path: Path) -> np.ndarray:
        """
        Computes a speaker embedding for an audio file.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Enrollment audio not found: {audio_path}")

        audio = self._load_audio(audio_path)
        audio = self._ensure_channel_layout(audio)

        with torch.no_grad():
            tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # [1, 2, N]
            embedding = self.model(tensor)
            embedding = embedding.squeeze(0).cpu().numpy()

        # Ensure consistent float type
        embedding = embedding.astype(np.float32)

        return embedding

    def compute_embedding_from_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute embedding directly from a numpy array (for streaming/similarity checks).
        """

        if sample_rate != self.sample_rate:
            audio = resampy.resample(audio.T, sample_rate, self.sample_rate).T
        audio = np.asarray(audio, dtype=np.float32)
        audio = self._ensure_channel_layout(audio)

        with torch.no_grad():
            tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # [1, 2, N]
            embedding = self.model(tensor)
            embedding = embedding.squeeze(0).cpu().numpy().astype(np.float32)
        return embedding