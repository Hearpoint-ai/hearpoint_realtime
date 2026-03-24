from pathlib import Path
from typing import Optional

import json
import numpy as np
import resampy
import soundfile as sf
import torch

from .ResemblyzerSpeakerEmbeddingModel import ResemblyzerSpeakerEmbeddingModel
from .interfaces import SpeakerEmbeddingModel
from ..models.tfgridnet_enrollment.tfgridnet import Net as BeamformerNet
from ..utils import get_torch_device

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BACKEND_DIR / "src" / "configs" / "beamformer_tfgridnet.json"
DEFAULT_CHECKPOINT_PATH = BACKEND_DIR / "weights" / "beamformer.ckpt"


class BeamformerResemblyzerSpeakerEmbeddingModel(SpeakerEmbeddingModel):
    """
    Enrollment model that denoises noisy binaural enrollment audio with TFGridNet
    beamforming, then computes a Resemblyzer embedding on the cleaned target signal.
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        config_path: Path | None = None,
        sample_rate: int = 16000,
        device: Optional[str] = None,
    ):
        print("Initializing BeamformerResemblyzerSpeakerEmbeddingModel====a=sdf=asdfas=df=as=")
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self.sample_rate = sample_rate
        self.device = torch.device(device) if device else get_torch_device()

        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Place beamformer.ckpt in weights/."
            )

        with self.config_path.open() as fp:
            config = json.load(fp)
        model_params = config.get("pl_module_args", {}).get("model_params", {})

        self._beamformer = BeamformerNet(**model_params).to(self.device)
        self._beamformer.eval()
        self._load_checkpoint()

        # Resemblyzer runs on the same device family when possible.
        self._resemblyzer = ResemblyzerSpeakerEmbeddingModel(device=self.device.type)

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint

        prefixes = ("model.model.", "model.", "module.")
        for pref in prefixes:
            if all(k.startswith(pref) for k in state_dict.keys()):
                state_dict = {k[len(pref):]: v for k, v in state_dict.items()}
                break

        missing, unexpected = self._beamformer.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[BeamformerResemblyzerSpeakerEmbeddingModel] Missing keys: {missing}")
        if unexpected:
            print(f"[BeamformerResemblyzerSpeakerEmbeddingModel] Unexpected keys: {unexpected}")

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        waveform, sr = sf.read(str(audio_path))
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1 or (waveform.ndim == 2 and waveform.shape[1] != 2):
            raise ValueError(
                f"Expected binaural (2-channel) audio, got shape {waveform.shape}: {audio_path}"
            )

        # [N, 2] -> [2, N]
        audio = waveform.T
        if sr != self.sample_rate:
            audio = resampy.resample(audio, sr, self.sample_rate, axis=-1)
        return audio.astype(np.float32, copy=False)

    def _beamform_target(self, audio_2xN: np.ndarray) -> np.ndarray:
        if audio_2xN.ndim != 2 or audio_2xN.shape[0] != 2:
            raise ValueError(f"Expected binaural audio with shape [2, N], got {audio_2xN.shape}")

        with torch.no_grad():
            mixture = torch.from_numpy(audio_2xN.astype(np.float32, copy=False)).unsqueeze(0).to(self.device)
            separated = self._beamformer(mixture)  # [B, n_srcs, T]
            est_target = separated[:, :1, :]  # [B, 1, T]
            cleaned = est_target.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        return cleaned

    def compute_embedding(self, audio_path: Path) -> np.ndarray:
        if not audio_path.exists():
            raise FileNotFoundError(f"Enrollment audio not found: {audio_path}")
        audio_2xN = self._load_audio(audio_path)
        cleaned_target = self._beamform_target(audio_2xN)
        embedding = self._resemblyzer.compute_embedding_from_array(
            cleaned_target[np.newaxis, :], self.sample_rate
        )
        return embedding.astype(np.float32)

    def compute_embedding_from_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError(f"Expected binaural audio with shape [2, N], got {audio.shape}")
        audio_2xN = audio.astype(np.float32, copy=False)
        if sample_rate != self.sample_rate:
            audio_2xN = resampy.resample(audio_2xN, sample_rate, self.sample_rate, axis=-1)
        cleaned_target = self._beamform_target(audio_2xN)
        embedding = self._resemblyzer.compute_embedding_from_array(
            cleaned_target[np.newaxis, :], self.sample_rate
        )
        return embedding.astype(np.float32)
