import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import resampy
import soundfile as sf
import torch

from .interfaces import TargetSpeechExtractionModel
from ..models.tfgridnet_realtime.net import Net
from ..utils import get_torch_device

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BACKEND_DIR / "src" / "configs" / "tfgridnet_cipic.json"
DEFAULT_CHECKPOINT_PATH = BACKEND_DIR / "weights" / "tfgridnet.ckpt"


class TFGridNetExtractionModel(TargetSpeechExtractionModel):
    def __init__(
        self,
        checkpoint_path: Path | None = None,
        config_path: Path | None = None,
        sample_rate: int = 16000,
        device: str | None = None,
        hrtf_left_path: Path | None = None,
        hrtf_right_path: Path | None = None,
    ):
        self.sample_rate = sample_rate
        self.device = torch.device(device) if device else get_torch_device()
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self.hrtf_left, self.hrtf_right = None, None

        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Place tfgridnet.ckpt in backend/weights/."
            )

        if hrtf_left_path and hrtf_right_path:
            left, left_sr = self._load_hrtf(hrtf_left_path)
            right, right_sr = self._load_hrtf(hrtf_right_path)
            self.hrtf_left = self._resample_hrtf(left, left_sr)
            self.hrtf_right = self._resample_hrtf(right, right_sr)

        with self.config_path.open() as fp:
            config = json.load(fp)
        model_params = config.get("pl_module_args", {}).get("model_params", {})
        self.model = Net(**model_params).to(self.device)
        self._load_checkpoint()
        logging.info(
            "Loaded TFGridNetExtractionModel from %s with config %s on device %s",
            self.checkpoint_path,
            self.config_path,
            self.device,
        )
        # Explicit print to surface in uvicorn/stdout even if logging handlers are not configured.
        print(
            f"[TFGridNetExtractionModel] Loaded checkpoint={self.checkpoint_path} "
            f"config={self.config_path} device={self.device}"
        )
        self.model.eval()

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint

        # Strip common wrappers like "model." or "module." to align with the bare Net keys.
        prefixes = ("model.model.", "model.", "module.")
        for pref in prefixes:
            if all(k.startswith(pref) for k in state_dict.keys()):
                state_dict = {k[len(pref):]: v for k, v in state_dict.items()}
                break

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logging.warning("Missing TFGridNet keys when loading checkpoint: %s", missing)
            print(f"[TFGridNetExtractionModel] Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logging.warning("Ignoring unexpected TFGridNet checkpoint keys: %s", unexpected)
            print(f"[TFGridNetExtractionModel] Ignoring unexpected keys when loading checkpoint: {unexpected}")

    def _load_hrtf(self, path: Path) -> tuple[np.ndarray, int]:
        """
        Loads an HRTF impulse response from .wav or .npy and returns (array, sample_rate).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HRTF file not found: {path}")
        if path.suffix.lower() == ".npy":
            data = np.load(path)
            sr = self.sample_rate  # Assume current model rate when sr is unknown.
        else:
            data, sr = sf.read(str(path))
        if data.ndim > 1:
            data = data[:, 0]
        return np.asarray(data, dtype=np.float32), int(sr)

    def _resample_hrtf(self, hrtf: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.sample_rate:
            return hrtf
        return resampy.resample(hrtf, sr, self.sample_rate)

    def _prepare_mixture(self, mixture: np.ndarray, sr: int) -> np.ndarray:
        """
        Returns binaural audio shaped as [2, samples] at self.sample_rate.
        """
        if mixture.ndim == 1:
            raise ValueError("Expected binaural (2-channel) mixture audio, got mono.")
        # soundfile returns [samples, channels]; transpose to [channels, samples]
        mixture = mixture.T
        if mixture.shape[0] != 2:
            raise ValueError(f"Expected 2-channel binaural mixture, got {mixture.shape[0]} channels.")

        if sr != self.sample_rate:
            mixture = resampy.resample(mixture, sr, self.sample_rate, axis=-1)

        return mixture.astype(np.float32, copy=False)

    def _load_embedding(self, embedding_path: Path) -> torch.Tensor:
        if not embedding_path.exists():
            raise FileNotFoundError(f"Speaker embedding not found: {embedding_path}")
        embedding = np.load(embedding_path)
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, 1, -1)
        return torch.from_numpy(embedding).to(self.device)

    def separate(
        self,
        mixture_audio_path: Path,
        speaker_embedding_paths: List[Path],
        output_dir: Path,
        output_name_prefix: str,
    ) -> List[Path]:
        if not speaker_embedding_paths:
            raise ValueError("At least one speaker embedding is required for extraction.")
        if not mixture_audio_path.exists():
            raise FileNotFoundError(f"Mixture audio not found: {mixture_audio_path}")

        audio, sr = sf.read(str(mixture_audio_path))
        prepared = self._prepare_mixture(audio, sr)
        mixture_tensor = torch.from_numpy(prepared).unsqueeze(0).to(self.device)

        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: List[Path] = []

        for index, embedding_path in enumerate(speaker_embedding_paths):
            embedding_tensor = self._load_embedding(embedding_path)
            with torch.no_grad():
                enhanced = self.model(mixture_tensor, embedding_tensor)
            enhanced_audio = enhanced.squeeze(0).cpu().numpy()  # [channels, samples]
            enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
            enhanced_audio = enhanced_audio.T  # [samples, channels]
            output_path = output_dir / f"{output_name_prefix}_{index}.wav"
            sf.write(str(output_path), enhanced_audio, self.sample_rate)
            outputs.append(output_path)

        return outputs
