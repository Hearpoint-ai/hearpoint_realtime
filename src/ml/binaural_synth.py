"""
Binaural synthesis helpers for TF-GridNet inference.

Given binaural enrollment audio and optional mono interferers, synthesize a
two-channel binaural mixture. Interferers are spatialized using CIPIC HRIRs.
Enrollment audio is already binaural (recorded with binaural mic) and used directly.

All operations are deterministic given an np.random.Generator.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import resampy
import soundfile as sf
import torch
import torch.nn.functional as F


@dataclass
class HrtfEntry:
    az_idx: int
    el_idx: int
    az_deg: float
    el_deg: float
    left: np.ndarray  # [T_hrtf]
    right: np.ndarray  # [T_hrtf]


class CipicHrirSet:
    """
    Loads a CIPIC HRIR grid and lets you sample directions.

    Supports either:
    - A CIPIC `hrir_final.mat` file (preferred).
    - A directory containing per-direction stereo WAV/NPY files named
      `az{az_idx}_el{el_idx}_{left|right}.{wav|npy}`.
    """

    def __init__(self, entries: list[HrtfEntry], sample_rate: int):
        if not entries:
            raise ValueError("No HRTF entries provided.")
        self.entries = entries
        self.sample_rate = sample_rate

    @classmethod
    def from_sofa(cls, sofa_path: Path, target_sr: int) -> "CipicHrirSet":
        """
        Load a CIPIC SOFA file (subject_*.sofa).
        Requires pysofaconventions (present in training env).
        """
        try:
            import pysofaconventions
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError("pysofaconventions is required to load SOFA files") from exc

        sofa = pysofaconventions.SOFAFile(str(sofa_path), "r")
        ir = sofa.getDataIR()  # shape (M, R=2, E)
        src_pos = sofa.getVariableValue("SourcePosition")  # shape (M, 3)
        ir = np.asarray(ir)
        src_pos = np.asarray(src_pos)
        if np.ma.isMaskedArray(ir):
            ir = ir.filled(0.0)
        if np.ma.isMaskedArray(src_pos):
            src_pos = src_pos.filled(0.0)
        # Units are typically degrees for CIPIC (spherical coordinates).
        sr = int(sofa.getSamplingRate())

        entries: list[HrtfEntry] = []
        for idx in range(ir.shape[0]):
            left = ir[idx, 0].astype(np.float32)
            right = ir[idx, 1].astype(np.float32)
            az_deg = float(src_pos[idx, 0])
            el_deg = float(src_pos[idx, 1])
            if target_sr and target_sr != sr:
                left = resampy.resample(left, sr, target_sr)
                right = resampy.resample(right, sr, target_sr)
            entries.append(
                HrtfEntry(
                    az_idx=idx,
                    el_idx=idx,  # SOFA does not expose discrete grid indices; reuse idx.
                    az_deg=az_deg,
                    el_deg=el_deg,
                    left=left,
                    right=right,
                )
            )
        return cls(entries=entries, sample_rate=target_sr)

    @classmethod
    def from_mat(cls, mat_path: Path, target_sr: int) -> "CipicHrirSet":
        import scipy.io  # Lazy import; scipy is available in the env.

        mat = scipy.io.loadmat(mat_path)
        hrir_l = mat["hrir_l"]  # [n_elev, n_az, taps]
        hrir_r = mat["hrir_r"]
        n_el, n_az, _ = hrir_l.shape
        # Attempt to read angles; if absent, fall back to evenly spaced grids.
        az_vals = None
        el_vals = None
        for key in ("azim_v", "azim", "az_v"):
            if key in mat:
                az_vals = np.squeeze(mat[key])
                break
        for key in ("elev_v", "elev", "el_v"):
            if key in mat:
                el_vals = np.squeeze(mat[key])
                break
        if az_vals is None:
            az_vals = np.linspace(-80, 80, n_az)
        if el_vals is None:
            el_vals = np.linspace(-45, 230, n_el)

        entries: list[HrtfEntry] = []
        for el_idx in range(n_el):
            for az_idx in range(n_az):
                left = hrir_l[el_idx, az_idx].astype(np.float32)
                right = hrir_r[el_idx, az_idx].astype(np.float32)
                if target_sr and target_sr != 44100:
                    left = resampy.resample(left, 44100, target_sr)
                    right = resampy.resample(right, 44100, target_sr)
                entries.append(
                    HrtfEntry(
                        az_idx=az_idx,
                        el_idx=el_idx,
                        az_deg=float(az_vals[az_idx]) if az_idx < len(az_vals) else float(az_idx),
                        el_deg=float(el_vals[el_idx]) if el_idx < len(el_vals) else float(el_idx),
                        left=left,
                        right=right,
                    )
                )
        return cls(entries=entries, sample_rate=target_sr)

    @classmethod
    def from_dir(cls, hrtf_dir: Path, sample_rate: int) -> "CipicHrirSet":
        """
        Expect files like az{az}_el{el}_left.wav/right.wav or .npy.
        """
        entries: list[HrtfEntry] = []
        for side in ("left", "right"):
            pass  # placeholder to enforce structure

        for path in hrtf_dir.glob("az*_el*_*.*"):
            name = path.stem  # e.g., az10_el12_left
            parts = name.split("_")
            if len(parts) < 3:
                continue
            try:
                az_idx = int(parts[0].replace("az", ""))
                el_idx = int(parts[1].replace("el", ""))
            except ValueError:
                continue
            # Ensure we only load once per direction when we hit "_left".
            if not name.endswith("left"):
                continue
            left = _load_hrtf_file(path, sample_rate)
            right_path = path.with_name(name.replace("left", "right") + path.suffix)
            if not right_path.exists():
                continue
            right = _load_hrtf_file(right_path, sample_rate)
            entries.append(
                HrtfEntry(
                    az_idx=az_idx,
                    el_idx=el_idx,
                    az_deg=float(az_idx),
                    el_deg=float(el_idx),
                    left=left,
                    right=right,
                )
            )
        return cls(entries=entries, sample_rate=sample_rate)

    def sample(self, rng: np.random.Generator) -> HrtfEntry:
        idx = rng.integers(0, len(self.entries))
        return self.entries[int(idx)]


def _load_hrtf_file(path: Path, target_sr: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        return arr
    wav, sr = sf.read(str(path))
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != target_sr:
        wav = resampy.resample(wav, sr, target_sr)
    return wav


def _to_numpy(audio: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert to numpy, preserving shape (no squeeze that could collapse channels)."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def _ensure_length(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        return x[..., :target_len]
    pad_len = target_len - x.shape[-1]
    return np.pad(x, (0, pad_len), mode="constant")


def _convolve_mono_hrir(mono: np.ndarray, hrtf: HrtfEntry) -> np.ndarray:
    left = np.convolve(mono, hrtf.left, mode="full")
    right = np.convolve(mono, hrtf.right, mode="full")
    min_len = min(left.shape[-1], right.shape[-1])
    left = left[:min_len]
    right = right[:min_len]
    return np.stack([left, right], axis=0).astype(np.float32)


def _load_random_clip(file_list: Sequence[Path], target_len: int, target_sr: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    if not file_list:
        return None
    path = file_list[int(rng.integers(0, len(file_list)))]
    audio, sr = sf.read(str(path))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
    audio = _ensure_length(audio, target_len)
    return audio


def _generate_noise(kind: str, length: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "white":
        return rng.standard_normal(length).astype(np.float32)
    if kind == "pink":
        # Voss-McCartney: sum of octaves
        n_rows = 16
        array = rng.standard_normal((n_rows, length)).astype(np.float32)
        array = np.cumsum(array, axis=1)
        weights = 1 / (np.arange(1, n_rows + 1))
        pink = (weights[:, None] * array).sum(axis=0)
        pink /= np.max(np.abs(pink) + 1e-8)
        return pink.astype(np.float32)
    raise ValueError(f"Unsupported noise kind: {kind}")


def synthesize_binaural_mixture(
    enrollment_binaural: np.ndarray | torch.Tensor,
    sample_rate: int,
    hrtf_set: CipicHrirSet,
    interferer_files: Sequence[Path],
    num_interferers: int = 1,
    num_speech_interferers: Optional[int] = None,
    speech_interferer_files: Optional[Sequence[Path]] = None,
    add_noise: bool = False,
    rng: Optional[np.random.Generator] = None,
    require_speech_first: bool = False,
) -> dict:
    """
    Build a binaural mixture compatible with TF-GridNet.

    Args:
        enrollment_binaural: Target speaker audio, already binaural [2, T].
        sample_rate: Target sample rate (Hz).
        hrtf_set: CIPIC HRIR set to sample directions from (used for interferers).
        interferer_files: Paths to background noise clips.
        num_interferers: How many interfering sources to add.
        speech_interferer_files: Optional pool of speech clips for interferers.
        add_noise: If True, add low-level pink + white noise.
        rng: Optional np.random.Generator for determinism.
    Returns:
        {
            "binaural_mixture": np.ndarray [2, T],
            "binaural_target": np.ndarray [2, T],
            "direction_interferers_deg": [(az, el), ...],
            "hrtf_indices": {"interferers": [(az_idx, el_idx), ...]},
        }
    """
    rng = rng or np.random.default_rng()
    # Enrollment is already binaural [2, T] — use directly as the target.
    binaural_target = _to_numpy(enrollment_binaural)
    if binaural_target.ndim != 2 or binaural_target.shape[0] != 2:
        raise ValueError(f"Expected binaural enrollment with shape [2, T], got {binaural_target.shape}")

    target_len = binaural_target.shape[-1]
    interferer_dirs: list[tuple[float, float]] = []
    interferer_idx: list[tuple[int, int]] = []
    binaural_interferers: list[np.ndarray] = []

    # Decide how many speech vs noise interferers.
    speech_count = num_speech_interferers if num_speech_interferers is not None else 0
    if speech_interferer_files:
        if num_speech_interferers is None:
            speech_count = 1 if require_speech_first else 0
        speech_count = min(speech_count, num_interferers)
    else:
        speech_count = 0
    noise_count = max(0, num_interferers - speech_count)

    for i in range(num_interferers):
        hrir = hrtf_set.sample(rng)
        interferer_idx.append((hrir.az_idx, hrir.el_idx))
        interferer_dirs.append((hrir.az_deg, hrir.el_deg))
        # Choose source audio (mono interferers get HRTF-convolved to binaural).
        src = None
        if speech_count > 0:
            src = _load_random_clip(speech_interferer_files or (), target_len, sample_rate, rng)
            speech_count -= 1
        elif noise_count > 0 and interferer_files:
            src = _load_random_clip(interferer_files, target_len, sample_rate, rng)
            noise_count -= 1
        if src is None:
            src = np.zeros(target_len, dtype=np.float32)
        binaural_interferers.append(_convolve_mono_hrir(src, hrir))

    mixture = binaural_target.copy()
    for inter in binaural_interferers:
        inter = _ensure_length(inter, mixture.shape[-1])
        mixture = _ensure_length(mixture, inter.shape[-1])
        mixture[:, : inter.shape[-1]] += inter

    if add_noise:
        noise_len = mixture.shape[-1]
        pink = _generate_noise("pink", noise_len, rng) * 0.01
        white = _generate_noise("white", noise_len, rng) * 0.005
        mixture[0] += pink
        mixture[1] += white

    peak = np.max(np.abs(mixture)) + 1e-8
    mixture = (mixture / peak).astype(np.float32)
    binaural_target = _ensure_length(binaural_target, mixture.shape[-1]).astype(np.float32)

    return {
        "binaural_mixture": mixture,
        "binaural_target": binaural_target,
        "direction_interferers_deg": interferer_dirs,
        "hrtf_indices": {
            "interferers": interferer_idx,
        },
    }
