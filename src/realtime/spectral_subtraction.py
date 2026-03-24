"""
Spectral subtraction for stationary noise using STFT magnitude subtraction.

Noise profile: mean |STFT| across time frames of a noise-only recording.
Real-time path: overlap-add streaming with scipy.signal.ShortTimeFFT.

Optional parameters (reduce musical noise / over-suppression):
  alpha — over-subtraction factor (>1 subtracts more noise; can increase musical noise).
  beta  — spectral floor as a fraction of |N|; leaves residual noise but reduces warbling.
  min_magnitude_ratio — never output less than this fraction of the input magnitude per bin
    (prevents silence when |N| was over-estimated vs speech).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from scipy.signal import ShortTimeFFT, get_window


@dataclass(frozen=True)
class NoiseProfileMeta:
    """Metadata stored alongside .npy magnitude vectors for sanity checks."""

    sample_rate: int
    n_fft: int
    hop_length: int
    win_length: int


def _build_sft(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> ShortTimeFFT:
    if win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")
    win = get_window("hann", win_length, fftbins=True).astype(np.float64)
    # Unscaled complex STFT so |Z| matches classical magnitude for subtraction.
    return ShortTimeFFT(win, hop_length, sample_rate, mfft=n_fft, scale_to=None)


def estimate_noise_magnitude_spectrum(
    noise_mono: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    """
    Average noise magnitude spectrum over STFT frames (noise-only segment).

    Args:
        noise_mono: 1D float audio, no speech.
        sample_rate: Hz.
        n_fft: FFT size (zero-pad window to this length).
        hop_length: STFT hop in samples.
        win_length: Analysis window length (Hann).

    Returns:
        1D array, shape (n_fft // 2 + 1,), mean |STFT| per bin.
    """
    x = np.asarray(noise_mono, dtype=np.float64).ravel()
    if x.size < win_length:
        raise ValueError(
            f"Noise sample too short: need at least win_length={win_length} samples, got {x.size}"
        )
    sft = _build_sft(sample_rate, n_fft, hop_length, win_length)
    Zxx = sft.stft(x)
    # Zxx shape: (freq_bins, n_frames)
    mag = np.abs(Zxx)
    return np.mean(mag, axis=1, dtype=np.float64)


def save_noise_profile(
    path: Path | str,
    magnitude: np.ndarray,
    meta: NoiseProfileMeta,
) -> None:
    """Save mean magnitude (.npy) and a small JSON sidecar with STFT parameters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, magnitude.astype(np.float32))
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "sample_rate": meta.sample_rate,
                "n_fft": meta.n_fft,
                "hop_length": meta.hop_length,
                "win_length": meta.win_length,
            },
            indent=2,
        )
    )


def load_noise_profile(path: Path | str) -> tuple[np.ndarray, NoiseProfileMeta | None]:
    """Load .npy magnitude; optional .meta.json sidecar for STFT parameters."""
    path = Path(path)
    mag = np.load(path)
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if sidecar.is_file():
        raw = json.loads(sidecar.read_text())
        meta = NoiseProfileMeta(
            sample_rate=int(raw["sample_rate"]),
            n_fft=int(raw["n_fft"]),
            hop_length=int(raw["hop_length"]),
            win_length=int(raw["win_length"]),
        )
        return mag.astype(np.float64), meta
    return mag.astype(np.float64), None


def load_noise_wav_mono(path: Path | str, target_sr: int) -> np.ndarray:
    """Load WAV/FLAC; mixdown to mono; resample to target_sr if needed."""
    import resampy

    path = Path(path)
    audio, sr = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = resampy.resample(audio.astype(np.float64), sr, target_sr).astype(np.float32)
    return audio.astype(np.float64)


class StreamingSpectralSubtractorMono:
    """
    One channel: buffers input, runs STFT frames (hop advance), overlap-add ISTFT.

    Latency: overlap-add needs several frames before the first output matches steady-state;
    until the internal queue has enough samples, this returns zeros (warmup).
    """

    def __init__(
        self,
        sample_rate: int,
        noise_mag: np.ndarray,
        n_fft: int,
        hop_length: int,
        win_length: int,
        *,
        alpha: float = 1.0,
        beta: float = 0.05,
        min_magnitude_ratio: float = 0.08,
    ) -> None:
        self._sft = _build_sft(sample_rate, n_fft, hop_length, win_length)
        n_bins = n_fft // 2 + 1
        if noise_mag.shape[0] != n_bins:
            raise ValueError(
                f"noise_mag length {noise_mag.shape[0]} != STFT bins {n_bins} "
                f"(n_fft={n_fft})"
            )
        self._noise_mag = noise_mag.astype(np.float64)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._min_mag_ratio = max(0.0, float(min_magnitude_ratio))
        self._hop = hop_length
        self._win_length = win_length

        self._buf = np.zeros(0, dtype=np.float64)
        self._frame_idx = 0
        # Overlap-add buffer; sample 0 is the next output time index.
        self._ola = np.zeros(0, dtype=np.float64)

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float64)
        self._frame_idx = 0
        self._ola = np.zeros(0, dtype=np.float64)

    def _ensure_ola(self, end_exclusive: int) -> None:
        if end_exclusive > self._ola.size:
            self._ola = np.resize(self._ola, end_exclusive + 4096)

    def _drain_frames(self) -> None:
        while self._buf.size >= self._win_length:
            seg = self._buf[: self._win_length]
            self._buf = self._buf[self._hop :]
            Z = self._sft.stft(seg)
            if Z.shape[1] > 1:
                Z = Z[:, :1]
            mag = np.abs(Z[:, 0])
            phase = np.angle(Z[:, 0])
            # alpha: over-subtract (stronger denoise, more musical noise risk).
            # beta: floor relative to noise spectrum (reduces isolated bin drops / warble).
            mag_new = np.maximum(
                mag - self._alpha * self._noise_mag,
                self._beta * self._noise_mag,
            )
            if self._min_mag_ratio > 0:
                mag_new = np.maximum(mag_new, self._min_mag_ratio * mag)
            Z_new = np.zeros_like(Z)
            Z_new[:, 0] = mag_new * np.exp(1j * phase)
            y = np.asarray(self._sft.istft(Z_new), dtype=np.float64)
            p = self._frame_idx
            start = p * self._hop
            end = start + y.size
            self._ensure_ola(end)
            self._ola[start:end] += y
            self._frame_idx += 1

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Process one chunk of samples; returns same length as *chunk*."""
        x = np.asarray(chunk, dtype=np.float64).ravel()
        n = x.size
        if n == 0:
            return x
        self._buf = np.concatenate([self._buf, x])
        self._drain_frames()

        # Until enough frames have been overlap-added, emit zeros (warmup / latency).
        if self._ola.size < n:
            return np.zeros(n, dtype=np.float64)
        out = self._ola[:n].copy()
        self._ola = self._ola[n:]
        return out


class StreamingSpectralSubtractor:
    """Stereo (or mono): independent StreamingSpectralSubtractorMono per channel."""

    def __init__(
        self,
        sample_rate: int,
        noise_mag: np.ndarray,
        n_fft: int,
        hop_length: int,
        win_length: int,
        *,
        alpha: float = 1.0,
        beta: float = 0.05,
        min_magnitude_ratio: float = 0.08,
    ) -> None:
        self._kw = dict(
            sample_rate=sample_rate,
            noise_mag=noise_mag,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            alpha=alpha,
            beta=beta,
            min_magnitude_ratio=min_magnitude_ratio,
        )
        self._L = StreamingSpectralSubtractorMono(**self._kw)
        self._R: StreamingSpectralSubtractorMono | None = None

    def _ensure_right(self) -> StreamingSpectralSubtractorMono:
        if self._R is None:
            self._R = StreamingSpectralSubtractorMono(**self._kw)
        return self._R

    def reset(self) -> None:
        self._L.reset()
        if self._R is not None:
            self._R.reset()

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        chunk: shape [N], [N,1], or [N,2]. Returns same shape (float32).
        """
        a = np.asarray(chunk, dtype=np.float32)
        if a.ndim == 1:
            return self._L.process(a).astype(np.float32)
        if a.shape[1] == 1:
            out1 = self._L.process(a[:, 0])
            return out1[:, np.newaxis].astype(np.float32)
        if a.shape[1] == 2:
            r = self._ensure_right()
            left = self._L.process(a[:, 0])
            right = r.process(a[:, 1])
            return np.stack([left, right], axis=1).astype(np.float32)
        raise ValueError(f"Expected 1 or 2 channels, got shape {a.shape}")


def apply_streaming_subtractor(
    subtractor: StreamingSpectralSubtractor | None,
    chunk: np.ndarray,
) -> np.ndarray:
    """If *subtractor* is None, returns *chunk* unchanged."""
    if subtractor is None:
        return chunk
    return subtractor.process(chunk)


def build_streaming_subtractor_from_config(
    config: Any,
) -> StreamingSpectralSubtractor | None:
    """
    Build subtractor from realtime Config.spectral_subtraction.
    Returns None if disabled or paths missing.
    """
    ss = getattr(config, "spectral_subtraction", None)
    if ss is None or not getattr(ss, "enabled", False):
        return None

    sr = config.audio.sample_rate
    n_fft = ss.n_fft
    hop = ss.hop_length
    win = ss.win_length

    noise_npy = getattr(ss, "noise_profile_npy", None)
    noise_wav = getattr(ss, "noise_wav", None)

    if noise_npy is not None and Path(noise_npy).is_file():
        mag, meta = load_noise_profile(noise_npy)
        if meta is not None:
            if (
                meta.sample_rate != sr
                or meta.n_fft != n_fft
                or meta.hop_length != hop
                or meta.win_length != win
            ):
                raise ValueError(
                    f"Noise profile meta {meta} does not match config STFT/sr; "
                    "re-estimate or fix config."
                )
    elif noise_wav is not None and Path(noise_wav).is_file():
        mono = load_noise_wav_mono(noise_wav, sr)
        mag = estimate_noise_magnitude_spectrum(mono, sr, n_fft, hop, win)
    else:
        return None

    return StreamingSpectralSubtractor(
        sr,
        mag,
        n_fft,
        hop,
        win,
        alpha=ss.alpha,
        beta=ss.beta,
        min_magnitude_ratio=ss.min_magnitude_ratio,
    )


def build_streaming_subtractor_from_magnitude(config: Any, magnitude: np.ndarray) -> StreamingSpectralSubtractor:
    """Build subtractor from a precomputed mean |STFT| vector (same STFT params as config)."""
    ss = config.spectral_subtraction
    sr = config.audio.sample_rate
    return StreamingSpectralSubtractor(
        sr,
        magnitude.astype(np.float64),
        ss.n_fft,
        ss.hop_length,
        ss.win_length,
        alpha=ss.alpha,
        beta=ss.beta,
        min_magnitude_ratio=ss.min_magnitude_ratio,
    )


def subtract_streaming_full_track(
    audio_mono: np.ndarray,
    subtractor_factory: Callable[[], StreamingSpectralSubtractorMono],
) -> np.ndarray:
    """
    Feed an entire 1D signal through a fresh mono subtractor in fixed chunks (validates OLA).
    """
    sub = subtractor_factory()
    chunk = 256
    n = audio_mono.size
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        sl = audio_mono[i : i + chunk]
        out[i : i + sl.size] = sub.process(sl)
    return out

