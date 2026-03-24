"""Overlap-save spectral gate and input noise gate for streaming noise reduction."""

from __future__ import annotations

import numpy as np


class InputNoiseGate:
    """Time-domain noise gate applied to input audio before model inference.

    Gates silence/ambient noise so the model doesn't try to extract a speaker
    from low-level noise.  Per 128-sample chunk:
      1. Compute frame RMS → dBFS
      2. If above threshold → target gain = 1.0, reset hold timer
      3. If below but hold timer active → target gain = 1.0, decrement hold
      4. Otherwise → target gain = 0.0
      5. Smooth gain with attack/release EMA coefficients
      6. Multiply chunk by smoothed gain
    """

    def __init__(
        self,
        enabled: bool = True,
        threshold_db: float = -40.0,
        attack_ms: float = 1.0,
        release_ms: float = 50.0,
        hold_ms: float = 20.0,
        sample_rate: int = 16000,
        chunk_size: int = 128,
    ) -> None:
        self.enabled = enabled
        self.threshold_db = threshold_db
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Convert ms to chunk counts for hold timer
        chunk_duration_ms = chunk_size / sample_rate * 1000.0
        self._hold_chunks = max(1, int(round(hold_ms / chunk_duration_ms)))

        # EMA coefficients: alpha = 1 - exp(-chunk_duration / tau)
        # tau = time_constant_ms / 1000 * sample_rate / chunk_size ... simplified:
        self._attack_coeff = 1.0 - np.exp(-chunk_duration_ms / max(attack_ms, 0.01))
        self._release_coeff = 1.0 - np.exp(-chunk_duration_ms / max(release_ms, 0.01))

        # State
        self._smooth_gain: float = 0.0
        self._hold_counter: int = 0

    def reset(self) -> None:
        self._smooth_gain = 0.0
        self._hold_counter = 0

    def _rms_dbfs(self, x: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        if rms < 1e-10:
            return -100.0
        return 20.0 * np.log10(rms)

    def process_chunk(self, chunk_1d: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return chunk_1d

        db = self._rms_dbfs(chunk_1d)

        if db >= self.threshold_db:
            target = 1.0
            self._hold_counter = self._hold_chunks
        elif self._hold_counter > 0:
            target = 1.0
            self._hold_counter -= 1
        else:
            target = 0.0

        # EMA smoothing
        if target > self._smooth_gain:
            coeff = self._attack_coeff
        else:
            coeff = self._release_coeff
        self._smooth_gain += coeff * (target - self._smooth_gain)

        return (chunk_1d * self._smooth_gain).astype(chunk_1d.dtype)

    def process_chunk_stereo(self, chunk_2d: np.ndarray) -> np.ndarray:
        """Gate a stereo chunk [chunk_size, 2]. Uses max RMS across channels."""
        if not self.enabled:
            return chunk_2d

        # Use max RMS across channels for gating decision
        db_l = self._rms_dbfs(chunk_2d[:, 0])
        db_r = self._rms_dbfs(chunk_2d[:, 1])
        db = max(db_l, db_r)

        if db >= self.threshold_db:
            target = 1.0
            self._hold_counter = self._hold_chunks
        elif self._hold_counter > 0:
            target = 1.0
            self._hold_counter -= 1
        else:
            target = 0.0

        if target > self._smooth_gain:
            coeff = self._attack_coeff
        else:
            coeff = self._release_coeff
        self._smooth_gain += coeff * (target - self._smooth_gain)

        return (chunk_2d * self._smooth_gain).astype(chunk_2d.dtype)


class SpectralGate:
    """Streaming spectral gate using overlap-save.

    A ring buffer of *frame_size* samples is maintained.  Each time a new
    *hop_size* chunk arrives the full frame is FFT'd, a soft spectral gate
    is applied, and the last *hop_size* samples of the IFFT are taken as
    output (overlap-save extracts the valid, non-circular region).

    State footprint (mono): input ring (frame_size) + noise floor
    (frame_size//2+1) ≈ 3 KB at frame_size=512.
    """

    def __init__(
        self,
        frame_size: int = 512,
        hop_size: int = 128,
        sr: int = 16000,
        noise_alpha: float = 0.98,
        gain_floor: float = 0.05,
        strength: float = 1.0,
        enabled: bool = False,
    ) -> None:
        if frame_size % hop_size != 0:
            raise ValueError(f"frame_size ({frame_size}) must be a multiple of hop_size ({hop_size})")

        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sr = sr
        self.noise_alpha = noise_alpha
        self.gain_floor = gain_floor
        self.strength = strength
        self.enabled = enabled

        self._n_bins = frame_size // 2 + 1

        self._ring: np.ndarray | None = None
        self._noise_floor: np.ndarray | None = None
        self._initialised = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        self._ring = np.zeros(self.frame_size, dtype=np.float32)
        self._noise_floor = np.zeros(self._n_bins, dtype=np.float32)
        self._initialised = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state."""
        self._initialised = False
        self._ring = None
        self._noise_floor = None

    def process_chunk(self, chunk_1d: np.ndarray) -> np.ndarray:
        """Denoise a single-channel chunk of *hop_size* samples.

        Returns an array of the same shape and dtype.
        """
        if not self.enabled:
            return chunk_1d

        if not self._initialised:
            self._init_state()

        assert self._ring is not None
        assert self._noise_floor is not None

        hop = self.hop_size
        # Shift ring buffer left and insert new chunk at the end
        self._ring[:-hop] = self._ring[hop:]
        self._ring[-hop:] = chunk_1d

        # FFT (no window — overlap-save relies on extracting valid samples)
        X = np.fft.rfft(self._ring)
        P = np.abs(X) ** 2

        # Update noise floor EMA (only noise-dominated bins)
        alpha = self.noise_alpha
        noise_mask = P < 2.0 * (self._noise_floor + 1e-10)
        self._noise_floor[noise_mask] = (
            alpha * self._noise_floor[noise_mask] + (1 - alpha) * P[noise_mask]
        )
        # Bootstrap: for bins never updated, seed with current power
        cold = self._noise_floor == 0.0
        self._noise_floor[cold] = P[cold]

        # Soft spectral gate
        G = np.maximum(
            1.0 - self.strength * self._noise_floor / (P + 1e-10),
            self.gain_floor,
        )
        Y = G * X

        # IFFT — take last hop_size samples (overlap-save valid region)
        y = np.fft.irfft(Y, n=self.frame_size).astype(np.float32)
        return y[-hop:]

    def process_chunk_stereo(self, chunk_2d: np.ndarray) -> np.ndarray:
        """Denoise a stereo chunk of shape [chunk_size, 2].

        Each channel is processed independently by its own SpectralGate instance.
        On first call, the second-channel gate is lazily created.
        """
        if not self.enabled:
            return chunk_2d

        if not hasattr(self, "_ch1_gate"):
            self._ch1_gate = SpectralGate(
                frame_size=self.frame_size,
                hop_size=self.hop_size,
                sr=self.sr,
                noise_alpha=self.noise_alpha,
                gain_floor=self.gain_floor,
                strength=self.strength,
                enabled=True,
            )

        out = np.empty_like(chunk_2d)
        out[:, 0] = self.process_chunk(chunk_2d[:, 0])
        out[:, 1] = self._ch1_gate.process_chunk(chunk_2d[:, 1])
        return out
