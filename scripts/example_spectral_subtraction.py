#!/usr/bin/env python3
"""
Minimal example: estimate a noise magnitude profile from a noise-only segment,
then run streaming spectral subtraction on a synthetic mixture (tone + noise).

Requires: numpy, scipy, soundfile (same env as the realtime stack).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.realtime.spectral_subtraction import (  # noqa: E402
    StreamingSpectralSubtractorMono,
    estimate_noise_magnitude_spectrum,
    save_noise_profile,
    NoiseProfileMeta,
)


def main() -> None:
    sr = 16_000
    duration_s = 2.0
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr

    # Stationary noise (what we record as a noise-only profile)
    rng = np.random.default_rng(0)
    noise_only = (0.1 * rng.standard_normal(n)).astype(np.float64)

    n_fft = 512
    hop = 128
    win = 512

    noise_mag = estimate_noise_magnitude_spectrum(noise_only, sr, n_fft, hop, win)
    meta = NoiseProfileMeta(sample_rate=sr, n_fft=n_fft, hop_length=hop, win_length=win)
    out_dir = REPO_ROOT / "media" / "spectral_subtraction_example"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_noise_profile(out_dir / "noise_mag.npy", noise_mag, meta)

    # Synthetic mixture: sinusoid + same noise class
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    mixture = (tone + 0.1 * rng.standard_normal(n)).astype(np.float64)

    sub = StreamingSpectralSubtractorMono(sr, noise_mag, n_fft, hop, win, alpha=1.0, beta=0.05)
    chunk = 128
    cleaned = np.empty_like(mixture)
    for i in range(0, n, chunk):
        sl = mixture[i : i + chunk]
        cleaned[i : i + sl.size] = sub.process(sl)

    sf.write(str(out_dir / "mixture.wav"), mixture, sr)
    sf.write(str(out_dir / "cleaned.wav"), cleaned.astype(np.float32), sr)
    sf.write(str(out_dir / "noise_only.wav"), noise_only.astype(np.float32), sr)
    print(f"Wrote WAVs and noise_mag.npy under {out_dir}")


if __name__ == "__main__":
    main()
