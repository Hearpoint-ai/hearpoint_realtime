import unittest

import numpy as np

try:
    import scipy  # noqa: F401

    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from src.realtime.spectral_subtraction import (
    StreamingSpectralSubtractorMono,
    estimate_noise_magnitude_spectrum,
)


@unittest.skipUnless(_HAVE_SCIPY, "scipy required")
class SpectralSubtractionTests(unittest.TestCase):
    def test_estimate_mean_matches_manual_stft(self) -> None:
        sr = 16_000
        rng = np.random.default_rng(42)
        n_fft, hop, win = 256, 64, 256
        noise = rng.standard_normal(5 * sr).astype(np.float64)
        mag_mean = estimate_noise_magnitude_spectrum(noise, sr, n_fft, hop, win)

        from scipy.signal import ShortTimeFFT, get_window

        w = get_window("hann", win, fftbins=True).astype(np.float64)
        sft = ShortTimeFFT(w, hop, sr, mfft=n_fft, scale_to=None)
        Z = sft.stft(noise)
        ref = np.mean(np.abs(Z), axis=1, dtype=np.float64)
        np.testing.assert_allclose(mag_mean, ref, rtol=1e-5, atol=1e-8)

    def test_noise_only_streaming_reduces_rms(self) -> None:
        sr = 16_000
        rng = np.random.default_rng(0)
        n_fft, hop, win = 512, 128, 512
        noise = rng.standard_normal(3 * sr).astype(np.float64)
        noise_mag = estimate_noise_magnitude_spectrum(noise, sr, n_fft, hop, win)
        sub = StreamingSpectralSubtractorMono(sr, noise_mag, n_fft, hop, win, alpha=1.0, beta=0.0)

        chunk = 256
        n = 2 * sr
        out = np.empty(n, dtype=np.float64)
        x = noise[:n].copy()
        for i in range(0, n, chunk):
            sl = x[i : i + chunk]
            out[i : i + sl.size] = sub.process(sl)

        # Ignore startup zeros / tail where OLA has not settled
        mid = slice(sr // 2, n - sr // 2)
        rms_in = float(np.sqrt(np.mean(x[mid] ** 2)))
        rms_out = float(np.sqrt(np.mean(out[mid] ** 2)))
        self.assertLess(rms_out, rms_in * 0.95)


if __name__ == "__main__":
    unittest.main()
