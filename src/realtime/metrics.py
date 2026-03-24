from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import yaml

from .config import DEFAULT_THRESHOLDS_PATH


# Operators hard-coded per metric key — do not infer from value
_THRESHOLD_OPS: dict[str, str] = {
    "drops_input": "==",
    "drops_output": "==",
    "underruns": "==",
    "nan_count": "==",
    "rtf_avg": "<",
    "clip_ratio": "<=",
    "cosine_similarity_delta": ">=",
    "si_sdr_improvement": ">=",
    "spectral_flatness": "<=",
    "hf_energy_ratio": "<=",
    "noise_floor_db": "<=",
    "estimated_snr_db": ">=",
}

# Excluded from threshold evaluation in file mode (always stub-zero)
_FILE_MODE_EXCLUDED: set[str] = {"drops_input", "drops_output", "underruns"}


def _load_threshold_profile(profile: str, path: Path = DEFAULT_THRESHOLDS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"thresholds.yaml not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if profile not in data:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(data.keys())}")
    return {k: v for k, v in data[profile].items() if v is not None}


def _evaluate_thresholds(stats: dict, profile_thresholds: dict, mode: str) -> list[str]:
    """Return list of metric names that failed their threshold check."""
    excluded = _FILE_MODE_EXCLUDED if mode == "file" else set()
    failed = []
    for metric, limit in profile_thresholds.items():
        if metric in excluded or metric not in stats:
            continue
        op = _THRESHOLD_OPS.get(metric, "==")
        actual = stats[metric]
        if actual is None:
            continue  # skip optional metrics absent from this run
        if op == "==" and actual != limit:
            failed.append(metric)
        elif op == "<" and not (actual < limit):
            failed.append(metric)
        elif op == "<=" and not (actual <= limit):
            failed.append(metric)
        elif op == ">=" and not (actual >= limit):
            failed.append(metric)
    return failed


def _write_report(stats: dict, report_path: Path, failed_thresholds: list[str]) -> None:
    """Write JSON report to disk annotated with pass/fail status."""
    out = dict(stats)
    out["status"] = "pass" if not failed_thresholds else "fail"
    out["failed_thresholds"] = failed_thresholds
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Report written: {report_path}")
    if failed_thresholds:
        print(f"THRESHOLD FAILURES: {failed_thresholds}")


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Return audio as [N, 2] float32, duplicating channel 0 if mono.

    Accepts:
      [N]    — 1-D mono
      [N, 1] — column-mono
      [N, 2] — stereo (pass-through)
    """
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.shape[1] == 1:
        audio = np.concatenate([audio, audio], axis=1)
    return audio.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SI-SDR in dB for a single channel [N]."""
    ref = reference.astype(np.float64) - reference.mean()
    est = estimate.astype(np.float64) - estimate.mean()
    # Normalize to unit scale to avoid float64 overflow in dot products
    ref_scale = np.linalg.norm(ref) + 1e-8
    ref = ref / ref_scale
    est = est / ref_scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
        target = alpha * ref
        noise = est - target
        target_pow = np.dot(target, target)
        noise_pow = np.dot(noise, noise)
    # Add 1e-10 offset after ratio so that zero-estimate → -100 dB, not 0 dB
    return float(10 * np.log10(target_pow / (noise_pow + 1e-8) + 1e-10))


def _si_sdr_stereo(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Mean SI-SDR across channels. Inputs: [N, 2]."""
    return float(np.mean([_si_sdr(reference[:, c], estimate[:, c]) for c in range(2)]))


# ---------------------------------------------------------------------------
# Noise evaluation metrics
# ---------------------------------------------------------------------------


def _spectral_flatness(audio_1d: np.ndarray, sr: int, frame_len: int = 512) -> float:
    """Geometric / arithmetic mean of power spectrum (Wiener entropy).

    White noise → 1.0, tonal/speech → 0.0.
    """
    n_frames = len(audio_1d) // frame_len
    if n_frames == 0:
        return 0.0

    ratios: list[float] = []
    for i in range(n_frames):
        frame = audio_1d[i * frame_len : (i + 1) * frame_len].astype(np.float64)
        P = np.abs(np.fft.rfft(frame)) ** 2 + 1e-12
        geo_mean = np.exp(np.mean(np.log(P)))
        arith_mean = np.mean(P)
        ratios.append(geo_mean / (arith_mean + 1e-12))

    return float(np.mean(ratios))


def _hf_energy_ratio(audio_1d: np.ndarray, sr: int, cutoff_hz: int = 4000, frame_len: int = 512) -> float:
    """Fraction of energy above *cutoff_hz*. White noise → ~0.5, speech → low."""
    n_frames = len(audio_1d) // frame_len
    if n_frames == 0:
        return 0.0

    n_bins = frame_len // 2 + 1
    cutoff_bin = int(cutoff_hz * frame_len / sr)
    cutoff_bin = min(cutoff_bin, n_bins)

    total_energy = 0.0
    hf_energy = 0.0
    for i in range(n_frames):
        frame = audio_1d[i * frame_len : (i + 1) * frame_len].astype(np.float64)
        P = np.abs(np.fft.rfft(frame)) ** 2
        total_energy += P.sum()
        hf_energy += P[cutoff_bin:].sum()

    return float(hf_energy / (total_energy + 1e-12))


def _noise_floor_db(audio_1d: np.ndarray, sr: int, frame_len_ms: float = 32) -> float:
    """10th percentile of frame-level RMS in dBFS. Lower = less noise."""
    frame_len = int(sr * frame_len_ms / 1000)
    n_frames = len(audio_1d) // frame_len
    if n_frames == 0:
        return -100.0

    rms_vals = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame = audio_1d[i * frame_len : (i + 1) * frame_len].astype(np.float64)
        rms_vals[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)

    p10 = float(np.percentile(rms_vals, 10))
    return float(20 * np.log10(p10 + 1e-12))


def _estimated_snr_db(audio_1d: np.ndarray, sr: int, frame_len_ms: float = 32) -> float:
    """Estimated SNR: active-speech RMS (>p50 frames) minus noise floor (p10) in dB."""
    frame_len = int(sr * frame_len_ms / 1000)
    n_frames = len(audio_1d) // frame_len
    if n_frames == 0:
        return 0.0

    rms_vals = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame = audio_1d[i * frame_len : (i + 1) * frame_len].astype(np.float64)
        rms_vals[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)

    p10 = float(np.percentile(rms_vals, 10))
    p50 = float(np.percentile(rms_vals, 50))
    active_rms = float(np.mean(rms_vals[rms_vals >= p50])) if np.any(rms_vals >= p50) else p50

    snr = 20 * np.log10((active_rms + 1e-12) / (p10 + 1e-12))
    return float(snr)
