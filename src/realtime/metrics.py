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
