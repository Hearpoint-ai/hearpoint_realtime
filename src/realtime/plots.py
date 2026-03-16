from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in subprocesses
import matplotlib.pyplot as plt

from .file_eval import PlotData


# ---------------------------------------------------------------------------
# Plot 1: Spectrogram comparison
# ---------------------------------------------------------------------------

def plot_spectrogram_comparison(plot_data: PlotData, output_dir: Path, ts: str = "") -> None:
    mixture_mono = plot_data.mixture.mean(axis=1)
    output_mono = plot_data.output.mean(axis=1)

    has_ref = plot_data.reference is not None
    n_panels = 3 if has_ref else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4), sharey=True)

    sr = plot_data.sample_rate
    nfft = 512
    noverlap = 384

    def _specgram(ax: plt.Axes, audio: np.ndarray, title: str) -> None:
        ax.specgram(audio, Fs=sr, NFFT=nfft, noverlap=noverlap, cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    _specgram(axes[0], mixture_mono, "Mixture")
    _specgram(axes[1], output_mono, "Output")
    if has_ref:
        ref_mono = plot_data.reference.mean(axis=1)  # type: ignore[union-attr]
        _specgram(axes[2], ref_mono, "Reference")

    title = f"Spectrogram Comparison — {ts}" if ts else "Spectrogram Comparison"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fname = f"{ts}_spectrogram.png" if ts else "spectrogram.png"
    fig.savefig(output_dir / fname, dpi=120)
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 2: SI-SDRi vs input SNR (requires reference)
# ---------------------------------------------------------------------------

def _si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR between two 1-D arrays."""
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    scale = np.linalg.norm(ref) + 1e-8
    ref = ref / scale
    est = est / scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dot = np.dot(est, ref)
        ref_energy = np.dot(ref, ref) + 1e-8
        proj = (dot / ref_energy) * ref
        noise = est - proj
        return float(10 * np.log10((np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8)))


def plot_sisdri_vs_snr(plot_data: PlotData, output_dir: Path, ts: str = "") -> None:
    if plot_data.reference is None:
        return

    sr = plot_data.sample_rate
    seg_len = sr  # 1-second windows

    n_samples = min(len(plot_data.mixture), len(plot_data.output), len(plot_data.reference))
    n_segs = n_samples // seg_len

    snr_in_vals: list[float] = []
    sisdri_vals: list[float] = []

    for i in range(n_segs):
        s = i * seg_len
        e = s + seg_len
        mix_seg = plot_data.mixture[s:e].mean(axis=1)
        out_seg = plot_data.output[s:e].mean(axis=1)
        ref_seg = plot_data.reference[s:e].mean(axis=1)

        snr_in = _si_sdr(mix_seg, ref_seg)
        si_sdr_out = _si_sdr(out_seg, ref_seg)
        sisdri_vals.append(si_sdr_out - snr_in)
        snr_in_vals.append(snr_in)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(snr_in_vals, sisdri_vals, alpha=0.6, s=20, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="SI-SDRi = 0")
    ax.set_xlabel("Input SNR / SI-SDR (dB)")
    ax.set_ylabel("SI-SDRi (dB)")
    title = f"SI-SDRi vs Input SNR (1-second segments) — {ts}" if ts else "SI-SDRi vs Input SNR (1-second segments)"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fname = f"{ts}_sisdri_vs_snr.png" if ts else "sisdri_vs_snr.png"
    fig.savefig(output_dir / fname, dpi=120)
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 3: RTF distribution
# ---------------------------------------------------------------------------

def plot_rtf_distribution(plot_data: PlotData, stats: dict, output_dir: Path, ts: str = "") -> None:
    rtf_vals = plot_data.chunk_times_s / plot_data.chunk_duration_s

    p50 = float(np.percentile(rtf_vals, 50))
    p95 = float(np.percentile(rtf_vals, 95))
    p99 = float(np.percentile(rtf_vals, 99))
    mean_rtf = float(np.mean(rtf_vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rtf_vals, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(1.0, color="red", linewidth=1.5, label="RTF = 1.0 (real-time limit)")
    ax.axvline(mean_rtf, color="orange", linestyle="--", linewidth=1.5, label=f"Mean = {mean_rtf:.3f}")
    ax.axvline(p50, color="green", linestyle=":", linewidth=1.2, label=f"p50 = {p50:.3f}")
    ax.axvline(p95, color="purple", linestyle=":", linewidth=1.2, label=f"p95 = {p95:.3f}")
    ax.axvline(p99, color="brown", linestyle=":", linewidth=1.2, label=f"p99 = {p99:.3f}")
    ax.set_xlabel("RTF (processing time / chunk duration)")
    ax.set_ylabel("Count")
    title = f"Per-chunk RTF Distribution — {ts}" if ts else "Per-chunk RTF Distribution"
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fname = f"{ts}_rtf_distribution.png" if ts else "rtf_distribution.png"
    fig.savefig(output_dir / fname, dpi=120)
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_plots(plot_data: PlotData, stats: dict, output_dir: Path, ts: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating plots -> {output_dir}")
    plot_rtf_distribution(plot_data, stats, output_dir, ts=ts)
    plot_spectrogram_comparison(plot_data, output_dir, ts=ts)
    if plot_data.reference is not None:
        plot_sisdri_vs_snr(plot_data, output_dir, ts=ts)
