#!/usr/bin/env python3
"""
Generate a synthetic SI-SDR fixture for eval-fast.

Produces:
  <output-dir>/mixture.wav        — stereo mix (model input)
  <output-dir>/reference.wav      — stereo clean target (SI-SDR ground truth)
  <output-dir>/enrollment.npy     — speaker embedding [256,]
  <output-dir>/enrollment.meta.json — sidecar for validation
  <output-dir>/fixture.json       — provenance metadata

Usage:
    python scripts/make_fixture.py \
        --target      data/our_speech_pool/Hady.wav \
        --interferers data/our_speech_pool/Himanshu.wav data/our_speech_pool/Matt.wav \
        --noise       data/wham_noise/011a0101_0.8207_2ajyi_-0.8207.wav \
        --output-dir  media/si_sdr_fixture \
        --embedding-model resemblyzer \
        --snr-speech  0 \
        --snr-noise   20
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ml.factory import EMBEDDING_MODEL_IDS, create_embedding_model, embedding_model_class_name

TARGET_SR = 16000


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Return [N, 2] float32. Binaural files pass through unchanged; mono is duplicated."""
    audio, sr = sf.read(str(path), always_2d=True)
    if sr != target_sr:
        import resampy
        audio = resampy.resample(audio.T, sr, target_sr).T
    if audio.shape[1] == 1:
        audio = np.concatenate([audio, audio], axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio.astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-8)


def make_fixture(
    target_path: Path,
    interferer_paths: list[Path],
    noise_path: Path | None,
    output_dir: Path,
    snr_speech: float,
    snr_noise: float,
    embedding_model_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all sources
    target = load_audio(target_path)
    interferers = [load_audio(p) for p in interferer_paths]
    noise = load_audio(noise_path) if noise_path else None

    # Trim to shortest speech segment
    min_len = min(len(target), *(len(i) for i in interferers))
    target = target[:min_len]
    interferers = [i[:min_len] for i in interferers]

    # Tile-then-trim noise to match length
    if noise is not None:
        if len(noise) < min_len:
            repeats = (min_len // len(noise)) + 1
            noise = np.tile(noise, (repeats, 1))
        noise = noise[:min_len]

    # RMS-based mixing
    target_norm = target / rms(target)
    att = 10 ** (-snr_speech / 20)
    speech_mix = target_norm + sum(att * (i / rms(i)) for i in interferers)

    if noise is not None:
        noise_scaled = (noise / rms(noise)) * rms(speech_mix) * 10 ** (-snr_noise / 20)
        mixture = speech_mix + noise_scaled
    else:
        mixture = speech_mix

    mixture = mixture / (np.abs(mixture).max() + 1e-8) * 0.9
    reference = target_norm  # scale-invariant; SI-SDR doesn't depend on absolute level

    # Write mixture and reference
    sf.write(str(output_dir / "mixture.wav"), mixture, TARGET_SR)
    sf.write(str(output_dir / "reference.wav"), reference, TARGET_SR)
    print(f"mixture.wav  : {mixture.shape}  @ {TARGET_SR}Hz")
    print(f"reference.wav: {reference.shape} @ {TARGET_SR}Hz")

    # Compute embedding from raw (un-normalised) target
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    sf.write(str(tmp), target, TARGET_SR)
    embedding_model = create_embedding_model(embedding_model_id)
    embedding = embedding_model.compute_embedding(tmp)
    tmp.unlink()

    emb_path = output_dir / "enrollment.npy"
    np.save(str(emb_path), embedding)
    print(f"enrollment.npy: {embedding.shape} (model={embedding_model_id})")

    # Write sidecar
    sidecar = {
        "embedding_model_id": embedding_model_id,
        "embedding_model_class": embedding_model_class_name(embedding_model_id),
        "sample_rate": TARGET_SR,
    }
    (output_dir / "enrollment.meta.json").write_text(json.dumps(sidecar, indent=2))
    print("enrollment.meta.json written")

    # Provenance
    duration = min_len / TARGET_SR
    fixture_meta = {
        "target": str(target_path),
        "interferers": [str(p) for p in interferer_paths],
        "noise": str(noise_path) if noise_path else None,
        "embedding_model_id": embedding_model_id,
        "embedding_model_class": embedding_model_class_name(embedding_model_id),
        "snr_speech_db": snr_speech,
        "snr_noise_db": snr_noise,
        "duration_s": round(duration, 3),
        "sample_rate": TARGET_SR,
        "num_samples": min_len,
    }
    (output_dir / "fixture.json").write_text(json.dumps(fixture_meta, indent=2))
    print(f"fixture.json written  (duration={duration:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SI-SDR eval fixture")
    parser.add_argument("--target", type=Path, required=True, help="Clean WAV of the target speaker")
    parser.add_argument("--interferers", type=Path, nargs="+", required=True, help="Clean WAV(s) of competing speakers")
    parser.add_argument("--noise", type=Path, default=None, help="Background noise WAV (optional)")
    parser.add_argument("--output-dir", type=Path, default=Path("media/si_sdr_fixture"))
    parser.add_argument("--snr-speech", type=float, default=0.0, help="dB of target over each interferer")
    parser.add_argument("--snr-noise", type=float, default=20.0, help="dB of speech mix above noise")
    parser.add_argument(
        "--embedding-model",
        choices=EMBEDDING_MODEL_IDS,
        default="resemblyzer",
        help="Speaker embedding model used to build enrollment.npy (default: resemblyzer)",
    )
    args = parser.parse_args()

    make_fixture(
        target_path=args.target,
        interferer_paths=args.interferers,
        noise_path=args.noise,
        output_dir=args.output_dir,
        snr_speech=args.snr_speech,
        snr_noise=args.snr_noise,
        embedding_model_id=args.embedding_model,
    )


if __name__ == "__main__":
    main()
