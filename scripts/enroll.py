"""
Standalone CLI script to enroll a speaker from an audio file or mic recording.

Usage:
    python enroll.py --name "Alice" --audio /path/to/recording.wav
    python enroll.py --name "Alice" --record --duration 5
"""

import argparse
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MEDIA_DIR = REPO_ROOT / "media"
ENROLLMENTS_DIR = MEDIA_DIR / "enrollments"
DATA_FILE = MEDIA_DIR / "data.json"

sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 16000


def find_stereo_input_device():
    """Find an input device with at least 2 channels. Raises if none found."""
    devices = sd.query_devices()
    # Prefer the default input device if it has 2+ channels
    default_idx = sd.default.device[0]
    if default_idx is not None:
        default_dev = sd.query_devices(default_idx, "input")
        if default_dev["max_input_channels"] >= 2:
            return default_idx
    # Otherwise search all input devices
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] >= 2:
            return i
    raise RuntimeError(
        "No stereo (2-channel) input device found. "
        "Available devices:\n" + sd.query_devices().__repr__()
    )


def record_from_mic(duration: float, dest_path: Path):
    """Record from a 2-channel mic and save as WAV."""
    device_idx = find_stereo_input_device()
    dev_info = sd.query_devices(device_idx, "input")
    print(f"Recording from: {dev_info['name']} ({dev_info['max_input_channels']}ch)")
    print(f"Duration: {duration}s | Sample rate: {SAMPLE_RATE} Hz")
    print("Speak now...")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=2,
        dtype="float32",
        device=device_idx,
    )
    sd.wait()
    print("Recording complete.")

    # audio shape: [N, 2]
    if audio.shape[1] != 2:
        raise RuntimeError(f"Expected 2-channel audio, got {audio.shape[1]} channels")

    sf.write(str(dest_path), audio, SAMPLE_RATE)


def main():
    parser = argparse.ArgumentParser(description="Enroll a speaker from an audio file or mic recording.")
    parser.add_argument("--name", required=True, help="Speaker name")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="Path to a stereo WAV file")
    source.add_argument("--record", action="store_true", help="Record from a 2-channel microphone")

    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds (default: 5)")
    args = parser.parse_args()

    ENROLLMENTS_DIR.mkdir(parents=True, exist_ok=True)
    speaker_id = str(uuid.uuid4())
    dest_audio = ENROLLMENTS_DIR / f"{speaker_id}.wav"

    if args.record:
        record_from_mic(args.duration, dest_audio)
    else:
        audio_path = Path(args.audio).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        shutil.copy2(audio_path, dest_audio)

    # Validate the saved audio is binaural
    info = sf.info(str(dest_audio))
    if info.channels != 2:
        dest_audio.unlink()
        raise ValueError(
            f"Audio must be binaural (2-channel), got {info.channels} channel(s). "
            "Please provide a stereo recording."
        )

    # Compute embedding
    from src.ml.TFGridNetSpeakerEmbeddingModel import TFGridNetSpeakerEmbeddingModel

    t0 = time.perf_counter()
    model = TFGridNetSpeakerEmbeddingModel()
    embedding = model.compute_embedding(dest_audio)
    processing_ms = (time.perf_counter() - t0) * 1000

    # Save embedding
    dest_embedding = ENROLLMENTS_DIR / f"{speaker_id}.npy"
    np.save(dest_embedding, embedding)

    # Update data.json
    from src.models import Speaker
    from src.persistence import MediaJsonStore

    store = MediaJsonStore(media_root=MEDIA_DIR, data_file=DATA_FILE)
    speakers, recordings, extractions = store.load()

    speaker = Speaker(
        id=speaker_id,
        name=args.name,
        created_at=datetime.now(timezone.utc),
        embedding_path=dest_embedding,
        enrollment_audio_path=dest_audio,
        processing_ms=processing_ms,
    )
    speakers.append(speaker)
    store.save(speakers, recordings, extractions)

    print(f"Enrolled speaker '{args.name}' (id={speaker_id}) in {processing_ms:.0f}ms")


if __name__ == "__main__":
    main()
