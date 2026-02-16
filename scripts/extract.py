"""
Standalone CLI script to run TFGridNet extraction on a mixture WAV file.

Usage:
    python extract.py --audio /path/to/mixture.wav --speaker "Alice"
    python extract.py --audio /path/to/mixture.wav --speaker "Alice" --speaker "Bob"
"""

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
MEDIA_DIR = BACKEND_DIR / "media"
DATA_FILE = MEDIA_DIR / "data.json"
EXTRACTED_DIR = MEDIA_DIR / "extracted"


def main():
    parser = argparse.ArgumentParser(
        description="Extract target speaker(s) from a binaural mixture."
    )
    parser.add_argument("--audio", required=True, help="Path to a stereo WAV mixture file")
    parser.add_argument(
        "--speaker",
        required=True,
        action="append",
        dest="speakers",
        help="Speaker name to extract (repeatable)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    info = sf.info(str(audio_path))
    if info.channels != 2:
        raise ValueError(
            f"Audio must be binaural (2-channel), got {info.channels} channel(s)."
        )

    from src.persistence import MediaJsonStore

    store = MediaJsonStore(media_root=MEDIA_DIR, data_file=DATA_FILE)
    speakers, _recordings, _extractions = store.load()

    # Look up each requested speaker name
    speaker_names_to_find = args.speakers
    found_speakers = []
    for name in speaker_names_to_find:
        match = next((s for s in speakers if s.name == name), None)
        if match is None:
            enrolled = [s.name for s in speakers]
            raise ValueError(
                f"Speaker '{name}' not found. Enrolled speakers: {enrolled}"
            )
        found_speakers.append(match)

    embedding_paths = [s.embedding_path for s in found_speakers]

    from src.ml.TFGridNetExtractionModel import TFGridNetExtractionModel
    from src.utils import sanitize_for_filename

    model = TFGridNetExtractionModel()

    # Build a combined prefix from speaker names
    prefix = "_".join(sanitize_for_filename(s.name) for s in found_speakers)

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {len(found_speakers)} speaker(s) from {audio_path.name}...")
    t0 = time.perf_counter()
    output_paths = model.separate(
        mixture_audio_path=audio_path,
        speaker_embedding_paths=embedding_paths,
        output_dir=EXTRACTED_DIR,
        output_name_prefix=prefix,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Done in {elapsed_ms:.0f}ms. Output files:")
    for p in output_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
