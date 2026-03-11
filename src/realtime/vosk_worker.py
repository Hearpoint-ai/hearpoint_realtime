"""
Standalone Vosk name-detection worker that runs in a child process.

Runs in its own OS process (via multiprocessing) so Vosk's CPU-bound
recognition cannot starve the 8 ms audio processing thread of GIL time.
"""

import json
import multiprocessing
from pathlib import Path

import numpy as np


def vosk_worker(
    audio_queue: multiprocessing.Queue,
    detection_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
    model_path: str,
    sample_rate: int,
    target_word: str,
) -> None:
    """Entry point for the Vosk child process.

    Parameters
    ----------
    audio_queue : multiprocessing.Queue
        Receives float32 numpy audio chunks (shape [N, C]).
    detection_event : multiprocessing.Event
        Set when the *target_word* is detected.
    stop_event : multiprocessing.Event
        Parent sets this to signal the worker to exit.
    model_path : str
        Filesystem path to the Vosk model directory.
    sample_rate : int
        Audio sample rate (Hz).
    target_word : str
        Lower-cased word to listen for.
    """
    from vosk import Model, KaldiRecognizer  # heavy import inside child only

    model_dir = Path(model_path)
    if not model_dir.exists():
        return

    model = Model(str(model_dir))
    recognizer = KaldiRecognizer(model, sample_rate)
    target = target_word.lower().strip()

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except Exception:
            # queue.Empty (or EOFError on teardown)
            continue

        try:
            # Convert float32 (stereo or mono) → mono int16 bytes for Vosk
            if chunk.ndim > 1:
                mono = chunk.mean(axis=1)
            else:
                mono = chunk.flatten()
            data = (mono * 32767).astype(np.int16).tobytes()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = (result.get("text") or "").lower().strip()
                if target in text.split():
                    detection_event.set()
            else:
                partial = json.loads(recognizer.PartialResult())
                text = (partial.get("partial") or "").lower().strip()
                if text and target in text.split():
                    detection_event.set()
        except Exception:
            pass
