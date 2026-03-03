# HearPoint Realtime

## Overview

HearPoint Realtime is a real-time hearing aid system designed to solve the **cocktail party problem** — isolating the voice of a desired speaker in noisy, multi-speaker environments. The goal is to build a low-latency streaming pipeline that can extract a target speaker's voice from a binaural audio mixture, enabling hearing-impaired users to focus on a specific person in real-world social settings.

## How It Works

1. **Speaker Enrollment**: A target speaker is enrolled by recording a short audio sample. A TFGridNet-based embedding model computes a 256-dimensional speaker embedding vector, which is stored for later use.

2. **Realtime Inference**: Binaural audio is captured from a microphone in small chunks (128 samples / ~8ms at 16kHz). Each chunk is passed through a causal streaming TFGridNet model conditioned on the enrolled speaker's embedding. The model extracts the target speaker's voice from the mixture in real time, with state caching to maintain temporal context across chunks. Processed audio is played back with ~40-50ms latency.

3. **Offline Extraction**: Pre-recorded mixtures can also be processed in batch mode for testing and evaluation.

## Tech Stack

- **PyTorch** — deep learning framework for the TFGridNet models
- **sounddevice** — real-time audio I/O
- **asteroid-filterbanks** — STFT/filterbank implementations
- **Resemblyzer** — alternative speaker embedding model
- **NumPy, resampy, soundfile** — audio processing utilities
- **Conda** — environment management (Python 3.11)

## Project Structure

```
src/
  ml/              # Model wrappers (embedding + extraction)
  models/
    tfgridnet_realtime/    # Causal streaming TFGridNet for inference
    tfgridnet_enrollment/  # TFGridNet for speaker embedding computation
  realtime/        # Realtime inference engine (realtime_inference.py, config.yaml)
  configs/         # Model architecture configs (JSON)
scripts/
  enroll.py        # CLI for speaker enrollment
  extract.py       # CLI for offline extraction
weights/           # Model checkpoints (.ckpt)
media/
  enrollments/     # Speaker embeddings (.npy) and enrollment audio (.wav)
  mixtures/        # Input binaural mixtures
  extracted/       # Extracted output audio
```

## Key Entry Points

- **Enrollment**: `scripts/enroll.py`
- **Offline Extraction**: `scripts/extract.py`
- **Realtime Streaming**: `src/realtime/realtime_inference.py`
- **Runtime Config**: `src/realtime/config.yaml`
