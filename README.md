# HearPoint Realtime

Real-time target speech extraction using TFGridNet. Given a speaker embedding, the system isolates that speaker's voice from a binaural audio mixture in real-time, targeting hearing aid latency requirements (~40-50 ms).

## Repository Structure

```
hearpoint_realtime/
├── src/
│   ├── ml/                        # ML model wrappers
│   │   ├── interfaces.py          # Abstract base classes (SpeakerEmbeddingModel, TargetSpeechExtractionModel)
│   │   ├── TFGridNetSpeakerEmbeddingModel.py   # Speaker enrollment embedding model
│   │   ├── TFGridNetExtractionModel.py         # Offline target speech extraction
│   │   ├── CleanedTfSpeakerEmbedding.py        # Experimental cleaned-up embedding model
│   │   ├── ResemblyzerSpeakerEmbeddingModel.py # Resemblyzer-based embedding model
│   │   ├── CopyMixtureExtractionModel.py       # Passthrough baseline extraction model
│   │   ├── MockSpeakerEmbeddingModel.py        # Mock for testing
│   │   └── binaural_synth.py      # Binaural audio synthesis utilities
│   ├── models/                    # Neural network architectures
│   │   ├── tfgridnet_realtime/    # Causal streaming TFGridNet (used for real-time inference)
│   │   └── tfgridnet_enrollment/  # TFGridNet variant for computing speaker embeddings
│   ├── realtime/                  # Real-time inference engine
│   │   ├── realtime_inference.py  # Streaming audio capture, model inference, and playback
│   │   ├── config.yaml            # Runtime configuration
│   │   └── README.md              # Detailed real-time module documentation
│   ├── persistence.py             # JSON-based data store for speakers/recordings
│   └── utils.py                   # Shared utilities (device detection, filename sanitization)
├── scripts/
│   ├── enroll.py                  # CLI to enroll a speaker from audio file or mic recording
│   └── extract.py                 # CLI to run offline target speech extraction on a mixture
├── weights/                       # Model checkpoints (.ckpt files)
├── media/                         # Audio files (enrollments, mixtures, extracted outputs)
└── environment.yml                # Conda environment specification
```

## Setup

```bash
conda env create -f environment.yml
conda activate hearpoint-realtime
```

Model checkpoints (`tfgridnet.ckpt` and `tfgridnet_enroll.ckpt`) should be placed in the `weights/` directory.

## Usage

### Enroll a Speaker

Create a speaker embedding from an audio file or live recording:

```bash
python scripts/enroll.py --name "Alice" --audio /path/to/recording.wav
python scripts/enroll.py --name "Alice" --record --duration 5
```

### Offline Extraction

Extract a target speaker from a binaural mixture:

```bash
python scripts/extract.py --audio /path/to/mixture.wav --speaker "Alice"
python scripts/extract.py --audio /path/to/mixture.wav --speaker "Alice" --speaker "Bob"
```

### Real-Time Inference

Run streaming target speech extraction from a live microphone:

```bash
cd src/realtime
python realtime_inference.py --list-devices          # find your audio device indices
python realtime_inference.py                          # run with config.yaml settings
python realtime_inference.py --embedding speaker.npy  # override embedding via CLI
python realtime_inference.py --test-file input.wav    # validate with a pre-recorded file
```

Configuration is managed via `src/realtime/config.yaml`. See [`src/realtime/README.md`](src/realtime/README.md) for full documentation on configuration, debugging, and performance tuning.
