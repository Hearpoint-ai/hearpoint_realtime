# Real-Time TFGridNet Inference

This module provides real-time target speech extraction using the TFGridNet model. It captures audio from a microphone, processes it through the neural network in real-time, and outputs the enhanced audio to headphones.

## Requirements

### Python Dependencies

```bash
pip install sounddevice numpy torch soundfile resampy pyyaml
```

### System Requirements

- **macOS**: PortAudio (installed via Homebrew: `brew install portaudio`)
- **Linux**: ALSA or PulseAudio (usually pre-installed)
- **Windows**: No additional requirements

### Hardware

- Microphone (mono supported)
- Headphones/speakers (stereo output)
- Recommended: Low-latency audio interface for best results

## Quick Start

### 1. List Available Audio Devices

```bash
cd backend/test_realtime
python realtime_inference.py --list-devices
```

This shows all available audio input/output devices with their indices.

### 2. Configure Your Setup

Edit `config.yaml` with your settings:

```yaml
# Speaker embedding (required)
embedding: /path/to/speaker_embedding.npy

# Audio devices
audio:
  input_device: 3   # Your microphone device index
  output_device: 2  # Your headphone/speaker device index
```

### 3. Run Real-Time Inference

```bash
python realtime_inference.py
```

### 4. Test with Pre-Recorded Audio

To validate the streaming inference without a live microphone:

```bash
python realtime_inference.py --test-file input.wav
```

## Configuration

All runtime parameters are configured via `config.yaml`. See the file for detailed documentation of each option.

| Section | Parameters |
|---------|------------|
| `embedding` | Path to speaker embedding .npy file (required) |
| `audio` | sample_rate, chunk_size, input/output devices, channels, buffer_size |
| `hrtf` | Enable binaural synthesis with Head-Related Transfer Functions |
| `debug` | Verbose logging, passthrough mode, debug file saving |

## How It Works

### Architecture Overview

```
                        ┌─────────────────────────────────────────┐
                        │           Processing Thread             │
┌──────────┐            │  ┌───────────────────────────────────┐  │           ┌──────────┐
│   Mono   │  Input     │  │         TFGridNet Model           │  │  Output   │  Stereo  │
│   Mic    │──Queue────▶│  │  ┌─────┐  ┌─────┐  ┌─────┐       │  │──Queue───▶│ Headphone│
│          │            │  │  │STFT │─▶│Grid │─▶│iSTFT│       │  │           │          │
└──────────┘            │  │  │     │  │Blocks│  │     │       │  │           └──────────┘
                        │  │  └─────┘  └─────┘  └─────┘       │  │
                        │  │           ▲                       │  │
                        │  │           │ Speaker Embedding     │  │
                        │  └───────────┴───────────────────────┘  │
                        │              ▲                          │
                        │              │ State Buffers            │
                        │              │ (persisted across chunks)│
                        └──────────────┴──────────────────────────┘
```

### Audio Processing Pipeline

1. **Audio Capture**: Microphone audio is captured in chunks and placed in an input queue
2. **Mono-to-Stereo**: Mono input is duplicated to create a 2-channel signal (required by the model)
3. **STFT Encoding**: Audio is converted to time-frequency representation
4. **GridNet Processing**: Neural network processes the spectrogram with speaker embedding conditioning
5. **iSTFT Decoding**: Frequency domain output is converted back to time domain
6. **Stereo Output**: Enhanced audio is sent to the output queue and played through headphones

### Streaming State Management

The TFGridNet model maintains several state buffers across chunks for causal processing:

| Buffer | Purpose |
|--------|---------|
| `conv_buf` | Temporal context for input convolution (kernel_size-1 frames) |
| `deconv_buf` | Temporal context for output deconvolution |
| `istft_buf` | Overlap-add buffer for inverse STFT reconstruction |
| `gridnet_bufs` | Per-layer state including LSTM hidden/cell states and attention K/V cache |

These buffers enable the model to process audio in small chunks while maintaining temporal coherence.

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 16 kHz | Audio sample rate |
| Chunk Size | 128 samples | 8 ms per chunk |
| STFT Window | 192 samples | 128 chunk + 64 lookahead |
| Local Attention | 50 frames | ~400 ms context window |
| Embedding Dim | 256 | Speaker embedding dimensions |

### Latency Analysis

The system introduces several sources of latency:

1. **Audio Input Buffering**: ~32 ms (4 chunks for stability)
2. **Processing**: Variable (depends on hardware, typically 1-5 ms per chunk on GPU)
3. **STFT Lookahead**: 64 samples (4 ms)
4. **Audio Output Buffering**: ~8 ms (1 chunk)

**Total Expected Latency**: ~40-50 ms (acceptable for hearing aid applications)

## Performance Tuning

### Real-Time Factor (RTF)

The script reports RTF (Real-Time Factor) during processing:
- **RTF < 1.0**: Processing is faster than real-time (good)
- **RTF > 1.0**: Processing is slower than real-time (audio will be delayed/choppy)

### Tips for Better Performance

1. **Use GPU**: CUDA or MPS provides significant speedup (use `--device cuda` or `--device mps`)

2. **Adjust Buffer Size**: Modify `buffer_size_chunks` in config (higher = more stable, more latency)
   ```yaml
   audio:
     buffer_size_chunks: 4  # default, adjust as needed
   ```

3. **Close Other Applications**: Ensure CPU/GPU resources are available

4. **Use Low-Latency Audio Interface**: Professional audio interfaces typically have lower latency than built-in sound cards

## Creating Speaker Embeddings

Speaker embeddings are required to tell the model whose voice to extract. You can create them using the existing enrollment system:

### Option 1: Using the Backend API

If the FastAPI server is running:

```python
import requests

# Upload enrollment audio
with open("enrollment_audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/enroll",
        files={"audio": f},
        data={"speaker_name": "John"}
    )

# The embedding will be saved and can be found in the response
embedding_path = response.json()["embedding_path"]
```

### Option 2: Direct Embedding Extraction

```python
import numpy as np
from app.ml.TFGridNetSpeakerEmbeddingModel import TFGridNetSpeakerEmbeddingModel

# Load model
model = TFGridNetSpeakerEmbeddingModel()

# Compute embedding from audio file
embedding = model.compute_embedding("enrollment_audio.wav")

# Save embedding
np.save("speaker_embedding.npy", embedding)
```

## Troubleshooting

### "No input/output device found"

Run `--list-devices` to see available devices and update `config.yaml`:
```bash
python realtime_inference.py --list-devices
```

### Audio glitches or dropouts

- Increase `buffer_size_chunks` in config.yaml
- Check CPU/GPU utilization
- Try a different audio device

### High latency

- Use GPU acceleration (`--device cuda` or `--device mps`)
- Reduce other system load
- Check that RTF is below 1.0

### Import errors

Ensure you're running from the correct directory:
```bash
cd backend/test_realtime
python realtime_inference.py --help
```

## Debugging

If you hear static, buzzing, or no audio, use these debug modes to isolate the problem.

### Step 1: Test Passthrough Mode

Bypass the model entirely to verify audio I/O works. Set in config.yaml:

```yaml
debug:
  passthrough: true
```

**Expected results:**
- You hear your voice clearly → Audio I/O is working, problem is in the model
- Still static/no audio → Problem is in audio device configuration

### Step 2: Enable Debug Logging

See detailed input/output statistics for each audio chunk. Set in config.yaml:

```yaml
debug:
  verbose: true
```

This prints for each chunk:
- Input shape, min/max/mean values, NaN detection
- Output shape, min/max/mean values, NaN detection
- Queue sizes (input/output/accumulator)

**What to look for:**
- Input values near zero → Microphone not capturing
- Output contains NaN → Model producing invalid values
- Output min/max are extreme (e.g., >1000) → Model producing garbage
- Queue sizes growing unbounded → Processing too slow

### Step 3: Save Debug Audio Files

Save raw input and model output for offline inspection. Set in config.yaml:

```yaml
debug:
  save_dir: ./debug_audio/
```

This saves:
- `debug_audio/debug_input.wav` - Raw microphone capture
- `debug_audio/debug_output.wav` - Model output before playback

Open these files in Audacity or another audio editor to inspect:
- Is the input capturing your voice correctly?
- What does the model output look like? (Waveform, spectrogram)

### Input Level Meter

The stats output includes a real-time input level meter:

```
Chunks:   1234 | Avg:  2.50ms | Max:  5.00ms | RTF: 0.312 | Queue: 2/3 | Level: [========                      ] -35.2dB
```

**Interpreting the level:**
- `-60 dB` to `-40 dB`: Very quiet (silence or distant sound)
- `-40 dB` to `-20 dB`: Normal speech level
- `-20 dB` to `0 dB`: Loud (close to clipping)

If the level stays at `-60 dB` while speaking, the microphone isn't capturing audio.

### Common Debug Scenarios

| Symptom | Config Setting | What to Check |
|---------|----------------|---------------|
| Static/buzzing | `debug.passthrough: true` | If still broken, check audio devices |
| Silence | `debug.verbose: true` | Check if input values are near zero |
| Distorted audio | `debug.save_dir: ./debug/` | Inspect output waveform for clipping |
| Choppy audio | `debug.verbose: true` | Check RTF > 1.0 or growing queue sizes |
| Model crash | `debug.verbose: true` | Look for NaN values in output |

## File Structure

```
test_realtime/
├── README.md              # This documentation
├── config.yaml            # Configuration file
├── realtime_inference.py  # Main real-time inference script
└── __init__.py           # Package marker
```

## Technical Details

### Model Configuration

The model uses these default parameters from `configs/tfgridnet_cipic.json`:

```json
{
    "embed_dim": 256,
    "stft_chunk_size": 128,
    "stft_pad_size": 64,
    "num_ch": 2,
    "D": 64,
    "B": 3,
    "H": 64,
    "L": 4,
    "use_attn": true,
    "lookahead": true,
    "chunk_causal": true,
    "local_atten_len": 50
}
```

### Thread Model

```
Main Thread                 Processing Thread
    │                            │
    ▼                            ▼
┌────────────┐              ┌────────────┐
│ Audio I/O  │              │  Neural    │
│ Callbacks  │───Queues────▶│  Network   │
│            │◀─────────────│ Inference  │
└────────────┘              └────────────┘
```

- **Main Thread**: Handles audio I/O callbacks and user interface
- **Processing Thread**: Runs neural network inference in background
- **Queues**: Thread-safe buffers for audio data exchange

This design ensures that audio I/O is never blocked by potentially slow neural network inference.
