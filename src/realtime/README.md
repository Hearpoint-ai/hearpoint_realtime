# Real-time TFGridNet Inference

This module runs the TFGridNet target speech extraction model in real-time. It captures audio from a microphone, processes it through the model, and outputs enhanced audio to headphones — isolating a target speaker based on a pre-enrolled speaker embedding.

## Usage

All settings are configured in `config.yaml`. Edit that file before running:

```bash
python realtime_inference.py
```

### Common CLI Options

```bash
# List available audio devices (use this to find device indices for config.yaml)
python realtime_inference.py --list-devices

# Override the torch device
python realtime_inference.py --device mps
```

## Configuration (`config.yaml`)

All settings should be configured in `config.yaml`. The key sections are:

### `embedding` (required)

Path to a `.npy` speaker embedding file. This is the enrollment vector for the target speaker you want to isolate. Relative paths are resolved from the repository root.

```yaml
embedding: media/enrollments/speaker.npy
```

### `audio`

| Setting              | Default | Description                                                    |
| -------------------- | ------- | -------------------------------------------------------------- |
| `sample_rate`        | 16000   | Sample rate in Hz. Must match the model's expected rate.       |
| `chunk_size`         | 128     | Samples per processing chunk (128 = 8ms at 16kHz).            |
| `input_device`       | null    | Input device index. Run `--list-devices` to find yours.       |
| `output_device`      | null    | Output device index. `null` uses the system default.           |
| `input_channels`     | 2       | Number of input channels from the microphone.                  |
| `output_channels`    | 2       | Output channels. `null` to auto-detect from the output device. |
| `buffer_size_chunks` | 4       | Number of chunks to buffer on input. Higher = more stable but more latency. |

### `debug`

| Setting       | Default | Description                                                        |
| ------------- | ------- | ------------------------------------------------------------------ |
| `verbose`     | false   | Print per-chunk input/output stats (levels, shapes, NaN checks).   |
| `passthrough` | false   | Bypass the model entirely and pass audio through unchanged. Useful for testing the audio I/O pipeline. |
| `save_dir`    | null    | Directory to save `debug_input.wav` and `debug_output.wav` after a session. Set to a path like `./debug_audio/` to enable. |

### `test`

File-based test mode processes a pre-recorded audio file chunk-by-chunk, simulating real-time behavior. This is useful for validating model output without needing a live microphone.

| Setting       | Default | Description                                              |
| ------------- | ------- | -------------------------------------------------------- |
| `enabled`     | false   | Set to `true` to run in file-based test mode.            |
| `input_file`  | null    | Path to the input audio file.                            |
| `output_file` | null    | Path for the output file. `null` auto-generates a name.  |

## How the Script Works

1. **Configuration loading** — Settings are loaded from `config.yaml`, with any CLI arguments applied as overrides. Relative paths are resolved from the repository root.

2. **Model initialization** — The TFGridNet model is loaded from a checkpoint (`weights/tfgridnet.ckpt`) using architecture parameters from a JSON config (`src/configs/tfgridnet_cipic.json`). The model is set to eval mode on the detected device (CPU, CUDA, or MPS). Internal state buffers are initialized for streaming inference.

3. **Speaker embedding** — A pre-computed speaker embedding (`.npy`) is loaded. This vector tells the model which speaker to isolate from the mixture.

4. **Audio streaming** (real-time mode) — Two `sounddevice` streams are opened:
   - An **input stream** captures audio from the microphone and pushes chunks into an input queue.
   - An **output stream** pulls processed chunks from an output queue and plays them through headphones.
   - A **background processing thread** sits between the two queues. It accumulates input samples, and once enough are available (chunk + lookahead), runs inference and pushes the result to the output queue.

5. **Chunk processing** — Each chunk goes through:
   - Mono-to-stereo duplication (the model expects binaural input).
   - Forward pass through `model.predict()` with the speaker embedding and cached state. The state is updated after each call, enabling continuous streaming without re-processing prior audio.
   - Lookahead: the next `stft_pad_size` samples are passed as lookahead audio to reduce boundary artifacts.
   - Output is clipped to `[-1, 1]` and placed in the output queue.

6. **Monitoring** — Every second, the main thread prints statistics: chunks processed, average/max processing time, real-time factor (RTF), queue sizes, and an input level meter in dB.

7. **File-based test mode** — When `test.enabled` is `true` or `--test-file` is provided, the script reads an audio file, processes it chunk-by-chunk (with lookahead) through the same model pipeline, and writes the enhanced output to a file.
