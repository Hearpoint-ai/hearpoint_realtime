#!/usr/bin/env python3
"""
Real-time TFGridNet inference for target speech extraction.

This script captures audio from a microphone, processes it through the TFGridNet
model in real-time, and outputs the enhanced audio to headphones.

Supports:
- Mono microphone input (duplicated to stereo for the model)
- Stereo headphone output
- Continuous streaming with state caching
- YAML configuration file for easy parameter management
"""

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import resampy
import yaml

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.models.tfgridnet_realtime.net import Net
from app.utils import get_torch_device


# Default paths
DEFAULT_CONFIG_PATH = BACKEND_DIR / "configs" / "tfgridnet_cipic.json"
DEFAULT_CHECKPOINT_PATH = BACKEND_DIR / "weights" / "tfgridnet.ckpt"
DEFAULT_HRTF_LEFT_PATH = BACKEND_DIR / "data" / "hrtf" / "cipic_left.wav"
DEFAULT_HRTF_RIGHT_PATH = BACKEND_DIR / "data" / "hrtf" / "cipic_right.wav"
DEFAULT_YAML_CONFIG_PATH = SCRIPT_DIR / "config.yaml"

# Audio parameters (defaults, can be overridden by config)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 128  # 8ms at 16kHz - matches model's stft_chunk_size


@dataclass
class ModelConfig:
    """Model-related configuration."""
    embedding: Path | None = None
    checkpoint: Path | None = None
    config: Path | None = None
    device: str | None = None


@dataclass
class AudioConfig:
    """Audio-related configuration."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    input_device: int | None = None
    output_device: int | None = None
    input_channels: int = 2
    output_channels: int | None = 2
    buffer_size_chunks: int = 4
    crossfade_samples: int = 16  # ~1ms at 16kHz for smoothing chunk boundaries


@dataclass
class HRTFConfig:
    """HRTF-related configuration."""
    enabled: bool = False
    left_path: Path | None = None
    right_path: Path | None = None


@dataclass
class DebugConfig:
    """Debug-related configuration."""
    verbose: bool = False
    passthrough: bool = False
    save_dir: Path | None = None


@dataclass
class OptimizationConfig:
    """Performance optimization configuration."""
    precision: str = "fp32"  # "fp32", "fp16", or "bf16"
    use_torch_compile: bool = False


@dataclass
class TestConfig:
    """File-based test configuration."""
    input_file: Path | None = None
    output_file: Path | None = None


@dataclass
class Config:
    """Complete configuration for real-time inference."""
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    hrtf: HRTFConfig = field(default_factory=HRTFConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    test: TestConfig = field(default_factory=TestConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from a YAML file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with yaml_path.open() as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from a dictionary."""
        def to_path(val: Any) -> Path | None:
            return Path(val) if val else None

        audio_data = data.get("audio", {}) or {}
        hrtf_data = data.get("hrtf", {}) or {}
        debug_data = data.get("debug", {}) or {}

        # Load embedding path from top-level config
        embedding_path = to_path(data.get("embedding"))

        return cls(
            model=ModelConfig(embedding=embedding_path),
            audio=AudioConfig(
                sample_rate=audio_data.get("sample_rate", DEFAULT_SAMPLE_RATE),
                chunk_size=audio_data.get("chunk_size", DEFAULT_CHUNK_SIZE),
                input_device=audio_data.get("input_device", 5),
                output_device=audio_data.get("output_device", 4),
                input_channels=audio_data.get("input_channels", 2),
                output_channels=audio_data.get("output_channels", 2),
                buffer_size_chunks=audio_data.get("buffer_size_chunks", 4),
                crossfade_samples=audio_data.get("crossfade_samples", 16),
            ),
            hrtf=HRTFConfig(
                enabled=hrtf_data.get("enabled", False),
                left_path=to_path(hrtf_data.get("left_path")),
                right_path=to_path(hrtf_data.get("right_path")),
            ),
            debug=DebugConfig(
                verbose=debug_data.get("verbose", False),
                passthrough=debug_data.get("passthrough", False),
                save_dir=to_path(debug_data.get("save_dir")),
            ),
            test=TestConfig(),
        )

    def get_checkpoint_path(self) -> Path:
        """Get checkpoint path with default fallback."""
        return self.model.checkpoint or DEFAULT_CHECKPOINT_PATH

    def get_model_config_path(self) -> Path:
        """Get model config path with default fallback."""
        return self.model.config or DEFAULT_CONFIG_PATH

    def get_hrtf_left_path(self) -> Path:
        """Get HRTF left path with default fallback."""
        return self.hrtf.left_path or DEFAULT_HRTF_LEFT_PATH

    def get_hrtf_right_path(self) -> Path:
        """Get HRTF right path with default fallback."""
        return self.hrtf.right_path or DEFAULT_HRTF_RIGHT_PATH


class RealtimeInference:
    """Real-time TFGridNet inference engine."""

    def __init__(self, config: Config):
        """
        Initialize real-time inference engine.

        Args:
            config: Configuration object containing all parameters.
        """
        self.config = config
        self.sample_rate = config.audio.sample_rate
        self.chunk_size = config.audio.chunk_size
        self.input_channels = config.audio.input_channels
        self.input_device = config.audio.input_device
        self.output_device = config.audio.output_device
        self.buffer_size_chunks = config.audio.buffer_size_chunks

        # Auto-detect output channels from device if not specified
        if config.audio.output_channels is None:
            self.output_channels = self._detect_output_channels(self.output_device)
        else:
            self.output_channels = config.audio.output_channels

        # Debug options
        self.passthrough_mode = config.debug.passthrough
        self.debug = config.debug.verbose
        self.save_debug_dir = config.debug.save_dir
        if self.save_debug_dir:
            self.save_debug_dir.mkdir(parents=True, exist_ok=True)
            self.debug_inputs = []
            self.debug_outputs = []

        # Load HRTFs for mono-to-binaural conversion
        self.hrtf_left = None
        self.hrtf_right = None
        self.hrtf_conv_buffer = None
        self.use_hrtf = config.hrtf.enabled
        if self.use_hrtf:
            hrtf_left_path = config.get_hrtf_left_path()
            hrtf_right_path = config.get_hrtf_right_path()
            if hrtf_left_path.exists() and hrtf_right_path.exists():
                self._load_hrtfs(hrtf_left_path, hrtf_right_path)
            else:
                print(f"Warning: HRTF files not found, disabling HRTF processing")
                self.use_hrtf = False
        if not self.use_hrtf:
            print("HRTF disabled - using simple channel duplication")

        # Threading control
        self.running = False
        self.input_queue = queue.Queue(maxsize=64)
        self.output_queue = queue.Queue(maxsize=64)

        # For input level monitoring
        self.recent_input_level = 0.0
        self.input_level_lock = threading.Lock()

        # Set up device
        self.device = torch.device(config.model.device) if config.model.device else get_torch_device()
        print(f"Using device: {self.device}")

        if self.passthrough_mode:
            print("*** PASSTHROUGH MODE - bypassing model ***")
            self.model = None
            self.embedding = None
            self.state = None
            self.stft_pad_size = 0
        else:
            # Load model
            self._load_model(config.get_checkpoint_path(), config.get_model_config_path())

            # Load speaker embedding
            if config.model.embedding is None:
                raise ValueError("Speaker embedding path is required")
            self._load_embedding(config.model.embedding)

            # Initialize model state buffers
            self.state = self.model.init_buffers(batch_size=1, device=self.device)
            self.stft_pad_size = self.model.stft_pad_size

        # Input accumulator for collecting enough samples before processing
        self.input_accumulator = np.zeros(0, dtype=np.float32)

        # Crossfade buffer for smoothing chunk boundaries
        self.crossfade_samples = config.audio.crossfade_samples
        self.prev_output_tail = None  # Will store last crossfade_samples of previous output

        # Statistics
        self.chunks_processed = 0
        self.processing_times = []

    def _load_model(self, checkpoint_path: Path, config_path: Path) -> None:
        """Load the TFGridNet model from checkpoint."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load config
        with config_path.open() as fp:
            config = json.load(fp)
        model_params = config.get("pl_module_args", {}).get("model_params", {})

        print(f"Model parameters: {model_params}")

        # Create model
        self.model = Net(**model_params).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint

        # Strip common wrapper prefixes
        prefixes = ("model.model.", "model.", "module.")
        for pref in prefixes:
            if all(k.startswith(pref) for k in state_dict.keys()):
                state_dict = {k[len(pref):]: v for k, v in state_dict.items()}
                break

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected}")

        self.model.eval()
        print(f"Model loaded from {checkpoint_path}")

    def _load_embedding(self, embedding_path: Path) -> None:
        """Load speaker embedding from .npy file."""
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding not found: {embedding_path}")

        embedding = np.load(embedding_path)
        # Shape: [1, 1, embed_dim] for batch processing
        embedding = embedding.astype(np.float32).reshape(1, 1, -1)
        self.embedding = torch.from_numpy(embedding).to(self.device)
        print(f"Loaded embedding from {embedding_path}, shape: {embedding.shape}")

    def _detect_output_channels(self, output_device: int | str | None) -> int:
        """Detect the maximum number of output channels supported by the device."""
        try:
            if output_device is not None:
                device_info = sd.query_devices(output_device)
            else:
                device_info = sd.query_devices(kind='output')

            max_channels = device_info.get('max_output_channels', 2)
            # Prefer stereo if available, otherwise use what's supported
            channels = min(2, max_channels)
            print(f"Output device supports {max_channels} channels, using {channels}")
            return channels
        except Exception as e:
            print(f"Warning: Could not detect output channels ({e}), defaulting to 1")
            return 1

    def _load_hrtfs(self, left_path: Path, right_path: Path) -> None:
        """Load HRTF impulse responses for binaural synthesis."""
        def load_hrtf(path: Path) -> np.ndarray:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"HRTF file not found: {path}")
            if path.suffix.lower() == ".npy":
                data = np.load(path)
                sr = self.sample_rate
            else:
                data, sr = sf.read(str(path))
            if data.ndim > 1:
                data = data[:, 0]
            # Resample if needed
            if sr != self.sample_rate:
                data = resampy.resample(data, sr, self.sample_rate)
            return data.astype(np.float32)

        self.hrtf_left = load_hrtf(left_path)
        self.hrtf_right = load_hrtf(right_path)

        # Initialize convolution buffer (overlap-save method)
        # Buffer length = max HRTF length - 1
        max_hrtf_len = max(len(self.hrtf_left), len(self.hrtf_right))
        self.hrtf_conv_buffer = np.zeros(max_hrtf_len - 1, dtype=np.float32)

        print(f"Loaded HRTFs: left={len(self.hrtf_left)} samples, right={len(self.hrtf_right)} samples")

    def _apply_hrtf_streaming(self, mono_chunk: np.ndarray) -> np.ndarray:
        """
        Apply HRTF convolution to mono chunk using overlap-save for streaming.

        Returns: [2, chunk_size] binaural audio
        """
        if self.hrtf_left is None or self.hrtf_right is None:
            # Fallback to simple duplication
            return np.stack([mono_chunk, mono_chunk], axis=0)

        # Prepend buffer from previous chunk for continuous convolution
        extended = np.concatenate([self.hrtf_conv_buffer, mono_chunk])

        # Convolve with both HRTFs
        left_full = np.convolve(extended, self.hrtf_left, mode='full')
        right_full = np.convolve(extended, self.hrtf_right, mode='full')

        # Extract the valid portion (skip transient, take chunk_size samples)
        # For overlap-save: valid output starts at (hrtf_len - 1) and has length = chunk_size
        buf_len = len(self.hrtf_conv_buffer)
        left_out = left_full[buf_len:buf_len + len(mono_chunk)]
        right_out = right_full[buf_len:buf_len + len(mono_chunk)]

        # Update buffer with the tail of the current chunk
        self.hrtf_conv_buffer = mono_chunk[-buf_len:].copy() if buf_len > 0 else np.array([])

        return np.stack([left_out, right_out], axis=0).astype(np.float32)

    def _mono_to_stereo(self, mono: np.ndarray) -> np.ndarray:
        """Convert mono audio to binaural stereo using HRTF or simple duplication."""
        # Input: [samples] or [samples, 1]
        if mono.ndim > 1:
            mono = mono.squeeze()

        # Use HRTF convolution if available for proper binaural synthesis
        if self.hrtf_left is not None and self.hrtf_right is not None:
            return self._apply_hrtf_streaming(mono)

        # Fallback: Output: [2, samples] (channels first for model)
        return np.stack([mono, mono], axis=0)

    def _process_chunk(self, audio_chunk: np.ndarray, lookahead: np.ndarray | None = None) -> np.ndarray:
        """
        Process a single audio chunk through the model.

        Args:
            audio_chunk: Mono audio chunk [chunk_size]
            lookahead: Optional mono lookahead samples [stft_pad_size] for real
                       audio lookahead instead of zero-padding.

        Returns:
            Stereo enhanced audio [chunk_size, 2]
        """
        start_time = time.perf_counter()

        # Debug: log input stats
        if self.debug:
            has_nan = np.isnan(audio_chunk).any()
            print(f"Input: shape={audio_chunk.shape}, min={audio_chunk.min():.4f}, "
                  f"max={audio_chunk.max():.4f}, mean={audio_chunk.mean():.4f}, nan={has_nan}")

        # Update input level for monitoring
        with self.input_level_lock:
            self.recent_input_level = np.abs(audio_chunk).max()

        # Passthrough mode: bypass model entirely
        if self.passthrough_mode:
            if self.output_channels == 1:
                output_audio = audio_chunk.reshape(-1, 1)
            else:
                output_audio = np.column_stack([audio_chunk, audio_chunk])
            elapsed = time.perf_counter() - start_time
            self.processing_times.append(elapsed)
            self.chunks_processed += 1

            if self.debug:
                print(f"Output (passthrough): shape={output_audio.shape}")

            # Save debug files
            if self.save_debug_dir:
                self.debug_inputs.append(audio_chunk.copy())
                self.debug_outputs.append(output_audio.copy())

            return output_audio

        # Convert mono to stereo [2, chunk_size]
        stereo_input = self._mono_to_stereo(audio_chunk)

        # Convert to tensor [1, 2, chunk_size]
        input_tensor = torch.from_numpy(stereo_input).unsqueeze(0).to(self.device)

        # Prepare lookahead tensor if available (real audio instead of zero-padding)
        la_tensor = None
        if lookahead is not None and len(lookahead) > 0:
            if self.use_hrtf and self.hrtf_left is not None:
                # Save HRTF state -- lookahead must not permanently advance it
                saved_hrtf_buf = self.hrtf_conv_buffer.copy()
                stereo_la = self._apply_hrtf_streaming(lookahead)
                self.hrtf_conv_buffer = saved_hrtf_buf
            else:
                stereo_la = np.stack([lookahead, lookahead], axis=0)
            la_tensor = torch.from_numpy(stereo_la).unsqueeze(0).to(self.device)

        # Run inference with state caching
        with torch.no_grad():
            output, self.state = self.model.predict(
                input_tensor,
                self.embedding[:, 0],  # [B, embed_dim]
                self.state,
                pad=True,
                lookahead_audio=la_tensor
            )

        # Convert output to numpy [chunk_size, 2]
        output_audio = output.squeeze(0).cpu().numpy()  # [2, samples]

        # Debug: log output stats before clipping
        if self.debug:
            has_nan = np.isnan(output_audio).any()
            print(f"Output: shape={output_audio.shape}, min={output_audio.min():.4f}, "
                  f"max={output_audio.max():.4f}, mean={output_audio.mean():.4f}, nan={has_nan}")

        output_audio = np.clip(output_audio, -1.0, 1.0)
        output_audio = output_audio.T  # [samples, 2]

        # Apply crossfade to smooth chunk boundaries (skip if crossfade_samples is 0)
        if self.crossfade_samples > 0:
            if self.prev_output_tail is not None and len(output_audio) >= self.crossfade_samples:
                # Create crossfade weights
                fade_in = np.linspace(0, 1, self.crossfade_samples, dtype=np.float32).reshape(-1, 1)
                fade_out = 1.0 - fade_in
                # Blend the overlap region
                output_audio[:self.crossfade_samples] = (
                    fade_out * self.prev_output_tail +
                    fade_in * output_audio[:self.crossfade_samples]
                )

            # Store tail for next chunk's crossfade
            if len(output_audio) >= self.crossfade_samples:
                self.prev_output_tail = output_audio[-self.crossfade_samples:].copy()

        # Convert to mono if needed
        if self.output_channels == 1:
            output_audio = output_audio.mean(axis=1, keepdims=True)

        # Save debug files
        if self.save_debug_dir:
            self.debug_inputs.append(audio_chunk.copy())
            self.debug_outputs.append(output_audio.copy())

        # Record processing time
        elapsed = time.perf_counter() - start_time
        self.processing_times.append(elapsed)
        self.chunks_processed += 1

        return output_audio

    def _input_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream."""
        if status:
            print(f"Input status: {status}")

        if self.running:
            # Add to input queue
            try:
                self.input_queue.put_nowait(indata.copy())
            except queue.Full:
                print("Warning: Input queue full, dropping audio")

    def _output_callback(self, outdata, frames, time_info, status):
        """Callback for audio output stream."""
        if status:
            print(f"Output status: {status}")

        try:
            data = self.output_queue.get_nowait()

            if self.debug:
                print(f"Output callback: frames={frames}, data.shape={data.shape}, "
                      f"data min={data.min():.4f}, max={data.max():.4f}")

            # Ensure correct shape
            if data.shape[0] < frames:
                # Pad if needed
                padding = np.zeros((frames - data.shape[0], self.output_channels), dtype=np.float32)
                data = np.vstack([data, padding])
            outdata[:] = data[:frames]
        except queue.Empty:
            # Output silence if no data available
            if self.debug:
                print(f"Output callback: UNDERRUN (queue empty), frames={frames}")
            outdata.fill(0)

    def _processing_thread(self):
        """Background thread for processing audio chunks."""
        while self.running:
            try:
                # Get input audio (blocking with timeout)
                indata = self.input_queue.get(timeout=0.1)

                if self.debug:
                    print(f"Queue sizes - input: {self.input_queue.qsize()}, output: {self.output_queue.qsize()}, "
                          f"accumulator: {len(self.input_accumulator)}")

                # Accumulate input samples
                mono_input = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                self.input_accumulator = np.concatenate([self.input_accumulator, mono_input])

                # Process complete chunks (need chunk + lookahead samples)
                required_samples = self.chunk_size + self.stft_pad_size

                while len(self.input_accumulator) >= required_samples:
                    # Extract chunk and lookahead
                    chunk = self.input_accumulator[:self.chunk_size].astype(np.float32)
                    lookahead = self.input_accumulator[self.chunk_size:required_samples].astype(np.float32)
                    # Advance by chunk_size only -- lookahead rolls into next chunk
                    self.input_accumulator = self.input_accumulator[self.chunk_size:]

                    # Process through model with real lookahead audio
                    output = self._process_chunk(chunk, lookahead)

                    # Add to output queue
                    try:
                        # If queue is getting too full, drop oldest items to prevent latency buildup
                        while self.output_queue.qsize() > 10:
                            try:
                                self.output_queue.get_nowait()
                                if self.debug:
                                    print("Warning: Dropping old output chunk to reduce latency")
                            except queue.Empty:
                                break
                        self.output_queue.put_nowait(output)
                    except queue.Full:
                        print("Warning: Output queue full, dropping audio")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                import traceback
                traceback.print_exc()

    def list_devices(self):
        """List available audio devices."""
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        print(f"\nDefault input device: {sd.default.device[0]}")
        print(f"Default output device: {sd.default.device[1]}")

    def run(self):
        """Start real-time processing."""
        print(f"\nStarting real-time inference...")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Chunk size: {self.chunk_size} samples ({self.chunk_size / self.sample_rate * 1000:.1f} ms)")
        print(f"  Input channels: {self.input_channels}")
        print(f"  Output channels: {self.output_channels}")
        print(f"  Input device: {self.input_device or 'default'}")
        print(f"  Output device: {self.output_device or 'default'}")

        self.running = True

        # Pre-fill output queue with silence to prevent initial underruns
        silence = np.zeros((self.chunk_size, self.output_channels), dtype=np.float32)
        for _ in range(self.buffer_size_chunks * 2):
            self.output_queue.put(silence)

        # Start processing thread
        process_thread = threading.Thread(target=self._processing_thread, daemon=True)
        process_thread.start()

        # Configure streams
        # Use slightly larger block size for input to reduce callback overhead
        input_blocksize = self.chunk_size * self.buffer_size_chunks
        output_blocksize = self.chunk_size

        try:
            with sd.InputStream(
                device=self.input_device,
                samplerate=self.sample_rate,
                channels=self.input_channels,
                dtype=np.float32,
                blocksize=input_blocksize,
                callback=self._input_callback,
            ), sd.OutputStream(
                device=self.output_device,
                samplerate=self.sample_rate,
                channels=self.output_channels,
                dtype=np.float32,
                blocksize=output_blocksize,
                callback=self._output_callback,
            ):
                print("\nProcessing... Press Ctrl+C to stop.\n")

                # Main loop - print stats periodically
                while self.running:
                    time.sleep(1.0)
                    if self.processing_times:
                        avg_time = np.mean(self.processing_times[-100:]) * 1000
                        max_time = np.max(self.processing_times[-100:]) * 1000
                        chunk_duration = self.chunk_size / self.sample_rate * 1000
                        rtf = avg_time / chunk_duration

                        # Get input level for visualization
                        with self.input_level_lock:
                            level = self.recent_input_level

                        # Convert to dB and create visual meter
                        level_db = 20 * np.log10(level + 1e-10)
                        bars = int(max(0, min(30, (level_db + 60) / 2)))  # -60dB to 0dB range
                        level_meter = f"[{'=' * bars}{' ' * (30 - bars)}] {level_db:5.1f}dB"

                        print(f"Chunks: {self.chunks_processed:6d} | "
                              f"Avg: {avg_time:5.2f}ms | "
                              f"Max: {max_time:5.2f}ms | "
                              f"RTF: {rtf:.3f} | "
                              f"Queue: {self.input_queue.qsize()}/{self.output_queue.qsize()} | "
                              f"Level: {level_meter}")

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            process_thread.join(timeout=1.0)

        # Save debug files if requested
        if self.save_debug_dir and self.debug_inputs:
            self._save_debug_files()

        # Print final statistics
        if self.processing_times:
            print(f"\nFinal Statistics:")
            print(f"  Total chunks processed: {self.chunks_processed}")
            print(f"  Average processing time: {np.mean(self.processing_times) * 1000:.2f} ms")
            print(f"  Max processing time: {np.max(self.processing_times) * 1000:.2f} ms")
            print(f"  Min processing time: {np.min(self.processing_times) * 1000:.2f} ms")

    def _save_debug_files(self):
        """Save accumulated debug audio to files."""
        try:
            import soundfile as sf

            if self.debug_inputs:
                input_audio = np.concatenate(self.debug_inputs)
                input_path = self.save_debug_dir / "debug_input.wav"
                sf.write(str(input_path), input_audio, self.sample_rate)
                print(f"Saved debug input: {input_path} ({len(input_audio) / self.sample_rate:.1f}s)")

            if self.debug_outputs:
                output_audio = np.concatenate(self.debug_outputs)
                output_path = self.save_debug_dir / "debug_output.wav"
                sf.write(str(output_path), output_audio, self.sample_rate)
                print(f"Saved debug output: {output_path} ({len(output_audio) / self.sample_rate:.1f}s)")

        except Exception as e:
            print(f"Error saving debug files: {e}")


class FileBasedTest:
    """Test real-time inference using pre-recorded audio files."""

    def __init__(self, config: Config):
        """
        Initialize file-based test runner.

        Args:
            config: Configuration object containing all parameters.
        """
        self.config = config
        self.sample_rate = config.audio.sample_rate
        self.chunk_size = config.audio.chunk_size
        self.device = torch.device(config.model.device) if config.model.device else get_torch_device()

        # Load model
        self._load_model(config.get_checkpoint_path(), config.get_model_config_path())
        if config.model.embedding is None:
            raise ValueError("Speaker embedding path is required")
        self._load_embedding(config.model.embedding)
        self.state = self.model.init_buffers(batch_size=1, device=self.device)

    def _load_model(self, checkpoint_path: Path, config_path: Path) -> None:
        """Load the TFGridNet model from checkpoint."""
        with config_path.open() as fp:
            config = json.load(fp)
        model_params = config.get("pl_module_args", {}).get("model_params", {})

        self.model = Net(**model_params).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint

        prefixes = ("model.model.", "model.", "module.")
        for pref in prefixes:
            if all(k.startswith(pref) for k in state_dict.keys()):
                state_dict = {k[len(pref):]: v for k, v in state_dict.items()}
                break

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _load_embedding(self, embedding_path: Path) -> None:
        """Load speaker embedding from .npy file."""
        embedding = np.load(embedding_path)
        embedding = embedding.astype(np.float32).reshape(1, 1, -1)
        self.embedding = torch.from_numpy(embedding).to(self.device)

    def process_file(self, input_path: Path, output_path: Path) -> None:
        """
        Process an audio file chunk-by-chunk, simulating real-time behavior.

        This validates that streaming inference produces correct results.
        """
        import soundfile as sf
        import resampy

        print(f"Processing {input_path} -> {output_path}")

        # Load and resample audio
        audio, sr = sf.read(str(input_path))
        if sr != self.sample_rate:
            audio = resampy.resample(audio, sr, self.sample_rate)

        # Convert to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Reset state for fresh processing
        self.state = self.model.init_buffers(batch_size=1, device=self.device)

        # Process chunk by chunk
        output_chunks = []
        num_chunks = len(audio) // self.chunk_size
        stft_pad_size = self.model.stft_pad_size

        processing_times = []
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk = audio[start:end]

            # Get real lookahead audio from file (next stft_pad_size samples)
            la_tensor = None
            la_end = end + stft_pad_size
            if la_end <= len(audio):
                lookahead = audio[end:la_end]
                stereo_la = np.stack([lookahead, lookahead], axis=0)
                la_tensor = torch.from_numpy(stereo_la).unsqueeze(0).to(self.device)

            # Convert mono to stereo
            stereo_input = np.stack([chunk, chunk], axis=0)
            input_tensor = torch.from_numpy(stereo_input).unsqueeze(0).to(self.device)

            start_time = time.perf_counter()
            with torch.no_grad():
                output, self.state = self.model.predict(
                    input_tensor,
                    self.embedding[:, 0],
                    self.state,
                    pad=True,
                    lookahead_audio=la_tensor
                )
            elapsed = time.perf_counter() - start_time
            processing_times.append(elapsed)

            output_audio = output.squeeze(0).cpu().numpy().T
            output_chunks.append(np.clip(output_audio, -1.0, 1.0))

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_chunks} chunks")

        # Concatenate and save
        output_audio = np.concatenate(output_chunks, axis=0)
        sf.write(str(output_path), output_audio, self.sample_rate)

        avg_time = np.mean(processing_times) * 1000
        chunk_duration = self.chunk_size / self.sample_rate * 1000
        print(f"Done! Avg processing time: {avg_time:.2f}ms per {chunk_duration:.1f}ms chunk (RTF: {avg_time/chunk_duration:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time TFGridNet inference for target speech extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List audio devices
  python realtime_inference.py --list-devices

  # Run real-time inference (set embedding path in config.yaml)
  python realtime_inference.py

  # Override embedding path via CLI
  python realtime_inference.py --embedding /path/to/speaker.npy

  # Test with pre-recorded file
  python realtime_inference.py --test-file input.wav

  # Use specific device
  python realtime_inference.py --device mps
"""
    )

    # Model arguments
    parser.add_argument(
        "--embedding", "-e",
        type=Path,
        default=None,
        help="Path to speaker embedding .npy file (overrides config.yaml)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default=None,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})"
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help=f"Path to model config JSON (default: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (cpu, cuda, mps). Auto-detected if not specified."
    )

    # Utility arguments
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )

    # Test mode arguments
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Process a pre-recorded file instead of live audio (for validation)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output file path (used with --test-file)"
    )

    args = parser.parse_args()

    # Handle --list-devices first
    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        print(f"\nDefault input: {sd.default.device[0]}")
        print(f"Default output: {sd.default.device[1]}")
        return

    # Load configuration from config.yaml
    if DEFAULT_YAML_CONFIG_PATH.exists():
        print(f"Loading config from: {DEFAULT_YAML_CONFIG_PATH}")
        config = Config.from_yaml(DEFAULT_YAML_CONFIG_PATH)
    else:
        print("No config.yaml found, using defaults")
        config = Config()

    # Apply CLI arguments for model settings
    if args.embedding is not None:
        config.model.embedding = args.embedding
    if args.checkpoint is not None:
        config.model.checkpoint = args.checkpoint
    if args.model_config is not None:
        config.model.config = args.model_config
    if args.device is not None:
        config.model.device = args.device
    if args.test_file is not None:
        config.test.input_file = args.test_file
    if args.output_file is not None:
        config.test.output_file = args.output_file

    # Validate required fields
    if config.model.embedding is None and not config.debug.passthrough:
        parser.error("embedding is required (set in config.yaml or use --embedding)")

    # Determine mode: file-based test or real-time
    if config.test.input_file:
        # File-based testing mode
        if config.test.output_file is None:
            config.test.output_file = SCRIPT_DIR / (config.test.input_file.stem + ".enhanced.wav")

        tester = FileBasedTest(config)
        tester.process_file(config.test.input_file, config.test.output_file)
    else:
        # Real-time mode
        engine = RealtimeInference(config)
        engine.run()


if __name__ == "__main__":
    main()
