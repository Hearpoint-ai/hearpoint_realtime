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
import yaml

# Add repo root to path for imports (so `from src.xxx` resolves)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.tfgridnet_realtime.net import Net
from src.utils import get_torch_device


# Default paths
SRC_DIR = REPO_ROOT / "src"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "tfgridnet_cipic.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "weights" / "tfgridnet.ckpt"
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


@dataclass
class DebugConfig:
    """Debug-related configuration."""
    verbose: bool = False
    passthrough: bool = False
    save_dir: Path | None = None


@dataclass
class OptimizationConfig:
    """Performance optimization configuration."""
    use_torch_compile: bool = False


@dataclass
class TestConfig:
    """File-based test configuration."""
    enabled: bool = False
    input_file: Path | None = None
    output_file: Path | None = None


@dataclass
class Config:
    """Complete configuration for real-time inference."""
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    test: TestConfig = field(default_factory=TestConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "Config":
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with yaml_path.open() as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from a dictionary."""
        def to_path(val: Any) -> Path | None:
            if not val:
                return None
            p = Path(val)
            if not p.is_absolute():
                p = REPO_ROOT / p
            return p

        audio_data = data.get("audio", {}) or {}
        debug_data = data.get("debug", {}) or {}
        opt_data = data.get("optimization", {}) or {}
        test_data = data.get("test", {}) or {}

        # Load embedding path from top-level config
        embedding_path = to_path(data.get("embedding"))

        return cls(
            model=ModelConfig(embedding=embedding_path),
            audio=AudioConfig(
                sample_rate=audio_data.get("sample_rate", DEFAULT_SAMPLE_RATE),
                chunk_size=audio_data.get("chunk_size", DEFAULT_CHUNK_SIZE),
                input_device=audio_data.get("input_device", None),
                output_device=audio_data.get("output_device", None),
                input_channels=audio_data.get("input_channels", 2),
                output_channels=audio_data.get("output_channels", 2),
                buffer_size_chunks=audio_data.get("buffer_size_chunks", 4),
            ),
            debug=DebugConfig(
                verbose=debug_data.get("verbose", False),
                passthrough=debug_data.get("passthrough", False),
                save_dir=to_path(debug_data.get("save_dir")),
            ),
            optimization=OptimizationConfig(
                use_torch_compile=opt_data.get("use_torch_compile", False),
            ),
            test=TestConfig(
                enabled=test_data.get("enabled", False),
                input_file=to_path(test_data.get("input_file")),
                output_file=to_path(test_data.get("output_file")),
            ),
        )

    def get_checkpoint_path(self) -> Path:
        """Get checkpoint path with default fallback."""
        return self.model.checkpoint or DEFAULT_CHECKPOINT_PATH

    def get_model_config_path(self) -> Path:
        """Get model config path with default fallback."""
        return self.model.config or DEFAULT_CONFIG_PATH

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

        # Validate device indices if explicitly set
        self._validate_device(self.input_device, "input")
        self._validate_device(self.output_device, "output")

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
            self._compiled = False
            self._input_buffer = None
            self._lookahead_buffer = None
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

            # --- Optimization: compile, pre-allocated tensors ---
            self._compiled = False
            if config.optimization.use_torch_compile:
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    self._compiled = True
                    print("torch.compile enabled (reduce-overhead)")
                except Exception as e:
                    print(f"torch.compile not available ({e}), continuing without it")

            # Pre-allocate reusable tensors
            self._input_buffer = torch.zeros(
                1, 2, self.chunk_size, device=self.device, dtype=torch.float32
            )
            self._lookahead_buffer = torch.zeros(
                1, 2, self.stft_pad_size, device=self.device, dtype=torch.float32
            )

        # Input accumulator for collecting enough samples before processing (stereo)
        self.input_accumulator = np.zeros((0, 2), dtype=np.float32)

        # Statistics
        self.chunks_processed = 0
        self.processing_times = []
        self.inference_times = []
        self.prep_times = []
        self.post_times = []
        self.drops_input = 0
        self.drops_output = 0
        self.underruns = 0

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

    def _validate_device(self, device_index: int, kind: str) -> None:
        """Warn if a specific device index does not exist; no-op when index is None."""
        if device_index is None:
            return
        try:
            sd.query_devices(device_index, kind)
        except Exception as e:
            print(f"Warning: {kind} device index {device_index} is invalid ({e}). "
                  f"sounddevice will fall back to the system default.")

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

    def _process_chunk(self, audio_chunk: np.ndarray, lookahead: np.ndarray | None = None) -> np.ndarray:
        """
        Process a single audio chunk through the model.

        Args:
            audio_chunk: Stereo audio chunk [chunk_size, 2]
            lookahead: Optional stereo lookahead samples [stft_pad_size, 2] for real
                       audio lookahead instead of zero-padding.

        Returns:
            Stereo enhanced audio [chunk_size, 2]
        """
        start_time = time.perf_counter()

        # Update input level for monitoring
        with self.input_level_lock:
            self.recent_input_level = np.abs(audio_chunk).max()

        # Passthrough mode: bypass model entirely
        if self.passthrough_mode:
            if self.output_channels == 1:
                output_audio = audio_chunk.mean(axis=1, keepdims=True)
            else:
                output_audio = audio_chunk
            elapsed = time.perf_counter() - start_time
            self.processing_times.append(elapsed)
            self.chunks_processed += 1

            # Save debug files
            if self.save_debug_dir:
                self.debug_inputs.append(audio_chunk.copy())
                self.debug_outputs.append(output_audio.copy())

            return output_audio

        # --- Prep: numpy -> tensor ---
        t_prep = time.perf_counter()
        stereo_input = audio_chunk.T
        self._input_buffer.copy_(torch.from_numpy(stereo_input).unsqueeze(0))

        la_tensor = None
        if lookahead is not None and len(lookahead) > 0:
            stereo_la = lookahead.T
            self._lookahead_buffer.copy_(torch.from_numpy(stereo_la).unsqueeze(0))
            la_tensor = self._lookahead_buffer

        # --- Inference ---
        t_infer = time.perf_counter()
        with torch.inference_mode():
            output, self.state = self.model.predict(
                self._input_buffer,
                self.embedding[:, 0],
                self.state,
                pad=True,
                lookahead_audio=la_tensor
            )

        # --- Post: tensor -> numpy ---
        t_post = time.perf_counter()
        output_audio = output.squeeze(0).cpu().numpy()
        output_audio = np.clip(output_audio, -1.0, 1.0)
        output_audio = output_audio.T

        if self.output_channels == 1:
            output_audio = output_audio.mean(axis=1, keepdims=True)

        t_done = time.perf_counter()

        # Save debug files
        if self.save_debug_dir:
            self.debug_inputs.append(audio_chunk.copy())
            self.debug_outputs.append(output_audio.copy())

        # Record timing breakdown
        elapsed = t_done - start_time
        self.processing_times.append(elapsed)
        self.prep_times.append(t_infer - t_prep)
        self.inference_times.append(t_post - t_infer)
        self.post_times.append(t_done - t_post)
        self.chunks_processed += 1

        if self.debug:
            chunk_ms = self.chunk_size / self.sample_rate * 1000
            print(f"[chunk {self.chunks_processed}] "
                  f"total={elapsed*1000:.2f}ms "
                  f"(prep={( t_infer - t_prep)*1000:.2f} "
                  f"infer={(t_post - t_infer)*1000:.2f} "
                  f"post={(t_done - t_post)*1000:.2f}) "
                  f"RTF={elapsed*1000/chunk_ms:.3f}")

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
                self.drops_input += 1

    def _output_callback(self, outdata, frames, time_info, status):
        """Callback for audio output stream."""
        if status:
            print(f"Output status: {status}")

        try:
            data = self.output_queue.get_nowait()

            # Ensure correct shape
            if data.shape[0] < frames:
                padding = np.zeros((frames - data.shape[0], self.output_channels), dtype=np.float32)
                data = np.vstack([data, padding])
            outdata[:] = data[:frames]
        except queue.Empty:
            self.underruns += 1
            outdata.fill(0)

    def _processing_thread(self):
        """Background thread for processing audio chunks."""
        while self.running:
            try:
                # Get input audio (blocking with timeout)
                indata = self.input_queue.get(timeout=0.1)

                # Accumulate input samples (preserve stereo)
                if indata.ndim > 1 and indata.shape[1] >= 2:
                    stereo_input = indata[:, :2]
                else:
                    # Mono input: duplicate to stereo
                    mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                    stereo_input = np.column_stack([mono, mono])
                self.input_accumulator = np.concatenate([self.input_accumulator, stereo_input]) if len(self.input_accumulator) > 0 else stereo_input

                # Process complete chunks (need chunk + lookahead samples)
                required_samples = self.chunk_size + self.stft_pad_size

                while len(self.input_accumulator) >= required_samples:
                    # Extract stereo chunk and lookahead [samples, 2]
                    chunk = self.input_accumulator[:self.chunk_size].astype(np.float32)
                    lookahead = self.input_accumulator[self.chunk_size:required_samples].astype(np.float32)
                    # Advance by chunk_size only -- lookahead rolls into next chunk
                    self.input_accumulator = self.input_accumulator[self.chunk_size:]

                    # Process through model with real lookahead audio (pass stereo)
                    output = self._process_chunk(chunk, lookahead)

                    # Add to output queue
                    try:
                        # If queue is getting too full, drop oldest items to prevent latency buildup
                        while self.output_queue.qsize() > 10:
                            try:
                                self.output_queue.get_nowait()
                                self.drops_output += 1
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
        print(f"  torch.compile: {'enabled' if self._compiled else 'disabled'}")

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
                        recent = self.processing_times[-100:]
                        recent_ms = np.array(recent) * 1000
                        chunk_duration = self.chunk_size / self.sample_rate * 1000
                        avg_time = np.mean(recent_ms)
                        p50 = np.percentile(recent_ms, 50)
                        p95 = np.percentile(recent_ms, 95)
                        p99 = np.percentile(recent_ms, 99)
                        rtf = avg_time / chunk_duration

                        # Get input level for visualization
                        with self.input_level_lock:
                            level = self.recent_input_level

                        # Convert to dB and create visual meter
                        level_db = 20 * np.log10(level + 1e-10)
                        bars = int(max(0, min(30, (level_db + 60) / 2)))  # -60dB to 0dB range
                        level_meter = f"[{'=' * bars}{' ' * (30 - bars)}] {level_db:5.1f}dB"

                        stats = (f"Chunks: {self.chunks_processed:6d} | "
                                 f"Avg: {avg_time:5.2f}ms | "
                                 f"p50: {p50:5.2f} p95: {p95:5.2f} p99: {p99:5.2f} | "
                                 f"RTF: {rtf:.3f} | "
                                 f"Q: {self.input_queue.qsize()}/{self.output_queue.qsize()} | "
                                 f"Level: {level_meter}")

                        if self.drops_input or self.drops_output or self.underruns:
                            stats += (f" | Drops(in/out): {self.drops_input}/{self.drops_output} "
                                      f"Underruns: {self.underruns}")

                        print(stats)

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
            all_ms = np.array(self.processing_times) * 1000
            chunk_duration = self.chunk_size / self.sample_rate * 1000
            print(f"\nFinal Statistics:")
            print(f"  Total chunks: {self.chunks_processed}")
            print(f"  Processing (ms) — avg: {np.mean(all_ms):.2f}  "
                  f"p50: {np.percentile(all_ms, 50):.2f}  "
                  f"p95: {np.percentile(all_ms, 95):.2f}  "
                  f"p99: {np.percentile(all_ms, 99):.2f}  "
                  f"max: {np.max(all_ms):.2f}")
            if self.inference_times:
                inf_ms = np.array(self.inference_times) * 1000
                prep_ms = np.array(self.prep_times) * 1000
                post_ms = np.array(self.post_times) * 1000
                print(f"  Breakdown (ms avg) — prep: {np.mean(prep_ms):.2f}  "
                      f"infer: {np.mean(inf_ms):.2f}  "
                      f"post: {np.mean(post_ms):.2f}")
            print(f"  RTF: {np.mean(all_ms) / chunk_duration:.3f}")
            print(f"  Drops (input/output): {self.drops_input}/{self.drops_output}  "
                  f"Underruns: {self.underruns}")

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

        # --- Optimization: compile ---
        self._compiled = False
        if config.optimization.use_torch_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
                print("torch.compile enabled (reduce-overhead)")
            except Exception as e:
                print(f"torch.compile not available ({e}), continuing without it")

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

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        audio = audio[:, :2].astype(np.float32)

        # Reset state for fresh processing
        self.state = self.model.init_buffers(batch_size=1, device=self.device)

        # Process chunk by chunk
        output_chunks = []
        num_chunks = audio.shape[0] // self.chunk_size
        stft_pad_size = self.model.stft_pad_size

        # Pre-allocate reusable tensors
        input_buffer = torch.zeros(1, 2, self.chunk_size, device=self.device, dtype=torch.float32)
        la_buffer = torch.zeros(1, 2, stft_pad_size, device=self.device, dtype=torch.float32)

        processing_times = []
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk = audio[start:end]  # [chunk_size, 2]

            start_time = time.perf_counter()

            # Reuse pre-allocated input tensor
            input_buffer.copy_(torch.from_numpy(chunk.T).unsqueeze(0))

            # Get real lookahead audio from file (next stft_pad_size samples)
            la_tensor = None
            la_end = end + stft_pad_size
            if la_end <= len(audio):
                lookahead = audio[end:la_end]  # [stft_pad_size, 2]
                la_buffer.copy_(torch.from_numpy(lookahead.T).unsqueeze(0))
                la_tensor = la_buffer

            with torch.inference_mode():
                output, self.state = self.model.predict(
                    input_buffer,
                    self.embedding[:, 0],
                    self.state,
                    pad=True,
                    lookahead_audio=la_tensor
                )

            output_audio = output.squeeze(0).cpu().numpy().T
            output_audio = np.clip(output_audio, -1.0, 1.0)

            elapsed = time.perf_counter() - start_time
            processing_times.append(elapsed)

            output_chunks.append(output_audio)

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

  # Run real-time inference (configure settings in config.yaml)
  python realtime_inference.py

  # Use specific device
  python realtime_inference.py --device mps
"""
    )

    # Model arguments
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
    parser.add_argument(
        "--embedding",
        type=Path,
        default=None,
        help="Path to speaker embedding .npy file (overrides config.yaml)"
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Path to input audio file for file-based test mode"
    )

    # Utility arguments
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
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
    if args.checkpoint is not None:
        config.model.checkpoint = args.checkpoint
    if args.model_config is not None:
        config.model.config = args.model_config
    if args.device is not None:
        config.model.device = args.device
    if args.embedding is not None:
        config.model.embedding = args.embedding.resolve()
    if args.test_file is not None:
        config.test.input_file = args.test_file.resolve()
        config.test.enabled = True

    # Validate required fields
    if config.model.embedding is None and not config.debug.passthrough:
        parser.error("embedding is required (set in config.yaml)")

    # Determine mode: file-based test or real-time
    if config.test.enabled:
        # File-based testing mode
        if config.test.input_file is None:
            parser.error("test input_file is required when test mode is enabled (set in config.yaml)")
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
