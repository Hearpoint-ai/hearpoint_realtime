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
import torch.nn as nn
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

_PRECISION_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


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
    hardware_sample_rate: int = DEFAULT_SAMPLE_RATE
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
    precision: str = "fp32"  # "fp32", "fp16", or "bf16"
    use_torch_compile: bool = False
    backend: str = "pytorch"  # "pytorch", "onnx_cpu", or "tensorrt"


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
    test: TestConfig = field(default_factory=TestConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

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
            if not val:
                return None
            p = Path(val)
            if not p.is_absolute():
                p = REPO_ROOT / p
            return p

        audio_data = data.get("audio", {}) or {}
        debug_data = data.get("debug", {}) or {}
        test_data = data.get("test", {}) or {}
        optimization_data = data.get("optimization", {}) or {}

        # Load embedding path from top-level config
        embedding_path = to_path(data.get("embedding"))

        return cls(
            model=ModelConfig(embedding=embedding_path),
            audio=AudioConfig(
                sample_rate=audio_data.get("sample_rate", DEFAULT_SAMPLE_RATE),
                hardware_sample_rate=audio_data.get("hardware_sample_rate", audio_data.get("sample_rate", DEFAULT_SAMPLE_RATE)),
                chunk_size=audio_data.get("chunk_size", DEFAULT_CHUNK_SIZE),
                input_device=audio_data.get("input_device", 5),
                output_device=audio_data.get("output_device", 4),
                input_channels=audio_data.get("input_channels", 2),
                output_channels=audio_data.get("output_channels", 2),
                buffer_size_chunks=audio_data.get("buffer_size_chunks", 4),
            ),
            debug=DebugConfig(
                verbose=debug_data.get("verbose", False),
                passthrough=debug_data.get("passthrough", False),
                save_dir=to_path(debug_data.get("save_dir")),
            ),
            test=TestConfig(
                enabled=test_data.get("enabled", False),
                input_file=to_path(test_data.get("input_file")),
                output_file=to_path(test_data.get("output_file")),
            ),
            optimization=OptimizationConfig(
                precision=optimization_data.get("precision", "fp32"),
                use_torch_compile=optimization_data.get("use_torch_compile", False),
                backend=optimization_data.get("backend", "pytorch"),
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
        self.hardware_sample_rate = config.audio.hardware_sample_rate
        self.downsample_factor = self.hardware_sample_rate // self.sample_rate  # e.g. 3

        if self.downsample_factor > 1:
            from scipy.signal import firwin
            _ntaps = 31
            self._ntaps = _ntaps
            self._taps_down = firwin(_ntaps, cutoff=self.sample_rate / 2,
                                     fs=self.hardware_sample_rate).astype(np.float32)
            self._taps_up = (firwin(_ntaps, cutoff=1.0 / self.downsample_factor)
                             * self.downsample_factor).astype(np.float32)

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

        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device.index or 0)
            # Enable cuDNN autotuner — input shapes are always fixed (chunk_size=128)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Build GPU resampling modules now that device is known
        if self.downsample_factor > 1:
            self._resample_down = nn.Conv1d(
                1, 1, self._ntaps, stride=self.downsample_factor, bias=False
            ).to(self.device)
            with torch.no_grad():
                self._resample_down.weight.copy_(
                    torch.from_numpy(self._taps_down).view(1, 1, -1)
                )
            self._resample_down.weight.requires_grad_(False)

            _up_pad = (self._ntaps - self.downsample_factor) // 2
            self._resample_up = nn.ConvTranspose1d(
                1, 1, self._ntaps, stride=self.downsample_factor, bias=False, padding=_up_pad
            ).to(self.device)
            with torch.no_grad():
                self._resample_up.weight.copy_(
                    torch.from_numpy(self._taps_up).view(1, 1, -1)
                )
            self._resample_up.weight.requires_grad_(False)

            # State for causal downsampling: last ntaps-1 samples per input channel
            self._dec_state = torch.zeros(
                config.audio.input_channels, 1, self._ntaps - 1, device=self.device
            )

        self.backend = config.optimization.backend

        if self.passthrough_mode:
            print("*** PASSTHROUGH MODE - bypassing model ***")
            self.model = None
            self.embedding = None
            self.state = None
            self.stft_pad_size = 0
        elif self.backend in ("onnx_cpu", "tensorrt"):
            self._init_onnx_backend(config)
        else:
            # Load model
            self._load_model(
                config.get_checkpoint_path(),
                config.get_model_config_path(),
                config.optimization.use_torch_compile,
            )

            # Load speaker embedding
            if config.model.embedding is None:
                raise ValueError("Speaker embedding path is required")
            self._load_embedding(config.model.embedding)

            # Initialize model state buffers
            self.state = self.model.init_buffers(batch_size=1, device=self.device)
            if self.config.optimization.precision in ('fp16', 'bf16'):
                dtype = _PRECISION_MAP[self.config.optimization.precision]
                for buf in self.state['gridnet_bufs'].values():
                    buf['h0'] = buf['h0'].to(dtype)
                    buf['c0'] = buf['c0'].to(dtype)
            self.stft_pad_size = self.model.stft_pad_size
            self._warmup()

            # Pre-allocate pinned memory for fast CPU→GPU chunk transfers
            self._input_pin = torch.zeros(self.chunk_size, 2, dtype=torch.float32).pin_memory()
            self._la_pin = torch.zeros(self.stft_pad_size, 2, dtype=torch.float32).pin_memory()

        # Input accumulator for collecting enough samples before processing (stereo)
        self.input_accumulator = np.zeros((0, 2), dtype=np.float32)

        # Statistics
        self.chunks_processed = 0
        self.processing_times = []

    def _load_model(self, checkpoint_path: Path, config_path: Path, use_torch_compile: bool = False) -> None:
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

        if use_torch_compile and self.device.type == 'cuda':
            print("Compiling model with torch.compile (default)...")
            self.model = torch.compile(self.model, mode="default")

    def _warmup(self) -> None:
        """Warm up CUDA kernels to eliminate first-inference latency spike."""
        if self.device.type != 'cuda':
            return
        print("Warming up CUDA kernels (20 passes)...")
        embed_dim = self.embedding.shape[-1]
        dummy_x = torch.zeros(1, 2, self.chunk_size, device=self.device)
        dummy_e = torch.zeros(1, embed_dim, device=self.device)
        dummy_la = torch.zeros(1, 2, self.stft_pad_size, device=self.device)
        warmup_state = self.model.init_buffers(batch_size=1, device=self.device)
        if self.config.optimization.precision in ('fp16', 'bf16'):
            dtype = _PRECISION_MAP[self.config.optimization.precision]
            for buf in warmup_state['gridnet_bufs'].values():
                buf['h0'] = buf['h0'].to(dtype)
                buf['c0'] = buf['c0'].to(dtype)
        with torch.inference_mode():
            for i in range(20):
                _, warmup_state = self.model.predict(
                    dummy_x, dummy_e, warmup_state, lookahead_audio=dummy_la
                )
                if (i + 1) % 5 == 0:
                    torch.cuda.synchronize()
        print("CUDA warmup complete.")

    def _init_onnx_backend(self, config: Config) -> None:
        """Initialize ONNX Runtime session for onnx_cpu or tensorrt backend."""
        import onnxruntime as ort
        from src.models.tfgridnet_realtime.export_wrapper import (
            STATE_INPUT_NAMES, get_state_shapes,
        )

        # Read stft_pad_size from model config JSON directly
        config_path = config.get_model_config_path()
        with config_path.open() as fp:
            model_json = json.load(fp)
        model_params = model_json["pl_module_args"]["model_params"]
        self.stft_pad_size = model_params["stft_pad_size"]

        # We need state shapes — instantiate Net briefly on CPU just for that
        net = Net(**model_params)
        state_shapes = get_state_shapes(net)
        del net

        # Determine ONNX file path
        onnx_path = REPO_ROOT / "weights" / "tfgridnet.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Run: python scripts/export_engine.py"
            )

        # Create ORT session
        if self.backend == "onnx_cpu":
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 4
            sess_opts.inter_op_num_threads = 1
            sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = ort.InferenceSession(
                str(onnx_path), sess_opts,
                providers=["CPUExecutionProvider"],
            )
        else:  # tensorrt
            trt_opts = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(REPO_ROOT / "weights"),
                "trt_fp16_enable": True,
            }
            self.ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=[
                    ("TensorrtExecutionProvider", trt_opts),
                    "CUDAExecutionProvider",
                ],
            )

        # Initialize state as numpy arrays
        self.state_names = STATE_INPUT_NAMES
        self.state_flat = [np.zeros(s, dtype=np.float32) for s in state_shapes]

        # Load embedding as numpy
        if config.model.embedding is None:
            raise ValueError("Speaker embedding path is required")
        embedding_path = config.model.embedding
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding not found: {embedding_path}")
        embedding = np.load(embedding_path).astype(np.float32).reshape(1, 1, -1)
        self.embedding_np = embedding
        self.embedding = torch.from_numpy(embedding)
        print(f"Loaded embedding from {embedding_path}, shape: {embedding.shape}")

        # CPU resampling filter state (for onnx_cpu)
        if self.backend == "onnx_cpu" and self.downsample_factor > 1:
            self._dec_zi = np.zeros((2, self._ntaps - 1), dtype=np.float32)

        # Cache scipy lfilter for onnx_cpu resampling (avoid per-call import)
        if self.backend == "onnx_cpu":
            from scipy.signal import lfilter
            self._lfilter = lfilter

        # Pre-allocate feed dict and numpy buffers for ONNX inference
        self._x_buf = np.zeros((1, 2, self.chunk_size), dtype=np.float32)
        self._la_buf = np.zeros((1, 2, self.stft_pad_size), dtype=np.float32)
        self._feed = {
            "x": self._x_buf,
            "embed": self.embedding_np[:, 0],
            "lookahead": self._la_buf,
        }
        for name, arr in zip(self.state_names, self.state_flat):
            self._feed[name] = arr

        # Warmup ORT session
        self._warmup_ort()
        print(f"ONNX backend ({self.backend}) initialized.")

    def _warmup_ort(self) -> None:
        """Warm up ONNX Runtime session with zero inputs."""
        print("Warming up ONNX Runtime (5 passes)...")
        feed = {
            "x": np.zeros((1, 2, self.chunk_size), dtype=np.float32),
            "embed": self.embedding_np[:, 0],
            "lookahead": np.zeros((1, 2, self.stft_pad_size), dtype=np.float32),
        }
        for name, arr in zip(self.state_names, self.state_flat):
            feed[name] = arr
        for _ in range(5):
            self.ort_session.run(None, feed)
        print("ONNX warmup complete.")

    def _downsample_cpu(self, audio: np.ndarray) -> np.ndarray:
        """CPU-based causal anti-alias + decimate. [N, ch] -> [N//factor, ch]"""
        lfilter = self._lfilter
        out_channels = []
        for c in range(audio.shape[1]):
            filtered, self._dec_zi[c] = lfilter(
                self._taps_down, 1.0, audio[:, c], zi=self._dec_zi[c]
            )
            out_channels.append(filtered[::self.downsample_factor])
        return np.column_stack(out_channels).astype(np.float32)

    def _upsample_cpu(self, audio: np.ndarray) -> np.ndarray:
        """CPU-based zero-stuff + interpolation filter. [N, ch] -> [N*factor, ch]"""
        lfilter = self._lfilter
        factor = self.downsample_factor
        out_channels = []
        for c in range(audio.shape[1]):
            stuffed = np.zeros(len(audio) * factor, dtype=np.float32)
            stuffed[::factor] = audio[:, c]
            out_channels.append(lfilter(self._taps_up, 1.0, stuffed))
        return np.column_stack(out_channels).astype(np.float32)

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

        # Debug: log input stats
        if self.debug:
            has_nan = np.isnan(audio_chunk).any()
            print(f"Input: shape={audio_chunk.shape}, min={audio_chunk.min():.4f}, "
                  f"max={audio_chunk.max():.4f}, mean={audio_chunk.mean():.4f}, nan={has_nan}")

        # Update input level for monitoring (every 16th chunk to reduce overhead)
        if self.chunks_processed % 16 == 0:
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

            if self.debug:
                print(f"Output (passthrough): shape={output_audio.shape}")

            # Save debug files
            if self.save_debug_dir:
                self.debug_inputs.append(audio_chunk.copy())
                self.debug_outputs.append(output_audio.copy())

            return output_audio

        # --- ONNX / TensorRT backend (all numpy, no torch) ---
        if self.backend in ("onnx_cpu", "tensorrt"):
            self._x_buf[:] = audio_chunk.T[np.newaxis]
            if lookahead is not None and len(lookahead) > 0:
                self._la_buf[:] = lookahead.T[np.newaxis]
            else:
                self._la_buf.fill(0)

            outputs = self.ort_session.run(None, self._feed)
            output_audio = outputs[0]            # [1, 2, chunk_size]
            self.state_flat = outputs[1:]        # 15 state arrays
            for name, arr in zip(self.state_names, self.state_flat):
                self._feed[name] = arr

            output_audio = np.clip(output_audio.squeeze(0).T, -1.0, 1.0)  # [chunk_size, 2]

            if self.output_channels == 1:
                output_audio = output_audio.mean(axis=1, keepdims=True)

            if self.save_debug_dir:
                self.debug_inputs.append(audio_chunk.copy())
                self.debug_outputs.append(output_audio.copy())

            elapsed = time.perf_counter() - start_time
            self.processing_times.append(elapsed)
            self.chunks_processed += 1
            return output_audio

        # --- PyTorch backend ---
        # Transpose from [chunk_size, 2] to [2, chunk_size] for model.
        # Use pinned memory for fast CPU→GPU transfer.
        self._input_pin.copy_(torch.from_numpy(audio_chunk))
        input_tensor = self._input_pin.to(self.device, non_blocking=True).T.unsqueeze(0)  # [1, 2, chunk_size]

        # Prepare lookahead tensor if available (real audio instead of zero-padding)
        la_tensor = None
        if lookahead is not None and len(lookahead) > 0:
            self._la_pin.copy_(torch.from_numpy(lookahead))
            la_tensor = self._la_pin.to(self.device, non_blocking=True).T.unsqueeze(0)

        # Run inference with state caching
        autocast_dtype = _PRECISION_MAP.get(self.config.optimization.precision, torch.float16)
        autocast_ctx = torch.autocast(device_type='cuda', dtype=autocast_dtype, cache_enabled=True)
        with torch.inference_mode(), autocast_ctx:
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

    def _downsample(self, audio: np.ndarray) -> np.ndarray:
        """GPU Conv1d causal anti-aliasing + integer decimation. [N, ch] → [N//factor, ch]"""
        ch = audio.shape[1]
        # [ch, 1, N] — each channel processed independently as a batch item
        x = torch.from_numpy(audio).float().T.unsqueeze(1).to(self.device, non_blocking=True)
        # Prepend causal state so filter has context across chunk boundaries
        x = torch.cat([self._dec_state, x], dim=2)  # [ch, 1, ntaps-1+N]
        self._dec_state = x[:, :, -(self._ntaps - 1):].detach()
        out = self._resample_down(x)  # [ch, 1, N//factor]
        return out.squeeze(1).T.cpu().numpy()  # [N//factor, ch]

    def _upsample(self, audio: np.ndarray) -> np.ndarray:
        """GPU ConvTranspose1d interpolation. [N, ch] → [N*factor, ch]"""
        x = torch.from_numpy(audio).float().T.unsqueeze(1).to(self.device, non_blocking=True)
        out = self._resample_up(x)  # [ch, 1, N*factor]
        return out.squeeze(1).T.cpu().numpy()  # [N*factor, ch]

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
                # Get input audio (blocking with timeout), then drain remaining
                indata = self.input_queue.get(timeout=0.1)
                blocks = [indata]
                while True:
                    try:
                        blocks.append(self.input_queue.get_nowait())
                    except queue.Empty:
                        break
                if len(blocks) > 1:
                    indata = np.concatenate(blocks, axis=0)

                if self.downsample_factor > 1:
                    if self.backend == "onnx_cpu":
                        indata = self._downsample_cpu(indata)
                    else:
                        indata = self._downsample(indata)

                if self.debug:
                    print(f"Queue sizes - input: {self.input_queue.qsize()}, output: {self.output_queue.qsize()}, "
                          f"accumulator: {len(self.input_accumulator)}")

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

                    if self.downsample_factor > 1:
                        if self.backend == "onnx_cpu":
                            output = self._upsample_cpu(output)
                        else:
                            output = self._upsample(output)

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
        print(f"  Hardware sample rate: {self.hardware_sample_rate} Hz")
        print(f"  Model sample rate:    {self.sample_rate} Hz")
        print(f"  Chunk size: {self.chunk_size} samples ({self.chunk_size / self.sample_rate * 1000:.1f} ms)")
        print(f"  Input channels: {self.input_channels}")
        print(f"  Output channels: {self.output_channels}")
        print(f"  Input device: {self.input_device or 'default'}")
        print(f"  Output device: {self.output_device or 'default'}")

        self.running = True

        # Pre-fill output queue with silence to prevent initial underruns
        silence = np.zeros((self.chunk_size * self.downsample_factor, self.output_channels), dtype=np.float32)
        for _ in range(self.buffer_size_chunks * 2):
            self.output_queue.put(silence)

        # Start processing thread
        process_thread = threading.Thread(target=self._processing_thread, daemon=True)
        process_thread.start()

        # Configure streams
        # Use slightly larger block size for input to reduce callback overhead
        input_blocksize = self.chunk_size * self.buffer_size_chunks
        output_blocksize = self.chunk_size * self.downsample_factor  # e.g. 128*3=384 at 48kHz

        try:
            with sd.InputStream(
                device=self.input_device,
                samplerate=self.hardware_sample_rate,
                channels=self.input_channels,
                dtype=np.float32,
                blocksize=input_blocksize,
                latency='high',
                callback=self._input_callback,
            ), sd.OutputStream(
                device=self.output_device,
                samplerate=self.hardware_sample_rate,
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
        self.config = config
        self.sample_rate = config.audio.sample_rate
        self.chunk_size = config.audio.chunk_size
        self.backend = config.optimization.backend

        if self.backend in ("onnx_cpu", "tensorrt"):
            self._init_onnx(config)
        else:
            self._init_pytorch(config)

    def _init_pytorch(self, config: Config) -> None:
        self.device = torch.device(config.model.device) if config.model.device else get_torch_device()

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._load_model(
            config.get_checkpoint_path(),
            config.get_model_config_path(),
            config.optimization.use_torch_compile,
        )
        if config.model.embedding is None:
            raise ValueError("Speaker embedding path is required")
        self._load_embedding(config.model.embedding)
        self.state = self.model.init_buffers(batch_size=1, device=self.device)
        self.stft_pad_size = self.model.stft_pad_size
        self._warmup()

    def _init_onnx(self, config: Config) -> None:
        import onnxruntime as ort
        from src.models.tfgridnet_realtime.export_wrapper import (
            STATE_INPUT_NAMES, get_state_shapes,
        )

        config_path = config.get_model_config_path()
        with config_path.open() as fp:
            model_json = json.load(fp)
        model_params = model_json["pl_module_args"]["model_params"]
        self.stft_pad_size = model_params["stft_pad_size"]

        net = Net(**model_params)
        state_shapes = get_state_shapes(net)
        del net

        onnx_path = REPO_ROOT / "weights" / "tfgridnet.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\nRun: python scripts/export_engine.py"
            )

        if self.backend == "onnx_cpu":
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 6
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = ort.InferenceSession(
                str(onnx_path), sess_opts, providers=["CPUExecutionProvider"],
            )
        else:
            trt_opts = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(REPO_ROOT / "weights"),
                "trt_fp16_enable": True,
            }
            self.ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=[("TensorrtExecutionProvider", trt_opts), "CUDAExecutionProvider"],
            )

        self.state_names = STATE_INPUT_NAMES
        self.state_flat = [np.zeros(s, dtype=np.float32) for s in state_shapes]

        if config.model.embedding is None:
            raise ValueError("Speaker embedding path is required")
        self.embedding_np = np.load(config.model.embedding).astype(np.float32).reshape(1, 1, -1)
        print(f"ONNX FileBasedTest ({self.backend}) initialized.")

    def _warmup(self) -> None:
        """Warm up CUDA kernels to eliminate first-inference latency spike."""
        if self.device.type != 'cuda':
            return
        print("Warming up CUDA kernels (20 passes)...")
        embed_dim = self.embedding.shape[-1]
        dummy_x = torch.zeros(1, 2, self.chunk_size, device=self.device)
        dummy_e = torch.zeros(1, embed_dim, device=self.device)
        dummy_la = torch.zeros(1, 2, self.model.stft_pad_size, device=self.device)
        warmup_state = self.model.init_buffers(batch_size=1, device=self.device)
        with torch.inference_mode():
            for i in range(20):
                _, warmup_state = self.model.predict(
                    dummy_x, dummy_e, warmup_state, lookahead_audio=dummy_la
                )
                if (i + 1) % 5 == 0:
                    torch.cuda.synchronize()
        print("CUDA warmup complete.")

    def _load_model(self, checkpoint_path: Path, config_path: Path, use_torch_compile: bool = False) -> None:
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

        if use_torch_compile and self.device.type == 'cuda':
            print("Compiling model with torch.compile (default)...")
            self.model = torch.compile(self.model, mode="default")

    def _load_embedding(self, embedding_path: Path) -> None:
        """Load speaker embedding from .npy file."""
        embedding = np.load(embedding_path)
        embedding = embedding.astype(np.float32).reshape(1, 1, -1)
        self.embedding = torch.from_numpy(embedding).to(self.device)

    def process_file(self, input_path: Path, output_path: Path) -> None:
        """Process an audio file chunk-by-chunk, simulating real-time behavior."""
        import soundfile as sf
        import resampy

        print(f"Processing {input_path} -> {output_path}")

        audio, sr = sf.read(str(input_path))
        if sr != self.sample_rate:
            audio = resampy.resample(audio, sr, self.sample_rate)

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        audio = audio[:, :2].astype(np.float32)

        # Reset state
        if self.backend in ("onnx_cpu", "tensorrt"):
            from src.models.tfgridnet_realtime.export_wrapper import get_state_shapes
            config_path = self.config.get_model_config_path()
            with config_path.open() as fp:
                model_json = json.load(fp)
            model_params = model_json["pl_module_args"]["model_params"]
            net = Net(**model_params)
            state_shapes = get_state_shapes(net)
            del net
            self.state_flat = [np.zeros(s, dtype=np.float32) for s in state_shapes]
        else:
            self.state = self.model.init_buffers(batch_size=1, device=self.device)

        output_chunks = []
        num_chunks = audio.shape[0] // self.chunk_size
        stft_pad_size = self.stft_pad_size

        processing_times = []
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk = audio[start:end]  # [chunk_size, 2]

            la_end = end + stft_pad_size
            lookahead = audio[end:la_end] if la_end <= len(audio) else None

            start_time = time.perf_counter()

            if self.backend in ("onnx_cpu", "tensorrt"):
                x = chunk.T[np.newaxis].astype(np.float32)
                embed = self.embedding_np[:, 0]
                la = (lookahead.T[np.newaxis].astype(np.float32)
                      if lookahead is not None
                      else np.zeros((1, 2, stft_pad_size), dtype=np.float32))

                feed = {"x": x, "embed": embed, "lookahead": la}
                for name, arr in zip(self.state_names, self.state_flat):
                    feed[name] = arr

                outputs = self.ort_session.run(None, feed)
                output_audio = outputs[0].squeeze(0).T  # [chunk_size, 2]
                self.state_flat = outputs[1:]
            else:
                la_tensor = None
                if lookahead is not None:
                    la_tensor = torch.from_numpy(lookahead.T).unsqueeze(0).to(self.device)

                input_tensor = torch.from_numpy(chunk.T).unsqueeze(0).to(self.device)

                autocast_dtype = _PRECISION_MAP.get(self.config.optimization.precision, torch.float16)
                autocast_ctx = torch.autocast(device_type='cuda', dtype=autocast_dtype, cache_enabled=True)
                with torch.inference_mode(), autocast_ctx:
                    output, self.state = self.model.predict(
                        input_tensor, self.embedding[:, 0], self.state,
                        pad=True, lookahead_audio=la_tensor
                    )
                output_audio = output.squeeze(0).cpu().numpy().T

            elapsed = time.perf_counter() - start_time
            processing_times.append(elapsed)
            output_chunks.append(np.clip(output_audio, -1.0, 1.0))

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_chunks} chunks")

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
