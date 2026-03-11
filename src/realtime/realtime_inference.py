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
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import resampy
import soundfile as sf
import torch
import yaml

# Add repo root to path for imports (so `from src.xxx` resolves)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.tfgridnet_realtime.net import Net
from src.ml.factory import EMBEDDING_MODEL_IDS, create_embedding_model
from src.utils import get_torch_device


# Default paths
SRC_DIR = REPO_ROOT / "src"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "tfgridnet_cipic.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "weights" / "tfgridnet.ckpt"
DEFAULT_YAML_CONFIG_PATH = SCRIPT_DIR / "config.yaml"
TRANSPARENCY_SOUND_PATH = REPO_ROOT / "static" / "transparency-sound-effect.wav"

# Audio parameters (defaults, can be overridden by config)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 128  # 8ms at 16kHz - matches model's stft_chunk_size

# Threshold configuration
DEFAULT_THRESHOLDS_PATH = SCRIPT_DIR / "thresholds.yaml"

# Operators hard-coded per metric key — do not infer from value
_THRESHOLD_OPS: dict[str, str] = {
    "drops_input":             "==",
    "drops_output":            "==",
    "underruns":               "==",
    "nan_count":               "==",
    "rtf_avg":                 "<",
    "clip_ratio":              "<=",
    "cosine_similarity_delta": ">=",
    "si_sdr_improvement":      ">=",
}

# Excluded from threshold evaluation in file mode (always stub-zero)
_FILE_MODE_EXCLUDED: set[str] = {"drops_input", "drops_output", "underruns"}


def _normalize_embedding_model_id(model_id: str | None) -> str:
    normalized = (model_id or "resemblyzer").strip().lower()
    if normalized not in EMBEDDING_MODEL_IDS:
        allowed = ", ".join(EMBEDDING_MODEL_IDS)
        raise ValueError(f"Unknown embedding model '{model_id}'. Allowed: {allowed}")
    return normalized


def _load_threshold_profile(profile: str, path: Path = DEFAULT_THRESHOLDS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"thresholds.yaml not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if profile not in data:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(data.keys())}")
    return {k: v for k, v in data[profile].items() if v is not None}


def _evaluate_thresholds(stats: dict, profile_thresholds: dict, mode: str) -> list[str]:
    """Return list of metric names that failed their threshold check."""
    excluded = _FILE_MODE_EXCLUDED if mode == "file" else set()
    failed = []
    for metric, limit in profile_thresholds.items():
        if metric in excluded or metric not in stats:
            continue
        op = _THRESHOLD_OPS.get(metric, "==")
        actual = stats[metric]
        if actual is None:
            continue                           # skip optional metrics absent from this run
        if   op == "==" and actual != limit:      failed.append(metric)
        elif op == "<"  and not (actual < limit):  failed.append(metric)
        elif op == "<=" and not (actual <= limit): failed.append(metric)
        elif op == ">=" and not (actual >= limit): failed.append(metric)
    return failed


def _write_report(stats: dict, report_path: Path, failed_thresholds: list[str]) -> None:
    """Write JSON report to disk annotated with pass/fail status."""
    out = dict(stats)
    out["status"] = "pass" if not failed_thresholds else "fail"
    out["failed_thresholds"] = failed_thresholds
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Report written: {report_path}")
    if failed_thresholds:
        print(f"THRESHOLD FAILURES: {failed_thresholds}")


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Return audio as [N, 2] float32, duplicating channel 0 if mono.

    Accepts:
      [N]    — 1-D mono
      [N, 1] — column-mono
      [N, 2] — stereo (pass-through)
    """
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.shape[1] == 1:
        audio = np.concatenate([audio, audio], axis=1)
    return audio.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SI-SDR in dB for a single channel [N]."""
    ref = reference.astype(np.float64) - reference.mean()
    est = estimate.astype(np.float64) - estimate.mean()
    alpha      = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    target     = alpha * ref
    noise      = est - target
    target_pow = np.dot(target, target)
    noise_pow  = np.dot(noise, noise)
    # Add 1e-10 offset after ratio so that zero-estimate → -100 dB, not 0 dB
    return float(10 * np.log10(target_pow / (noise_pow + 1e-8) + 1e-10))


def _si_sdr_stereo(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Mean SI-SDR across channels. Inputs: [N, 2]."""
    return float(np.mean([_si_sdr(reference[:, c], estimate[:, c]) for c in range(2)]))


@dataclass
class ModelConfig:
    """Model-related configuration."""
    embedding: Path | None = None
    embedding_model: str = "resemblyzer"
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
class NameDetectionConfig:
    """Name/target-word detection via Vosk (runs on a separate thread from same input stream)."""
    enabled: bool = False
    model_path: Path | None = None
    target_word: str = "matthew"


@dataclass
class Config:
    """Complete configuration for real-time inference."""
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    test: TestConfig = field(default_factory=TestConfig)
    name_detection: NameDetectionConfig = field(default_factory=NameDetectionConfig)

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
        name_detection_data = data.get("name_detection", {}) or {}

        # Load embedding settings from top-level config
        embedding_path = to_path(data.get("embedding"))
        embedding_model = _normalize_embedding_model_id(data.get("embedding_model", "resemblyzer"))

        # Name detection model path: from config or default under repo
        nd_model = name_detection_data.get("model_path")
        nd_model_path = to_path(nd_model) if nd_model else REPO_ROOT / "src" / "models" / "vosk-model-small-en-us-0.15"

        return cls(
            model=ModelConfig(
                embedding=embedding_path,
                embedding_model=embedding_model,
            ),
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
            name_detection=NameDetectionConfig(
                enabled=name_detection_data.get("enabled", False),
                model_path=nd_model_path,
                target_word=name_detection_data.get("target_word", "matthew"),
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

        # Name detection (Vosk)
        self.name_detection_enabled = config.name_detection.enabled
        self.name_detection_queue: queue.Queue | None = None
        if self.name_detection_enabled:
            self.name_detection_queue = queue.Queue(maxsize=64)
            self._name_detection_target = config.name_detection.target_word.lower().strip()
            self._name_detection_model_path = config.name_detection.model_path

        # For input level monitoring
        self.recent_input_level = 0.0
        self.input_level_lock = threading.Lock()

        # Set up device
        self.device = torch.device(config.model.device) if config.model.device else get_torch_device()
        print(f"Using device: {self.device}")

        if self.passthrough_mode:
            print("*** PASSTHROUGH MODE - bypassing model ***")

        # Always load model (needed for toggling passthrough→isolation at runtime)
        self._load_model(config.get_checkpoint_path(), config.get_model_config_path())
        self.state = self.model.init_buffers(batch_size=1, device=self.device)
        self.stft_pad_size = self.model.stft_pad_size

        # Load speaker embedding if provided (may be None for demo startup)
        if config.model.embedding is not None:
            self._load_embedding(config.model.embedding)
        else:
            self.embedding = None

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

        # Enrollment capture buffer
        self._enrollment_buffer = None
        self._enrollment_lock = threading.Lock()

        # Input accumulator for collecting enough samples before processing
        self.input_accumulator = np.zeros((0, self.input_channels), dtype=np.float32)

        # Statistics
        self.chunks_processed = 0
        self.processing_times = []
        self.inference_times = []
        self.prep_times = []
        self.post_times = []
        self.drops_input = 0
        self.drops_output = 0
        self.underruns = 0

        # Evaluation metric accumulators (set warmup_chunks before calling run())
        self.warmup_chunks = 0
        self.nan_count = 0
        self._sq_sum_in = 0.0
        self._sq_sum_out = 0.0
        self._total_post_warmup_samples = 0
        self._clipped_post_warmup = 0

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

        # Passthrough mode or no embedding: bypass model entirely
        if self.passthrough_mode or self.embedding is None:
            audio_chunk = _ensure_stereo(audio_chunk)
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

            # Metric tracking
            self.nan_count += int(np.isnan(output_audio).sum())
            if self.chunks_processed >= self.warmup_chunks:
                n = output_audio.size
                self._sq_sum_in += float(np.sum(audio_chunk.astype(np.float64) ** 2))
                self._sq_sum_out += float(np.sum(output_audio.astype(np.float64) ** 2))
                self._clipped_post_warmup += int((np.abs(output_audio) >= 1.0).sum())
                self._total_post_warmup_samples += n

            return output_audio

        # --- Prep: numpy -> tensor ---
        t_prep = time.perf_counter()
        audio_chunk = _ensure_stereo(audio_chunk)          # normalise to [chunk_size, 2]
        stereo_input = audio_chunk.T
        self._input_buffer.copy_(torch.from_numpy(stereo_input).unsqueeze(0))

        la_tensor = None
        if lookahead is not None and len(lookahead) > 0:
            lookahead = _ensure_stereo(lookahead)          # normalise to [stft_pad_size, 2]
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

        # Metric tracking
        self.nan_count += int(np.isnan(output_audio).sum())
        if self.chunks_processed >= self.warmup_chunks:
            n = output_audio.size
            self._sq_sum_in += float(np.sum(audio_chunk.astype(np.float64) ** 2))
            self._sq_sum_out += float(np.sum(output_audio.astype(np.float64) ** 2))
            self._clipped_post_warmup += int((np.abs(output_audio) >= 1.0).sum())
            self._total_post_warmup_samples += n

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

    def set_passthrough(self, enabled: bool) -> None:
        """Toggle passthrough mode at runtime. GIL-atomic bool assignment."""
        changed = (enabled != self.passthrough_mode)
        self.passthrough_mode = enabled
        if not enabled:
            # Reset model state for clean start when switching to isolation
            self.state = self.model.init_buffers(batch_size=1, device=self.device)
        if changed:
            threading.Thread(target=self._play_transparency_sound, daemon=True).start()

    def set_embedding(self, embedding_np: np.ndarray) -> None:
        """Swap speaker embedding at runtime. GIL-atomic reference swap."""
        emb = embedding_np.astype(np.float32).reshape(1, 1, -1)
        self.embedding = torch.from_numpy(emb).to(self.device)
        # Reset model recurrent state for new speaker
        self.state = self.model.init_buffers(batch_size=1, device=self.device)

    def start_enrollment_capture(self, duration_s: float) -> None:
        """Begin accumulating input audio for enrollment."""
        with self._enrollment_lock:
            self._enrollment_buffer = {
                "audio": [], "max_samples": int(duration_s * self.sample_rate), "collected": 0
            }

    def stop_enrollment_capture(self) -> np.ndarray | None:
        """Stop capture, return [N, channels] audio or None."""
        with self._enrollment_lock:
            buf = self._enrollment_buffer
            self._enrollment_buffer = None
        if buf is None or not buf["audio"]:
            return None
        return np.concatenate(buf["audio"], axis=0)

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

            # Tee audio to name-detection queue if enabled
            if self.name_detection_queue is not None:
                try:
                    self.name_detection_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass  # drop silently; name detection is best-effort

            # Capture audio for enrollment if active
            with self._enrollment_lock:
                if self._enrollment_buffer is not None:
                    buf = self._enrollment_buffer
                    if buf["collected"] < buf["max_samples"]:
                        remaining = buf["max_samples"] - buf["collected"]
                        chunk = indata[:remaining].copy()
                        buf["audio"].append(chunk)
                        buf["collected"] += len(chunk)

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

                # Keep only configured input channels; _process_chunk will ensure stereo
                raw = indata[:, :self.input_channels] if indata.ndim > 1 else indata[:, np.newaxis]
                self.input_accumulator = (
                    np.concatenate([self.input_accumulator, raw])
                    if len(self.input_accumulator) > 0 else raw
                )

                # Process complete chunks (need chunk + lookahead samples)
                required_samples = self.chunk_size + self.stft_pad_size

                while len(self.input_accumulator) >= required_samples:
                    # Extract chunk and lookahead [samples, input_channels]; _process_chunk normalises to stereo
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

    def _name_detection_thread(self) -> None:
        """Background thread: Vosk ASR on tee'd input stream, trigger on target word."""
        from vosk import Model, KaldiRecognizer  # lazy import — vosk not required when disabled

        if self._name_detection_model_path is None or not self._name_detection_model_path.exists():
            return
        model = Model(str(self._name_detection_model_path))
        recognizer = KaldiRecognizer(model, self.sample_rate)
        target = self._name_detection_target

        while self.running:
            try:
                chunk = self.name_detection_queue.get(timeout=0.1)
                # Convert float32 (stereo or mono) to mono int16 bytes — expected Vosk input format
                if chunk.ndim > 1:
                    mono = chunk.mean(axis=1)
                else:
                    mono = chunk.flatten()
                data = (mono * 32767).astype(np.int16).tobytes()

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = (result.get("text") or "").lower().strip()
                    if target in text.split() and not self.passthrough_mode:
                        self.set_passthrough(True)
                else:
                    partial = json.loads(recognizer.PartialResult())
                    text = (partial.get("partial") or "").lower().strip()
                    if text and target in text.split() and not self.passthrough_mode:
                        self.set_passthrough(True)

            except queue.Empty:
                continue
            except Exception as e:
                pass

    def _play_transparency_sound(self) -> None:
        """Play the transparency sound effect once (in a separate thread)."""
        if not TRANSPARENCY_SOUND_PATH.exists():
            return
        try:
            audio, file_sr = sf.read(str(TRANSPARENCY_SOUND_PATH), dtype="float32")
            if file_sr != self.sample_rate:
                if audio.ndim == 1:
                    audio = resampy.resample(audio, file_sr, self.sample_rate)
                else:
                    audio = resampy.resample(audio, file_sr, self.sample_rate, axis=0)
            if audio.ndim == 1:
                if self.output_channels == 2:
                    audio = np.column_stack([audio, audio])
            else:
                audio = audio[:, :2]
                if self.output_channels == 1:
                    audio = audio.mean(axis=1, keepdims=True)
            audio = audio.astype(np.float32)
            sd.play(audio, self.sample_rate, device=self.output_device, blocking=True)
        except Exception as e:
            pass

    def list_devices(self):
        """List available audio devices."""
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        print(f"\nDefault input device: {sd.default.device[0]}")
        print(f"Default output device: {sd.default.device[1]}")

    def start(self) -> None:
        """Open audio streams and start processing. Non-blocking."""
        self.running = True

        # Pre-fill output queue with silence to prevent initial underruns
        silence = np.zeros((self.chunk_size, self.output_channels), dtype=np.float32)
        for _ in range(self.buffer_size_chunks * 2):
            self.output_queue.put(silence)

        self._process_thread = threading.Thread(target=self._processing_thread, daemon=True)
        self._process_thread.start()

        self._name_detection_thread_handle = None
        if self.name_detection_enabled:
            self._name_detection_thread_handle = threading.Thread(
                target=self._name_detection_thread, daemon=True)
            self._name_detection_thread_handle.start()

        input_blocksize = self.chunk_size * self.buffer_size_chunks
        self._input_stream = sd.InputStream(
            device=self.input_device, samplerate=self.sample_rate,
            channels=self.input_channels, dtype=np.float32,
            blocksize=input_blocksize, callback=self._input_callback)
        self._output_stream = sd.OutputStream(
            device=self.output_device, samplerate=self.sample_rate,
            channels=self.output_channels, dtype=np.float32,
            blocksize=self.chunk_size, callback=self._output_callback)
        self._input_stream.start()
        self._output_stream.start()

    def stop(self) -> dict:
        """Stop streams, join threads, return stats dict."""
        self.running = False
        if hasattr(self, '_process_thread'):
            self._process_thread.join(timeout=2.0)
        if getattr(self, '_name_detection_thread_handle', None) is not None:
            self._name_detection_thread_handle.join(timeout=1.0)
        for stream_attr in ('_input_stream', '_output_stream'):
            s = getattr(self, stream_attr, None)
            if s is not None:
                s.stop()
                s.close()
        if self.save_debug_dir and hasattr(self, 'debug_inputs') and self.debug_inputs:
            self._save_debug_files()
        return self._build_stats()

    def _build_stats(self) -> dict:
        """Build and return stats dict from accumulated metrics."""
        chunk_duration_ms = self.chunk_size / self.sample_rate * 1000
        pw_times = self.processing_times[self.warmup_chunks:]
        if pw_times:
            pw_ms = np.array(pw_times) * 1000
            rtf_avg         = float(np.mean(pw_ms) / chunk_duration_ms)
            latency_ms_avg  = float(np.mean(pw_ms))
            latency_ms_p50  = float(np.percentile(pw_ms, 50))
            latency_ms_p95  = float(np.percentile(pw_ms, 95))
            latency_ms_p99  = float(np.percentile(pw_ms, 99))
            latency_ms_max  = float(np.max(pw_ms))
        else:
            rtf_avg = latency_ms_avg = latency_ms_p50 = latency_ms_p95 = latency_ms_p99 = latency_ms_max = 0.0

        n_pw = self._total_post_warmup_samples
        rms_in     = float(np.sqrt(self._sq_sum_in  / n_pw)) if n_pw > 0 else 0.0
        rms_out    = float(np.sqrt(self._sq_sum_out / n_pw)) if n_pw > 0 else 0.0
        clip_ratio = float(self._clipped_post_warmup / n_pw) if n_pw > 0 else 0.0

        return {
            "mode":                  "live",
            "timestamp":             datetime.now(timezone.utc).isoformat(),
            "host_id":               socket.gethostname(),
            "device":                str(self.device),
            "sample_rate":           self.sample_rate,
            "chunk_size":            self.chunk_size,
            "stft_pad_size_samples": self.stft_pad_size if not self.passthrough_mode else 0,
            "chunks":                self.chunks_processed,
            "warmup_chunks_excluded": min(self.warmup_chunks, self.chunks_processed),
            "rtf_avg":               rtf_avg,
            "latency_ms_avg":        latency_ms_avg,
            "latency_ms_p50":        latency_ms_p50,
            "latency_ms_p95":        latency_ms_p95,
            "latency_ms_p99":        latency_ms_p99,
            "latency_ms_max":        latency_ms_max,
            "drops_input":           self.drops_input,
            "drops_output":          self.drops_output,
            "underruns":             self.underruns,
            "nan_count":             self.nan_count,
            "clip_ratio":            clip_ratio,
            "rms_in":                rms_in,
            "rms_out":               rms_out,
        }

    def run(self, duration: float | None = None) -> dict:
        """Start real-time processing. Returns a stats dict."""
        print(f"\nStarting real-time inference...")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Chunk size: {self.chunk_size} samples ({self.chunk_size / self.sample_rate * 1000:.1f} ms)")
        print(f"  Input channels: {self.input_channels}")
        print(f"  Output channels: {self.output_channels}")
        print(f"  Input device: {self.input_device or 'default'}")
        print(f"  Output device: {self.output_device or 'default'}")
        print(f"  torch.compile: {'enabled' if self._compiled else 'disabled'}")
        self.start()

        try:
            print("\nProcessing... Press Ctrl+C to stop.\n")

            # Main loop - print stats periodically
            run_start = time.perf_counter()
            while self.running:
                time.sleep(1.0)
                if duration is not None and (time.perf_counter() - run_start) >= duration:
                    self.running = False
                    break
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

        return self.stop()

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
        self.embedding_model_id = _normalize_embedding_model_id(config.model.embedding_model)
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
        self.embedding_path = embedding_path
        embedding = np.load(embedding_path)
        embedding = embedding.astype(np.float32).reshape(1, 1, -1)
        self.embedding = torch.from_numpy(embedding).to(self.device)

    def process_file(self, input_path: Path, output_path: Path,
                     warmup_chunks: int = 10,
                     reference_path: Path | None = None) -> dict:
        """
        Process an audio file chunk-by-chunk, simulating real-time behavior.

        Returns a stats dict conforming to the JSON report schema.
        """
        print(f"Processing {input_path} -> {output_path}")

        # Load and resample audio
        audio, sr = sf.read(str(input_path))
        if sr != self.sample_rate:
            audio = resampy.resample(audio, sr, self.sample_rate)

        audio = audio.astype(np.float32)

        # Reset state for fresh processing
        self.state = self.model.init_buffers(batch_size=1, device=self.device)

        num_chunks = audio.shape[0] // self.chunk_size
        stft_pad_size = self.model.stft_pad_size

        # Pre-allocate reusable tensors
        input_buffer = torch.zeros(1, 2, self.chunk_size, device=self.device, dtype=torch.float32)
        la_buffer    = torch.zeros(1, 2, stft_pad_size,   device=self.device, dtype=torch.float32)

        output_chunks: list[np.ndarray] = []
        post_warmup_times: list[float]  = []
        post_warmup_inputs:  list[np.ndarray] = []
        post_warmup_outputs: list[np.ndarray] = []
        nan_count = 0

        for i in range(num_chunks):
            start = i * self.chunk_size
            end   = start + self.chunk_size
            chunk = _ensure_stereo(audio[start:end])  # always [chunk_size, 2]

            t0 = time.perf_counter()
            input_buffer.copy_(torch.from_numpy(chunk.T).unsqueeze(0))

            la_tensor = None
            la_end = end + stft_pad_size
            if la_end <= len(audio):
                la_stereo = _ensure_stereo(audio[end:la_end])  # [stft_pad_size, 2]
                la_buffer.copy_(torch.from_numpy(la_stereo.T).unsqueeze(0))
                la_tensor = la_buffer

            with torch.inference_mode():
                output, self.state = self.model.predict(
                    input_buffer,
                    self.embedding[:, 0],
                    self.state,
                    pad=True,
                    lookahead_audio=la_tensor,
                )

            out = np.clip(output.squeeze(0).cpu().numpy().T, -1.0, 1.0)
            elapsed = time.perf_counter() - t0

            # NaN tracking: always, from chunk 0 (NaNs during warmup are bugs too)
            nan_count += int(np.isnan(out).sum())

            if i >= warmup_chunks:
                post_warmup_times.append(elapsed)
                post_warmup_inputs.append(chunk)
                post_warmup_outputs.append(out)

            output_chunks.append(out)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_chunks} chunks")

        # Save full output
        sf.write(str(output_path), np.concatenate(output_chunks, axis=0), self.sample_rate)

        # Aggregate post-warmup metrics
        chunk_duration_ms = self.chunk_size / self.sample_rate * 1000
        actual_warmup = min(warmup_chunks, num_chunks)

        if post_warmup_times:
            pw_ms = np.array(post_warmup_times) * 1000
            rtf_avg        = float(np.mean(pw_ms) / chunk_duration_ms)
            latency_ms_avg = float(np.mean(pw_ms))
            latency_ms_p50 = float(np.percentile(pw_ms, 50))
            latency_ms_p95 = float(np.percentile(pw_ms, 95))
            latency_ms_p99 = float(np.percentile(pw_ms, 99))
            latency_ms_max = float(np.max(pw_ms))
        else:
            rtf_avg = latency_ms_avg = latency_ms_p50 = latency_ms_p95 = latency_ms_p99 = latency_ms_max = 0.0

        if post_warmup_inputs:
            inp = np.concatenate(post_warmup_inputs)
            out_all = np.concatenate(post_warmup_outputs)
            rms_in     = float(np.sqrt(np.mean(inp.astype(np.float64) ** 2)))
            rms_out    = float(np.sqrt(np.mean(out_all.astype(np.float64) ** 2)))
            clip_ratio = float(np.mean(np.abs(out_all) >= 1.0))
        else:
            rms_in = rms_out = clip_ratio = 0.0

        print(f"Done! avg={latency_ms_avg:.2f}ms  RTF={rtf_avg:.3f}  "
              f"nan={nan_count}  clip={clip_ratio:.4f}")

        # --- Cosine similarity ---
        sidecar_path = self.embedding_path.with_suffix(".meta.json")
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"Embedding sidecar not found: {sidecar_path}\n"
                f"Re-enroll/regenerate fixture with --embedding-model {self.embedding_model_id}"
            )
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid embedding sidecar JSON: {sidecar_path}") from exc

        sidecar_model_id = sidecar.get("embedding_model_id")
        if sidecar_model_id is None:
            raise ValueError(
                f"Embedding sidecar missing 'embedding_model_id': {sidecar_path}\n"
                f"Re-enroll/regenerate fixture with --embedding-model {self.embedding_model_id}"
            )
        sidecar_model_id = _normalize_embedding_model_id(str(sidecar_model_id))
        if sidecar_model_id != self.embedding_model_id:
            raise ValueError(
                "Embedding model mismatch: "
                f"enrollment sidecar has '{sidecar_model_id}', "
                f"but runtime selected '{self.embedding_model_id}'. "
                "Re-enroll/regenerate fixture with the selected model."
            )

        emb_model = create_embedding_model(self.embedding_model_id)
        emb_input  = emb_model.compute_embedding(input_path)
        emb_output = emb_model.compute_embedding(output_path)
        target_emb = self.embedding.squeeze().cpu().numpy()

        cos_before = _cosine_similarity(emb_input,  target_emb)
        cos_after  = _cosine_similarity(emb_output, target_emb)
        cos_delta  = cos_after - cos_before

        # --- SI-SDR (only when --reference-file provided) ---
        if reference_path is not None and post_warmup_inputs:
            ref_audio, ref_sr = sf.read(str(reference_path), always_2d=True)
            if ref_sr != self.sample_rate:
                ref_audio = resampy.resample(ref_audio.T, ref_sr, self.sample_rate).T
            ref_audio   = _ensure_stereo(ref_audio)
            ref_trimmed = ref_audio[actual_warmup * self.chunk_size : num_chunks * self.chunk_size]
            n_ref = min(len(ref_trimmed), len(inp))
            si_sdr_in  = _si_sdr_stereo(ref_trimmed[:n_ref], inp[:n_ref])
            si_sdr_out = _si_sdr_stereo(ref_trimmed[:n_ref], out_all[:n_ref])
            si_sdr_i   = si_sdr_out - si_sdr_in
        else:
            si_sdr_in = si_sdr_out = si_sdr_i = None

        return {
            "mode":                  "file",
            "timestamp":             datetime.now(timezone.utc).isoformat(),
            "host_id":               socket.gethostname(),
            "device":                str(self.device),
            "sample_rate":           self.sample_rate,
            "chunk_size":            self.chunk_size,
            "stft_pad_size_samples": stft_pad_size,
            "chunks":                num_chunks,
            "warmup_chunks_excluded": actual_warmup,
            "rtf_avg":               rtf_avg,
            "latency_ms_avg":        latency_ms_avg,
            "latency_ms_p50":        latency_ms_p50,
            "latency_ms_p95":        latency_ms_p95,
            "latency_ms_p99":        latency_ms_p99,
            "latency_ms_max":        latency_ms_max,
            "drops_input":           0,
            "drops_output":          0,
            "underruns":             0,
            "nan_count":             nan_count,
            "clip_ratio":            clip_ratio,
            "rms_in":                rms_in,
            "rms_out":               rms_out,
            "cosine_similarity_before": cos_before,
            "cosine_similarity_after":  cos_after,
            "cosine_similarity_delta":  cos_delta,
            "si_sdr_input":             si_sdr_in,
            "si_sdr_output":            si_sdr_out,
            "si_sdr_improvement":       si_sdr_i,
        }


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
        "--embedding-model",
        choices=EMBEDDING_MODEL_IDS,
        default=None,
        help="Speaker embedding model ID (overrides top-level embedding_model in config.yaml)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Path to input audio file for file-based test mode"
    )

    # Eval arguments
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write JSON eval report to this path after the run"
    )
    parser.add_argument(
        "--threshold-profile",
        type=str,
        default=None,
        metavar="PROFILE",
        help="Evaluate stats against this profile in thresholds.yaml (e.g. dev, target)"
    )
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=10,
        metavar="N",
        help="Exclude first N chunks from metric aggregation (default: 10)"
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=None,
        help="Clean reference audio for SI-SDR computation (optional; enables si_sdr_* metrics)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Stop live mode after this many seconds and write report"
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
        try:
            config = Config.from_yaml(DEFAULT_YAML_CONFIG_PATH)
        except ValueError as exc:
            parser.error(str(exc))
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
    if args.embedding_model is not None:
        config.model.embedding_model = _normalize_embedding_model_id(args.embedding_model)
    if args.test_file is not None:
        config.test.input_file = args.test_file.resolve()
        config.test.enabled = True
    config.model.embedding_model = _normalize_embedding_model_id(config.model.embedding_model)

    # Validate required fields
    if config.model.embedding is None and not config.debug.passthrough:
        parser.error("embedding is required (set in config.yaml)")

    # Load threshold profile once (fail fast if misconfigured)
    profile_thresholds: dict = {}
    if args.threshold_profile:
        profile_thresholds = _load_threshold_profile(args.threshold_profile)

    # Determine mode: file-based test or real-time
    if config.test.enabled:
        # File-based testing mode
        if config.test.input_file is None:
            parser.error("test input_file is required when test mode is enabled (set in config.yaml)")
        if config.test.output_file is None:
            config.test.output_file = SCRIPT_DIR / (config.test.input_file.stem + ".enhanced.wav")

        tester = FileBasedTest(config)
        stats = tester.process_file(
            config.test.input_file,
            config.test.output_file,
            warmup_chunks=args.warmup_chunks,
            reference_path=args.reference_file,
        )
        mode = "file"
    else:
        # Real-time mode
        engine = RealtimeInference(config)
        engine.warmup_chunks = args.warmup_chunks
        stats = engine.run(duration=args.duration)
        mode = "live"

    # Report and threshold evaluation
    failed: list[str] = []
    if profile_thresholds:
        failed = _evaluate_thresholds(stats, profile_thresholds, mode)

    if args.report:
        _write_report(stats, args.report, failed)

    if failed:
        print(f"Exiting non-zero: {len(failed)} threshold(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
