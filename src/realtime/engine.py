from __future__ import annotations

import json
import multiprocessing
import queue
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import resampy
import sounddevice as sd
import soundfile as sf
import torch

from src.models.tfgridnet_realtime.net import Net
from src.utils import get_torch_device

from .config import Config, TRANSPARENCY_SOUND_PATH
from .coreml_support import CoreMLModel
from .metrics import _ensure_stereo
from .perf_logger import PerformanceLogger


@dataclass
class ControlCommand:
    kind: str
    payload: Any = None
    manual: bool = False


class RealtimeInference:
    """Real-time TFGridNet inference engine."""

    def __init__(self, config: Config, logger: PerformanceLogger | None = None):
        """
        Initialize real-time inference engine.

        Args:
            config: Configuration object containing all parameters.
            logger: Optional PerformanceLogger for zero-latency stats logging.
        """
        self.config = config
        self._logger = logger
        self.sample_rate = config.audio.sample_rate
        self.chunk_size = config.audio.chunk_size
        self.input_channels = config.audio.input_channels
        self.input_device = config.audio.input_device
        self.output_device = config.audio.output_device
        self.buffer_size_chunks = config.audio.buffer_size_chunks
        self.output_gain = config.audio.output_gain

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
        self._control_queue: queue.Queue[ControlCommand] = queue.Queue()

        # Name detection (Vosk) — runs in a separate OS process to avoid GIL contention
        self.name_detection_enabled = config.name_detection.enabled
        self._name_detection_queue: multiprocessing.Queue | None = None
        self._name_detection_control_queue: multiprocessing.Queue | None = None
        self._name_detection_event: multiprocessing.Event | None = None
        self._name_detection_stop: multiprocessing.Event | None = None
        self.name_detection_armed = self.name_detection_enabled
        if self.name_detection_enabled:
            self._name_detection_queue = multiprocessing.Queue(maxsize=64)
            self._name_detection_control_queue = multiprocessing.Queue(maxsize=16)
            self._name_detection_event = multiprocessing.Event()
            self._name_detection_stop = multiprocessing.Event()
            self._name_detection_target = config.name_detection.target_word.lower().strip()
            self._name_detection_model_path = config.name_detection.model_path
        self._name_detection_grace_period_s = 1.0

        # For input level monitoring
        self.recent_input_level = 0.0
        self.input_level_lock = threading.Lock()

        # Set up device
        self.device = torch.device(config.model.device) if config.model.device else get_torch_device()

        self._using_coreml = False  # updated by _load_model

        # Always load model (needed for toggling passthrough→isolation at runtime)
        self._load_model(
            config.get_checkpoint_path(),
            config.get_model_config_path(),
            coreml_path=config.optimization.coreml_model_path if config.optimization.use_coreml else None,
        )
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
            except Exception:
                pass

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

        # Logging thread control
        self._logging_running = False

        # Statistics
        self.chunks_processed = 0
        self.processing_times = []
        self.passthrough_times = []
        self.isolation_times = []
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

    def _make_embedding_tensor(self, embedding_np: np.ndarray) -> torch.Tensor:
        emb = embedding_np.astype(np.float32).reshape(1, 1, -1)
        return torch.from_numpy(emb).to(self.device)

    def _silence_chunk(self) -> np.ndarray:
        return np.zeros((self.chunk_size, self.output_channels), dtype=np.float32)

    def _prefill_output_queue_with_silence(self) -> None:
        silence = self._silence_chunk()
        for _ in range(self.buffer_size_chunks * 2):
            try:
                self.output_queue.put_nowait(silence.copy())
            except queue.Full:
                break

    @staticmethod
    def _drain_thread_queue(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def _drain_process_queue(q: multiprocessing.Queue | None) -> None:
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
            except Exception:
                break

    def _play_transparency_sound_async(self) -> None:
        threading.Thread(target=self._play_transparency_sound, daemon=True).start()

    def _reset_runtime_context(self) -> None:
        self.state = self.model.init_buffers(batch_size=1, device=self.device)
        self.input_accumulator = np.zeros((0, self.input_channels), dtype=np.float32)
        self._drain_thread_queue(self.input_queue)
        self._drain_thread_queue(self.output_queue)
        self._prefill_output_queue_with_silence()

    def _reset_name_detection_stream(self) -> None:
        if self._name_detection_event is not None:
            self._name_detection_event.clear()
        self._drain_process_queue(self._name_detection_queue)
        if self._name_detection_control_queue is not None:
            try:
                self._name_detection_control_queue.put_nowait(
                    {"type": "reset", "grace_period_s": self._name_detection_grace_period_s}
                )
            except Exception:
                pass

    def _apply_control_command(self, command: ControlCommand) -> None:
        if command.kind == "set_output_gain":
            self.output_gain = float(command.payload)
            return

        if command.kind == "set_embedding":
            self.embedding = self._make_embedding_tensor(command.payload)
            self._reset_runtime_context()
            return

        if command.kind != "set_passthrough":
            raise ValueError(f"Unknown control command: {command.kind}")

        enabled = bool(command.payload)
        changed = enabled != self.passthrough_mode
        self.passthrough_mode = enabled

        if enabled:
            if self.name_detection_enabled and not command.manual:
                self.name_detection_armed = False
                if self._name_detection_event is not None:
                    self._name_detection_event.clear()
        else:
            self.state = self.model.init_buffers(batch_size=1, device=self.device)
            if self.name_detection_enabled and command.manual:
                self.name_detection_armed = True
                self._reset_name_detection_stream()

        self._reset_runtime_context()

        if changed:
            self._play_transparency_sound_async()

    def _submit_control_command(self, command: ControlCommand) -> None:
        if self.running and hasattr(self, "_process_thread"):
            self._control_queue.put(command)
            return
        self._apply_control_command(command)

    def _apply_pending_control_commands(self) -> None:
        while True:
            try:
                command = self._control_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_control_command(command)

    def _handle_name_detection_trigger(self) -> None:
        if self._name_detection_event is None or not self._name_detection_event.is_set():
            return
        self._name_detection_event.clear()
        if not self.name_detection_armed or self.passthrough_mode:
            return
        self._apply_control_command(ControlCommand(kind="set_passthrough", payload=True, manual=False))

    def _load_model(
        self,
        checkpoint_path: Path,
        config_path: Path,
        coreml_path: Path | None = None,
    ) -> None:
        """Load the TFGridNet model from checkpoint, or a CoreML .mlpackage if requested."""
        # --- CoreML path ---
        if coreml_path is not None:
            try:
                self.model = CoreMLModel(coreml_path)
                self._using_coreml = True
                return
            except Exception as e:
                print(f"Warning: CoreML load failed ({e}), falling back to PyTorch.")

        # --- PyTorch path ---
        self._using_coreml = False
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with config_path.open() as fp:
            config = json.load(fp)
        model_params = config.get("pl_module_args", {}).get("model_params", {})

        self.model = Net(**model_params).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint

        prefixes = ("model.model.", "model.", "module.")
        for pref in prefixes:
            if all(k.startswith(pref) for k in state_dict.keys()):
                state_dict = {k[len(pref) :]: v for k, v in state_dict.items()}
                break

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected}")

        self.model.eval()

    def _load_embedding(self, embedding_path: Path) -> None:
        """Load speaker embedding from .npy file."""
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding not found: {embedding_path}")

        embedding = np.load(embedding_path)
        # Shape: [1, 1, embed_dim] for batch processing
        embedding = embedding.astype(np.float32).reshape(1, 1, -1)
        self.embedding = torch.from_numpy(embedding).to(self.device)

    def _validate_device(self, device_index: int, kind: str) -> None:
        """Warn if a specific device index does not exist; no-op when index is None."""
        if device_index is None:
            return
        try:
            sd.query_devices(device_index, kind)
        except Exception as e:
            print(
                f"Warning: {kind} device index {device_index} is invalid ({e}). "
                f"sounddevice will fall back to the system default."
            )

    def _detect_output_channels(self, output_device: int | str | None) -> int:
        """Detect the maximum number of output channels supported by the device."""
        try:
            if output_device is not None:
                device_info = sd.query_devices(output_device)
            else:
                device_info = sd.query_devices(kind="output")

            max_channels = device_info.get("max_output_channels", 2)
            # Prefer stereo if available, otherwise use what's supported
            channels = min(2, max_channels)
            return channels
        except Exception:
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
        if audio_chunk.size == 0:
            return self._silence_chunk()

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
            output_audio = output_audio * self.output_gain
            output_audio = np.clip(output_audio, -1.0, 1.0)
            elapsed = time.perf_counter() - start_time
            self.processing_times.append(elapsed)
            self.passthrough_times.append(elapsed)
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
        audio_chunk = _ensure_stereo(audio_chunk)  # normalise to [chunk_size, 2]
        stereo_input = audio_chunk.T
        self._input_buffer.copy_(torch.from_numpy(stereo_input).unsqueeze(0))

        la_tensor = None
        if lookahead is not None and len(lookahead) > 0:
            lookahead = _ensure_stereo(lookahead)  # normalise to [stft_pad_size, 2]
            stereo_la = lookahead.T
            self._lookahead_buffer.copy_(torch.from_numpy(stereo_la).unsqueeze(0))
            la_tensor = self._lookahead_buffer

        # --- Inference ---
        t_infer = time.perf_counter()
        if self._using_coreml:
            output, self.state = self.model.predict(
                self._input_buffer,
                self.embedding[:, 0],
                self.state,
                pad=True,
                lookahead_audio=la_tensor,
            )
        else:
            with torch.inference_mode():
                output, self.state = self.model.predict(
                    self._input_buffer,
                    self.embedding[:, 0],
                    self.state,
                    pad=True,
                    lookahead_audio=la_tensor,
                )

        # --- Post: tensor -> numpy ---
        t_post = time.perf_counter()
        output_audio = output.squeeze(0).cpu().numpy()
        output_audio *= self.output_gain
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
        self.isolation_times.append(elapsed)
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
            print(
                f"[chunk {self.chunks_processed}] "
                f"total={elapsed * 1000:.2f}ms "
                f"(prep={(t_infer - t_prep) * 1000:.2f} "
                f"infer={(t_post - t_infer) * 1000:.2f} "
                f"post={(t_done - t_post) * 1000:.2f}) "
                f"RTF={elapsed * 1000 / chunk_ms:.3f}"
            )

        return output_audio

    def set_passthrough(self, enabled: bool) -> None:
        """Toggle passthrough mode at runtime through the processing thread."""
        self._submit_control_command(ControlCommand(kind="set_passthrough", payload=enabled, manual=True))

    def set_output_gain(self, gain: float) -> None:
        """Adjust output gain at runtime. Clamped to [0.0, 10.0]."""
        gain = max(0.0, min(10.0, gain))
        self._submit_control_command(ControlCommand(kind="set_output_gain", payload=gain))

    def set_target_word(self, word: str) -> None:
        """Change the Vosk name-detection target word at runtime."""
        self._name_detection_target = word.lower().strip()
        if self._name_detection_control_queue is not None:
            self._name_detection_control_queue.put({"type": "set_target", "word": self._name_detection_target})

    def set_embedding(self, embedding_np: np.ndarray) -> None:
        """Swap speaker embedding at runtime through the processing thread."""
        embedding_copy = np.array(embedding_np, copy=True)
        self._submit_control_command(ControlCommand(kind="set_embedding", payload=embedding_copy))

    def start_enrollment_capture(self, duration_s: float) -> None:
        """Begin accumulating input audio for enrollment."""
        with self._enrollment_lock:
            self._enrollment_buffer = {
                "audio": [],
                "max_samples": int(duration_s * self.sample_rate),
                "collected": 0,
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

        if self.running:
            # Add to input queue
            try:
                self.input_queue.put_nowait(indata.copy())
            except queue.Full:
                self.drops_input += 1

            # Tee audio to name-detection process queue if enabled
            if self._name_detection_queue is not None:
                try:
                    self._name_detection_queue.put_nowait(indata.copy())
                except Exception:
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
        INPUT_DRAIN_THRESHOLD = 8  # chunks (~64ms of lag)

        while self.running:
            if self.input_queue.qsize() >= INPUT_DRAIN_THRESHOLD:
                drained = 0
                while True:
                    try:
                        self.input_queue.get_nowait()
                        drained += 1
                    except queue.Empty:
                        break
                self.input_accumulator = np.zeros((0, self.input_channels), dtype=np.float32)
                self.state = self.model.init_buffers(batch_size=1, device=self.device)
                self.drops_input += drained
                continue
            self._apply_pending_control_commands()
            try:
                # Get input audio (blocking with timeout)
                indata = self.input_queue.get(timeout=0.1)

                # Keep only configured input channels; _process_chunk will ensure stereo
                raw = indata[:, : self.input_channels] if indata.ndim > 1 else indata[:, np.newaxis]
                self.input_accumulator = (
                    np.concatenate([self.input_accumulator, raw]) if len(self.input_accumulator) > 0 else raw
                )

                # Process complete chunks (need chunk + lookahead samples)
                required_samples = self.chunk_size + self.stft_pad_size

                while len(self.input_accumulator) >= required_samples:
                    self._apply_pending_control_commands()
                    if len(self.input_accumulator) < required_samples:
                        break

                    # Extract chunk and lookahead [samples, input_channels]; _process_chunk normalises to stereo
                    chunk = self.input_accumulator[: self.chunk_size].astype(np.float32)
                    lookahead = self.input_accumulator[self.chunk_size : required_samples].astype(np.float32)
                    # Advance by chunk_size only -- lookahead rolls into next chunk
                    self.input_accumulator = self.input_accumulator[self.chunk_size :]

                    # Process through model with real lookahead audio (pass stereo)
                    output = self._process_chunk(chunk, lookahead)

                    # Poll Vosk child-process detection event (non-blocking)
                    self._handle_name_detection_trigger()

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
                        pass

            except queue.Empty:
                continue
            except Exception:
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
        except Exception:
            pass

    def list_devices(self):
        """List available audio devices."""
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        print(f"\nDefault input device: {sd.default.device[0]}")
        print(f"Default output device: {sd.default.device[1]}")

    def _build_snapshot(self, elapsed_s: float) -> dict:
        """Build a snapshot record from current stats (called from logging thread)."""
        recent = self.processing_times[-100:]
        recent_ms = np.array(recent) * 1000
        chunk_duration = self.chunk_size / self.sample_rate * 1000
        avg_time = float(np.mean(recent_ms))
        p50 = float(np.percentile(recent_ms, 50))
        p95 = float(np.percentile(recent_ms, 95))
        p99 = float(np.percentile(recent_ms, 99))
        rtf = avg_time / chunk_duration
        with self.input_level_lock:
            level = self.recent_input_level
        level_db = float(20 * np.log10(level + 1e-10))
        return {
            "type": "snapshot",
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed_s, 3),
            "chunks": self.chunks_processed,
            "rtf": round(rtf, 4),
            "latency_ms_avg": round(avg_time, 4),
            "latency_ms_p50": round(p50, 4),
            "latency_ms_p95": round(p95, 4),
            "latency_ms_p99": round(p99, 4),
            "q_in": self.input_queue.qsize(),
            "q_out": self.output_queue.qsize(),
            "drops_input": self.drops_input,
            "drops_output": self.drops_output,
            "underruns": self.underruns,
            "rms_db": round(level_db, 2),
        }

    def _snapshot_thread_fn(self) -> None:
        """Background thread: emit a snapshot record every second while running."""
        start_time = time.perf_counter()
        while self._logging_running:
            time.sleep(1.0)
            if not self._logging_running:
                break
            if not self.passthrough_mode and self.processing_times:
                self._logger.log(self._build_snapshot(time.perf_counter() - start_time))

    def start(self) -> None:
        """Open audio streams and start processing. Non-blocking."""
        self.running = True

        # Pre-fill output queue with silence to prevent initial underruns
        self._prefill_output_queue_with_silence()

        self._process_thread = threading.Thread(target=self._processing_thread, daemon=True)
        self._process_thread.start()

        if self._logger:
            self._logging_running = True
            self._snapshot_thread = threading.Thread(
                target=self._snapshot_thread_fn, daemon=True, name="perf-log-tick"
            )
            self._snapshot_thread.start()

        self._name_detection_process = None
        if self.name_detection_enabled:
            from src.realtime.vosk_worker import vosk_worker

            self._name_detection_process = multiprocessing.Process(
                target=vosk_worker,
                args=(
                    self._name_detection_queue,
                    self._name_detection_control_queue,
                    self._name_detection_event,
                    self._name_detection_stop,
                    str(self._name_detection_model_path),
                    self.sample_rate,
                    self._name_detection_target,
                ),
                daemon=True,
            )
            self._name_detection_process.start()

        input_blocksize = self.chunk_size * self.buffer_size_chunks
        self._input_stream = sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=self.input_channels,
            dtype=np.float32,
            blocksize=input_blocksize,
            callback=self._input_callback,
        )
        self._output_stream = sd.OutputStream(
            device=self.output_device,
            samplerate=self.sample_rate,
            channels=self.output_channels,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=self._output_callback,
        )
        self._input_stream.start()
        self._output_stream.start()

    def stop(self) -> dict:
        """Stop streams, join threads, return stats dict."""
        self.running = False
        self._logging_running = False
        if hasattr(self, "_process_thread"):
            self._process_thread.join(timeout=2.0)
        if hasattr(self, "_snapshot_thread"):
            self._snapshot_thread.join(timeout=2.0)
        if getattr(self, "_name_detection_stop", None) is not None:
            self._name_detection_stop.set()
        if getattr(self, "_name_detection_process", None) is not None:
            self._name_detection_process.join(timeout=2.0)
            if self._name_detection_process.is_alive():
                self._name_detection_process.terminate()
        for stream_attr in ("_input_stream", "_output_stream"):
            s = getattr(self, stream_attr, None)
            if s is not None:
                s.stop()
                s.close()
        if self.save_debug_dir and hasattr(self, "debug_inputs") and self.debug_inputs:
            self._save_debug_files()
        stats = self._build_stats()
        if self._logger:
            self._logger.log({"type": "summary", **stats})
            self._logger.stop()
        return stats

    def _build_stats(self) -> dict:
        """Build and return stats dict from accumulated metrics."""
        chunk_duration_ms = self.chunk_size / self.sample_rate * 1000
        pw_times = self.processing_times[self.warmup_chunks :]
        if pw_times:
            pw_ms = np.array(pw_times) * 1000
            rtf_avg = float(np.mean(pw_ms) / chunk_duration_ms)
            latency_ms_avg = float(np.mean(pw_ms))
            latency_ms_p50 = float(np.percentile(pw_ms, 50))
            latency_ms_p95 = float(np.percentile(pw_ms, 95))
            latency_ms_p99 = float(np.percentile(pw_ms, 99))
            latency_ms_max = float(np.max(pw_ms))
        else:
            rtf_avg = latency_ms_avg = latency_ms_p50 = latency_ms_p95 = latency_ms_p99 = latency_ms_max = 0.0

        # Per-mode RTF averages (warmup excluded from isolation only)
        iso_times = self.isolation_times[self.warmup_chunks :]
        rtf_avg_isolation = float(np.mean(np.array(iso_times) * 1000) / chunk_duration_ms) if iso_times else 0.0
        rtf_avg_passthrough = (
            float(np.mean(np.array(self.passthrough_times) * 1000) / chunk_duration_ms)
            if self.passthrough_times
            else 0.0
        )

        n_pw = self._total_post_warmup_samples
        rms_in = float(np.sqrt(self._sq_sum_in / n_pw)) if n_pw > 0 else 0.0
        rms_out = float(np.sqrt(self._sq_sum_out / n_pw)) if n_pw > 0 else 0.0
        clip_ratio = float(self._clipped_post_warmup / n_pw) if n_pw > 0 else 0.0

        return {
            "mode": "live",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host_id": socket.gethostname(),
            "device": "coreml" if self._using_coreml else str(self.device),
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "stft_pad_size_samples": self.stft_pad_size if not self.passthrough_mode else 0,
            "chunks": self.chunks_processed,
            "warmup_chunks_excluded": min(self.warmup_chunks, self.chunks_processed),
            "rtf_avg": rtf_avg,
            "rtf_avg_isolation": rtf_avg_isolation,
            "rtf_avg_passthrough": rtf_avg_passthrough,
            "latency_ms_avg": latency_ms_avg,
            "latency_ms_p50": latency_ms_p50,
            "latency_ms_p95": latency_ms_p95,
            "latency_ms_p99": latency_ms_p99,
            "latency_ms_max": latency_ms_max,
            "drops_input": self.drops_input,
            "drops_output": self.drops_output,
            "underruns": self.underruns,
            "nan_count": self.nan_count,
            "clip_ratio": clip_ratio,
            "rms_in": rms_in,
            "rms_out": rms_out,
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
        print(f"  CoreML (ANE): {'enabled' if self._using_coreml else 'disabled'}")
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

                    stats = (
                        f"Chunks: {self.chunks_processed:6d} | "
                        f"Avg: {avg_time:5.2f}ms | "
                        f"p50: {p50:5.2f} p95: {p95:5.2f} p99: {p99:5.2f} | "
                        f"RTF: {rtf:.3f} | "
                        f"Q: {self.input_queue.qsize()}/{self.output_queue.qsize()} | "
                        f"Level: {level_meter}"
                    )

                    if self.drops_input or self.drops_output or self.underruns:
                        stats += (
                            f" | Drops(in/out): {self.drops_input}/{self.drops_output} "
                            f"Underruns: {self.underruns}"
                        )

                    print(stats)

        except KeyboardInterrupt:
            print("\nStopping...")

        # Print final statistics
        if self.processing_times:
            all_ms = np.array(self.processing_times) * 1000
            chunk_duration = self.chunk_size / self.sample_rate * 1000
            print(f"\nFinal Statistics:")
            print(f"  Total chunks: {self.chunks_processed}")
            print(
                f"  Processing (ms) — avg: {np.mean(all_ms):.2f}  "
                f"p50: {np.percentile(all_ms, 50):.2f}  "
                f"p95: {np.percentile(all_ms, 95):.2f}  "
                f"p99: {np.percentile(all_ms, 99):.2f}  "
                f"max: {np.max(all_ms):.2f}"
            )
            if self.inference_times:
                inf_ms = np.array(self.inference_times) * 1000
                prep_ms = np.array(self.prep_times) * 1000
                post_ms = np.array(self.post_times) * 1000
                print(
                    f"  Breakdown (ms avg) — prep: {np.mean(prep_ms):.2f}  "
                    f"infer: {np.mean(inf_ms):.2f}  "
                    f"post: {np.mean(post_ms):.2f}"
                )
            print(f"  RTF: {np.mean(all_ms) / chunk_duration:.3f}")
            print(
                f"  Drops (input/output): {self.drops_input}/{self.drops_output}  "
                f"Underruns: {self.underruns}"
            )

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
