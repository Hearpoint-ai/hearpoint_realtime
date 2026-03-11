from __future__ import annotations

import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
import torch

from src.ml.factory import create_embedding_model
from src.models.tfgridnet_realtime.net import Net
from src.utils import get_torch_device

from .config import Config, _normalize_embedding_model_id
from .metrics import _cosine_similarity, _ensure_stereo, _si_sdr_stereo


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
                state_dict = {k[len(pref) :]: v for k, v in state_dict.items()}
                break

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _load_embedding(self, embedding_path: Path) -> None:
        """Load speaker embedding from .npy file."""
        self.embedding_path = embedding_path
        embedding = np.load(embedding_path)
        embedding = embedding.astype(np.float32).reshape(1, 1, -1)
        self.embedding = torch.from_numpy(embedding).to(self.device)

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        warmup_chunks: int = 10,
        reference_path: Path | None = None,
    ) -> dict:
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
        la_buffer = torch.zeros(1, 2, stft_pad_size, device=self.device, dtype=torch.float32)

        output_chunks: list[np.ndarray] = []
        post_warmup_times: list[float] = []
        post_warmup_inputs: list[np.ndarray] = []
        post_warmup_outputs: list[np.ndarray] = []
        nan_count = 0

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
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
            rtf_avg = float(np.mean(pw_ms) / chunk_duration_ms)
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
            rms_in = float(np.sqrt(np.mean(inp.astype(np.float64) ** 2)))
            rms_out = float(np.sqrt(np.mean(out_all.astype(np.float64) ** 2)))
            clip_ratio = float(np.mean(np.abs(out_all) >= 1.0))
        else:
            rms_in = rms_out = clip_ratio = 0.0

        print(f"Done! avg={latency_ms_avg:.2f}ms  RTF={rtf_avg:.3f}  " f"nan={nan_count}  clip={clip_ratio:.4f}")

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
        emb_input = emb_model.compute_embedding(input_path)
        emb_output = emb_model.compute_embedding(output_path)
        target_emb = self.embedding.squeeze().cpu().numpy()

        cos_before = _cosine_similarity(emb_input, target_emb)
        cos_after = _cosine_similarity(emb_output, target_emb)
        cos_delta = cos_after - cos_before

        # --- SI-SDR (only when --reference-file provided) ---
        if reference_path is not None and post_warmup_inputs:
            ref_audio, ref_sr = sf.read(str(reference_path), always_2d=True)
            if ref_sr != self.sample_rate:
                ref_audio = resampy.resample(ref_audio.T, ref_sr, self.sample_rate).T
            ref_audio = _ensure_stereo(ref_audio)
            ref_trimmed = ref_audio[actual_warmup * self.chunk_size : num_chunks * self.chunk_size]
            n_ref = min(len(ref_trimmed), len(inp))
            si_sdr_in = _si_sdr_stereo(ref_trimmed[:n_ref], inp[:n_ref])
            si_sdr_out = _si_sdr_stereo(ref_trimmed[:n_ref], out_all[:n_ref])
            si_sdr_i = si_sdr_out - si_sdr_in
        else:
            si_sdr_in = si_sdr_out = si_sdr_i = None

        return {
            "mode": "file",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host_id": socket.gethostname(),
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "stft_pad_size_samples": stft_pad_size,
            "chunks": num_chunks,
            "warmup_chunks_excluded": actual_warmup,
            "rtf_avg": rtf_avg,
            "latency_ms_avg": latency_ms_avg,
            "latency_ms_p50": latency_ms_p50,
            "latency_ms_p95": latency_ms_p95,
            "latency_ms_p99": latency_ms_p99,
            "latency_ms_max": latency_ms_max,
            "drops_input": 0,
            "drops_output": 0,
            "underruns": 0,
            "nan_count": nan_count,
            "clip_ratio": clip_ratio,
            "rms_in": rms_in,
            "rms_out": rms_out,
            "cosine_similarity_before": cos_before,
            "cosine_similarity_after": cos_after,
            "cosine_similarity_delta": cos_delta,
            "si_sdr_input": si_sdr_in,
            "si_sdr_output": si_sdr_out,
            "si_sdr_improvement": si_sdr_i,
        }
