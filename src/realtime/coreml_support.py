"""CoreML inference wrapper for streaming TFGridNet.

Provides CoreMLModel — a drop-in replacement for Net that runs inference via a
pre-exported .mlpackage (targeting Apple Neural Engine on M-series Macs).

Export mode "nn_only" (current default):
  - STFT / iSTFT run on CPU via asteroid_filterbanks (same as training).
  - The neural-network backbone (Conv2d + GridNetBlocks + ConvTranspose2d) runs
    via CoreML / ANE.
  - This avoids asteroid_filterbanks ops that coremltools cannot lower.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Ordered keys for each GridNetBlock state buffer (must match export_coreml.py)
_BLOCK_STATE_KEYS = ("K_buf", "V_buf", "c0", "h0")


class CoreMLModel:
    """Wraps a CoreML .mlpackage for streaming inference.

    Implements the same interface as ``Net``:
      - ``predict(audio, embed, state, pad, lookahead_audio) -> (output, new_state)``
      - ``init_buffers(batch_size, device) -> state_dict``
      - ``stft_pad_size`` attribute

    State is kept on CPU (CoreML / ANE requires numpy inputs).  Audio/embed
    tensors may be on any device — the wrapper converts internally.

    Supported export modes (stored in the ``_meta.json`` sidecar):
      - ``nn_only``: STFT/iSTFT in Python, NN backbone in CoreML.
    """

    def __init__(self, mlpackage_path: Path):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*scikit-learn.*")
                warnings.filterwarnings("ignore", message=".*Torch version.*")
                import coremltools as ct
            self._ct = ct
        except ImportError as exc:
            raise ImportError(
                "coremltools is required for CoreML inference. "
                "Install with: pip install coremltools"
            ) from exc

        mlpackage_path = Path(mlpackage_path)
        if not mlpackage_path.exists():
            raise FileNotFoundError(f"CoreML model not found: {mlpackage_path}")

        meta_path = mlpackage_path.parent / (mlpackage_path.stem + "_meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"CoreML metadata not found: {meta_path}\n"
                "Re-run export_coreml.py to regenerate."
            )
        with meta_path.open() as f:
            meta = json.load(f)

        mode = meta.get("mode", "nn_only")
        if mode != "nn_only":
            raise ValueError(f"Unsupported CoreML export mode: '{mode}'. Only 'nn_only' is supported.")

        self._n_blocks: int = meta["n_blocks"]
        self._n_srcs: int = meta["n_srcs"]
        self._nn_state_names: list[str] = meta["nn_state_names"]
        self._nn_state_shapes: dict[str, list[int]] = meta["nn_state_shapes"]
        self._full_state_shapes: dict[str, list[int]] = meta["full_state_shapes"]
        self._output_names: list[str] = meta["output_names"]
        self.stft_pad_size: int = meta["stft_pad_size"]
        self._stft_chunk_size: int = meta["stft_chunk_size"]
        self._n_fft: int = meta["n_fft"]
        self._n_freqs: int = meta["n_freqs"]
        self._istft_lookback: int = meta["istft_lookback"]

        logger.info("Loading CoreML model from %s ...", mlpackage_path)
        self._mlmodel = ct.models.MLModel(str(mlpackage_path))
        logger.info("CoreML model loaded.")

        # Create STFT encoder / iSTFT decoder on CPU (deterministic from params)
        try:
            from asteroid_filterbanks import make_enc_dec
            enc_params = meta["enc_dec_params"]
            self._enc, self._dec = make_enc_dec(
                "stft",
                n_filters=enc_params["n_fft"],
                kernel_size=enc_params["n_fft"],
                stride=enc_params["stride"],
                window_type=enc_params["window_type"],
            )
            self._enc.eval()
            self._dec.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create STFT encoder/decoder: {exc}\n"
                "Ensure asteroid_filterbanks is installed."
            ) from exc

    # ------------------------------------------------------------------
    # Public interface (mirrors Net)
    # ------------------------------------------------------------------

    def init_buffers(self, batch_size: int, device=None) -> dict:
        """Return zero-initialised state dict on CPU (device arg is ignored)."""
        if batch_size != 1:
            raise ValueError("CoreMLModel only supports batch_size=1")
        shapes = self._full_state_shapes
        state: dict = {
            "conv_buf": torch.zeros(shapes["conv_buf"]),
            "deconv_buf": torch.zeros(shapes["deconv_buf"]),
            "istft_buf": torch.zeros(shapes["istft_buf"]),
            "gridnet_bufs": {},
        }
        for i in range(self._n_blocks):
            state["gridnet_bufs"][f"buf{i}"] = {
                key: torch.zeros(shapes[f"b{i}_{key}"])
                for key in _BLOCK_STATE_KEYS
            }
        return state

    def predict(
        self,
        audio: torch.Tensor,
        embed: torch.Tensor,
        input_state: dict,
        pad: bool = True,
        lookahead_audio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run one streaming chunk through CoreML (NN-only mode).

        Steps:
          1. Padding + concat lookahead  (CPU, mirrors Net.predict)
          2. STFT encoding               (CPU, asteroid_filterbanks)
          3. CoreML NN backbone          (ANE / CPU)
          4. Post-NN reshape             (CPU)
          5. iSTFT decoding              (CPU, asteroid_filterbanks)
          6. Output trimming             (CPU, mirrors Net.predict)
        """
        from src.models.tfgridnet_realtime.net import mod_pad

        audio_cpu = audio.detach().cpu().float()
        la_cpu = lookahead_audio.detach().cpu().float() if lookahead_audio is not None else None

        # Step 1: Padding (same logic as Net.predict)
        mod = 0
        if pad:
            if la_cpu is not None:
                audio_cpu, mod = mod_pad(audio_cpu, self._stft_chunk_size, (0, 0))
                audio_cpu = torch.cat([audio_cpu, la_cpu], dim=-1)
            else:
                audio_cpu, mod = mod_pad(
                    audio_cpu, self._stft_chunk_size, (0, self.stft_pad_size)
                )

        # Step 2: STFT encoding
        with torch.no_grad():
            batch = self._enc(audio_cpu)                             # [1, M, n_fft+2, T]
            batch = torch.cat(
                (batch[..., :self._n_freqs, :], batch[..., self._n_freqs:, :]), dim=1
            )                                                        # [1, 2M, n_freqs, T]
            stft_feats = batch.transpose(2, 3)                      # [1, 2M, T, n_freqs]

        # Step 3: CoreML NN backbone
        cml_inputs: dict = {
            "stft_feats": stft_feats.numpy(),
            "embed": embed.detach().cpu().float().numpy(),
            "conv_buf": input_state["conv_buf"].numpy(),
            "deconv_buf": input_state["deconv_buf"].numpy(),
        }
        for i in range(self._n_blocks):
            buf = input_state["gridnet_bufs"][f"buf{i}"]
            for key in _BLOCK_STATE_KEYS:
                cml_inputs[f"b{i}_{key}"] = buf[key].numpy()

        raw = self._mlmodel.predict(cml_inputs)

        # Parse NN outputs
        # output_names[0] = "deconv_out", [1..2] = "new_conv_buf", "new_deconv_buf",
        # [3 + i*4 .. 3 + i*4+3] = block state tensors
        deconv_out = torch.from_numpy(
            np.array(raw[self._output_names[0]], dtype=np.float32)
        )                                                            # [1, n_srcs*2, T, F]

        new_state: dict = {
            "conv_buf": torch.from_numpy(np.array(raw[self._output_names[1]], dtype=np.float32)),
            "deconv_buf": torch.from_numpy(np.array(raw[self._output_names[2]], dtype=np.float32)),
            "istft_buf": input_state["istft_buf"],  # updated in step 5
            "gridnet_bufs": {},
        }
        for i in range(self._n_blocks):
            base = 3 + i * len(_BLOCK_STATE_KEYS)
            new_state["gridnet_bufs"][f"buf{i}"] = {
                key: torch.from_numpy(
                    np.array(raw[self._output_names[base + j]], dtype=np.float32)
                )
                for j, key in enumerate(_BLOCK_STATE_KEYS)
            }

        # Step 4: Reshape deconv output (mirrors TFGridNet.forward post-deconv)
        n_batch, _, n_frames, n_freqs = deconv_out.shape
        batch = deconv_out.view(n_batch, self._n_srcs, 2, n_frames, n_freqs)
        batch = batch.transpose(3, 4)                               # [1, n_srcs, 2, F, T]
        batch = torch.cat([batch[:, :, 0], batch[:, :, 1]], dim=2) # [1, n_srcs, n_fft+2, T]

        # Step 5: iSTFT
        istft_buf = input_state["istft_buf"]
        batch = torch.cat([istft_buf, batch], dim=3)
        new_state["istft_buf"] = batch[..., -self._istft_lookback:]
        with torch.no_grad():
            batch = self._dec(batch)
        batch = batch[..., self._istft_lookback * self._stft_chunk_size:]

        # Step 6: Output trimming (mirrors Net.predict)
        output = batch[..., :-self.stft_pad_size]
        if mod != 0:
            output = output[:, :, :-mod]

        return output, new_state
