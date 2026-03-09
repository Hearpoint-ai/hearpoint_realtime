#!/usr/bin/env python3
"""
Export TFGridNet to ONNX (and optionally TensorRT engine).

Usage:
    python scripts/export_engine.py                   # ONNX only
    python scripts/export_engine.py --verify           # ONNX + numerical check
    python scripts/export_engine.py --trtexec          # ONNX + TRT engine via trtexec
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import asteroid_filterbanks.enc_dec as _af_enc_dec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.tfgridnet_realtime.net import Net
from src.models.tfgridnet_realtime.export_wrapper import (
    ExportWrapper, INPUT_NAMES, OUTPUT_NAMES, flatten_state,
)


def _patched_multishape_conv1d(
    waveform: torch.Tensor,
    filters: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    as_conv1d: bool = True,
) -> torch.Tensor:
    """Patched encoder conv without @script_if_tracing.

    Removing the decorator lets torch.jit.trace record only the executed
    branch, so the output rank is statically known in the ONNX graph.
    """
    if waveform.ndim == 1:
        return F.conv1d(
            waveform[None, None], filters, stride=stride, padding=padding,
        ).squeeze()
    elif waveform.ndim == 2:
        return F.conv1d(
            waveform.unsqueeze(1), filters, stride=stride, padding=padding,
        )
    elif waveform.ndim == 3:
        batch, channels, time_len = waveform.shape
        if channels == 1 and as_conv1d:
            return F.conv1d(waveform, filters, stride=stride, padding=padding)
        else:
            batched_conv = F.conv1d(
                waveform.view(-1, 1, waveform.shape[-1]),
                filters, stride=stride, padding=padding,
            )
            return batched_conv.view(
                waveform.shape[0], waveform.shape[1],
                batched_conv.shape[1], batched_conv.shape[2],
            )
    else:
        batched_conv = F.conv1d(
            waveform.view(-1, 1, waveform.shape[-1]),
            filters, stride=stride, padding=padding,
        )
        return batched_conv.view(
            waveform.shape[0], waveform.shape[1],
            batched_conv.shape[1], batched_conv.shape[2],
        )


def _patched_multishape_conv_transpose1d(
    spec: torch.Tensor,
    filters: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> torch.Tensor:
    """Patched version with explicit rank for ONNX-safe tracing.

    The 4D branch uses explicit integer indices in .view() so that the
    output rank is statically known (mirrors encoder-side fix in
    batch_packed_1d_conv).
    """
    if spec.ndim == 2:
        return F.conv_transpose1d(
            spec.unsqueeze(0), filters,
            stride=stride, padding=padding, output_padding=output_padding,
        ).squeeze()
    if spec.ndim == 3:
        return F.conv_transpose1d(
            spec, filters,
            stride=stride, padding=padding, output_padding=output_padding,
        )
    else:
        # 4D input: [B, n_srcs, nfft+2, T]
        b, s = spec.shape[0], spec.shape[1]
        out = F.conv_transpose1d(
            spec.reshape(-1, spec.shape[2], spec.shape[3]), filters,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        # Explicit integer indices so tracer knows output rank = 3
        return out.view(b, s, out.shape[-1])


def load_model(checkpoint_path: Path, config_path: Path) -> Net:
    """Load Net from config + checkpoint (same logic as RealtimeInference)."""
    with config_path.open() as fp:
        config = json.load(fp)
    model_params = config["pl_module_args"]["model_params"]

    net = Net(**model_params)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint
    )

    # Strip common wrapper prefixes
    for pref in ("model.model.", "model.", "module."):
        if all(k.startswith(pref) for k in state_dict.keys()):
            state_dict = {k[len(pref):]: v for k, v in state_dict.items()}
            break

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")

    net.eval()
    return net


def export_onnx(net: Net, output_path: Path, opset: int = 17) -> None:
    """Export wrapped model to ONNX."""
    wrapper = ExportWrapper(net)
    wrapper.eval()

    # Build dummy inputs
    chunk_size = net.stft_chunk_size
    pad_size = net.stft_pad_size
    state = net.init_buffers(1, "cpu")
    flat_state = flatten_state(state)

    dummy_x = torch.zeros(1, 2, chunk_size)
    dummy_embed = torch.zeros(1, 256)
    dummy_la = torch.zeros(1, 2, pad_size)
    dummy_inputs = (dummy_x, dummy_embed, dummy_la, *flat_state)

    # Monkey-patch encoder/decoder to remove @script_if_tracing (fixes ONNX export).
    # The decorator causes JIT-scripting of all branches, producing unknown-rank
    # outputs. Plain tracing only records the executed branch → rank is known.
    _af_enc_dec.multishape_conv1d = _patched_multishape_conv1d
    _af_enc_dec.multishape_conv_transpose1d = _patched_multishape_conv_transpose1d

    print(f"Exporting ONNX to {output_path} ...")
    print(f"  Inputs ({len(INPUT_NAMES)}): {INPUT_NAMES}")
    print(f"  Outputs ({len(OUTPUT_NAMES)}): {OUTPUT_NAMES}")

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        opset_version=opset,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes=None,
    )

    # Validate
    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print("ONNX model validated successfully.")


def verify_onnx(net: Net, onnx_path: Path) -> None:
    """Run one chunk through PyTorch and ONNX, compare outputs."""
    import onnxruntime as ort

    chunk_size = net.stft_chunk_size
    pad_size = net.stft_pad_size
    state = net.init_buffers(1, "cpu")
    flat_state = flatten_state(state)

    # Random inputs for better coverage
    x = torch.randn(1, 2, chunk_size)
    embed = torch.randn(1, 256)
    la = torch.randn(1, 2, pad_size)

    # PyTorch reference
    wrapper = ExportWrapper(net)
    wrapper.eval()
    with torch.no_grad():
        pt_outputs = wrapper(x, embed, la, *flat_state)

    # ONNX Runtime
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {
        "x": x.numpy(),
        "embed": embed.numpy(),
        "lookahead": la.numpy(),
    }
    for name, tensor in zip(INPUT_NAMES[3:], flat_state):
        feed[name] = tensor.numpy()

    ort_outputs = sess.run(None, feed)

    # Compare
    max_diff = 0.0
    for i, (pt_out, ort_out) in enumerate(zip(pt_outputs, ort_outputs)):
        diff = np.abs(pt_out.numpy() - ort_out).max()
        max_diff = max(max_diff, diff)
        name = OUTPUT_NAMES[i]
        ok = "OK" if diff < 1e-4 else "MISMATCH"
        print(f"  {name}: max_diff={diff:.2e} [{ok}]")

    if max_diff < 1e-4:
        print("Verification PASSED.")
    else:
        print(f"Verification FAILED (max diff={max_diff:.2e}).")
        sys.exit(1)


def _find_trtexec() -> str:
    """Locate trtexec binary, checking PATH then common Jetson install paths."""
    import shutil
    path = shutil.which("trtexec")
    if path:
        return path
    for candidate in ("/usr/src/tensorrt/bin/trtexec",):
        if Path(candidate).is_file():
            return candidate
    return "trtexec"  # fallback, will fail with a clear error


def build_trt_engine(onnx_path: Path) -> None:
    """Build TensorRT engine using trtexec CLI."""
    engine_path = onnx_path.with_suffix(".engine")
    trtexec = _find_trtexec()
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--memPoolSize=workspace:1024",
    ]
    print(f"Building TRT engine: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("trtexec failed. Is TensorRT installed?")
        sys.exit(1)
    print(f"TRT engine saved to {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Export TFGridNet to ONNX/TRT")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "weights" / "tfgridnet.ckpt")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "src" / "configs" / "tfgridnet_cipic.json")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "weights" / "tfgridnet.onnx")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX output matches PyTorch")
    parser.add_argument("--trtexec", action="store_true",
                        help="Build TensorRT engine via trtexec")
    args = parser.parse_args()

    net = load_model(args.checkpoint, args.config)
    export_onnx(net, args.output)

    if args.verify:
        print("\nVerifying ONNX vs PyTorch...")
        verify_onnx(net, args.output)

    if args.trtexec:
        build_trt_engine(args.output)


if __name__ == "__main__":
    main()
