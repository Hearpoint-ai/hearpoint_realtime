"""
ONNX export wrapper for TFGridNet.

Flattens the nested state dict into a flat list of tensors for ONNX I/O,
and provides utilities to convert between flat and nested representations.
"""

import torch
import torch.nn as nn

from .net import Net

# Canonical ordering of state tensors
N_BLOCKS = 3
TOP_STATE_KEYS = ["conv_buf", "deconv_buf", "istft_buf"]
BLOCK_STATE_KEYS = ["K_buf", "V_buf", "h0", "c0"]

# ONNX input/output naming
STATE_INPUT_NAMES = (
    [f"state_{k}" for k in TOP_STATE_KEYS]
    + [f"state_buf{i}_{k}" for i in range(N_BLOCKS) for k in BLOCK_STATE_KEYS]
)
STATE_OUTPUT_NAMES = (
    [f"out_{k}" for k in TOP_STATE_KEYS]
    + [f"out_buf{i}_{k}" for i in range(N_BLOCKS) for k in BLOCK_STATE_KEYS]
)
INPUT_NAMES = ["x", "embed", "lookahead"] + STATE_INPUT_NAMES
OUTPUT_NAMES = ["audio_out"] + STATE_OUTPUT_NAMES


def flatten_state(state_dict: dict) -> list[torch.Tensor]:
    """Extract state tensors in canonical order from nested dict."""
    flat = []
    for k in TOP_STATE_KEYS:
        flat.append(state_dict[k])
    for i in range(N_BLOCKS):
        buf = state_dict["gridnet_bufs"][f"buf{i}"]
        for k in BLOCK_STATE_KEYS:
            flat.append(buf[k])
    return flat


def unflatten_state(flat_list: list[torch.Tensor]) -> dict:
    """Rebuild nested state dict from flat list of tensors."""
    state = {}
    idx = 0
    for k in TOP_STATE_KEYS:
        state[k] = flat_list[idx]
        idx += 1
    gridnet_bufs = {}
    for i in range(N_BLOCKS):
        buf = {}
        for k in BLOCK_STATE_KEYS:
            buf[k] = flat_list[idx]
            idx += 1
        gridnet_bufs[f"buf{i}"] = buf
    state["gridnet_bufs"] = gridnet_bufs
    return state


def get_state_shapes(net: Net) -> list[tuple]:
    """Return shapes of all state tensors in canonical order (batch=1, CPU)."""
    state = net.init_buffers(1, "cpu")
    return [t.shape for t in flatten_state(state)]


class ExportWrapper(nn.Module):
    """Wraps Net.predict() for ONNX export with flat state I/O."""

    def __init__(self, net: Net):
        super().__init__()
        self.net = net

    def forward(self, x, embed, lookahead, *state_tensors):
        state = unflatten_state(list(state_tensors))
        audio_out, next_state = self.net.predict(
            x, embed, state, pad=True, lookahead_audio=lookahead
        )
        return (audio_out, *flatten_state(next_state))
