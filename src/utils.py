from __future__ import annotations

import os
import re
import torch


def get_torch_device() -> torch.device:
    """
    Choose the best available torch device in order: CUDA, MPS, CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sanitize_for_filename(name: str, fallback: str = "output") -> str:
    """
    Make a string safe to use as a filename component by removing path separators and
    collapsing disallowed characters to underscores.
    """
    name = name.replace(os.sep, "_").replace("/", "_").strip()
    safe = re.sub(r"[^\w.\-]+", "_", name)
    safe = safe.strip("._-")
    return safe or fallback
