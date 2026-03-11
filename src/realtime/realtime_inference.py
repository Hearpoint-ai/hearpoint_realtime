#!/usr/bin/env python3
"""
Compatibility shim for the real-time TFGridNet inference module.

This module re-exports the legacy symbols from the new split modules so that
existing imports and script entrypoint usage continue to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Keep script-mode behavior compatible: running
# `python src/realtime/realtime_inference.py` must resolve `from src...` imports.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ml.factory import EMBEDDING_MODEL_IDS, create_embedding_model
from src.realtime.cli import main
from src.realtime.config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_THRESHOLDS_PATH,
    DEFAULT_YAML_CONFIG_PATH,
    REPO_ROOT,
    SCRIPT_DIR,
    SRC_DIR,
    AudioConfig,
    Config,
    DebugConfig,
    ModelConfig,
    NameDetectionConfig,
    OptimizationConfig,
    TestConfig,
    TRANSPARENCY_SOUND_PATH,
    _normalize_embedding_model_id,
)
from src.realtime.engine import RealtimeInference
from src.realtime.file_eval import FileBasedTest
from src.realtime.metrics import (
    _FILE_MODE_EXCLUDED,
    _THRESHOLD_OPS,
    _cosine_similarity,
    _ensure_stereo,
    _evaluate_thresholds,
    _load_threshold_profile,
    _si_sdr,
    _si_sdr_stereo,
    _write_report,
)

__all__ = [
    "main",
    "Config",
    "ModelConfig",
    "AudioConfig",
    "DebugConfig",
    "OptimizationConfig",
    "TestConfig",
    "NameDetectionConfig",
    "RealtimeInference",
    "FileBasedTest",
    "_normalize_embedding_model_id",
    "_load_threshold_profile",
    "_evaluate_thresholds",
    "_write_report",
    "_ensure_stereo",
    "_cosine_similarity",
    "_si_sdr",
    "_si_sdr_stereo",
    "EMBEDDING_MODEL_IDS",
    "create_embedding_model",
    "SCRIPT_DIR",
    "REPO_ROOT",
    "SRC_DIR",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_YAML_CONFIG_PATH",
    "TRANSPARENCY_SOUND_PATH",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_THRESHOLDS_PATH",
    "_THRESHOLD_OPS",
    "_FILE_MODE_EXCLUDED",
]


if __name__ == "__main__":
    main()
