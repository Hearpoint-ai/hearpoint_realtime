from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.ml.factory import EMBEDDING_MODEL_IDS


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"

# Default model/runtime config paths
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "tfgridnet_cipic.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "weights" / "tfgridnet.ckpt"
DEFAULT_YAML_CONFIG_PATH = SCRIPT_DIR / "config.yaml"
TRANSPARENCY_SOUND_PATH = REPO_ROOT / "static" / "transparency-sound-effect.wav"

# Audio defaults
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 128  # 8ms at 16kHz

# Threshold config path
DEFAULT_THRESHOLDS_PATH = SCRIPT_DIR / "thresholds.yaml"


def _normalize_embedding_model_id(model_id: str | None) -> str:
    normalized = (model_id or "resemblyzer").strip().lower()
    if normalized not in EMBEDDING_MODEL_IDS:
        allowed = ", ".join(EMBEDDING_MODEL_IDS)
        raise ValueError(f"Unknown embedding model '{model_id}'. Allowed: {allowed}")
    return normalized


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
    output_gain: float = 1.0  # linear multiplier applied to output audio


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
        nd_model_path = (
            to_path(nd_model)
            if nd_model
            else REPO_ROOT / "src" / "models" / "vosk-model-small-en-us-0.15"
        )

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
                output_gain=audio_data.get("output_gain", 1.0),
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
