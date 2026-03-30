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
    input_gain: float = 1.0  # linear multiplier applied to input audio before model inference


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
    use_coreml: bool = False
    coreml_model_path: Path | None = None


@dataclass
class TestConfig:
    """File-based test configuration."""

    enabled: bool = False
    input_file: Path | None = None
    output_dir: Path | None = None
    reference_file: Path | None = None
    report_dir: Path | None = None
    threshold_profile: str | None = None
    warmup_chunks: int = 10
    generate_plots: bool = False


@dataclass
class NameDetectionConfig:
    """Name/target-word detection via Vosk (runs on a separate thread from same input stream)."""

    enabled: bool = False
    model_path: Path | None = None
    target_word: str = "matthew"


@dataclass
class LoggingConfig:
    """Performance logging configuration."""

    enabled: bool = False
    log_dir: Path = field(default_factory=lambda: Path("logs/realtime"))


@dataclass
class ControllerConfig:
    """Maps key identifiers to command names (loaded from the 'controller' YAML section)."""

    bindings: dict[str, str] = field(default_factory=dict)


@dataclass
class EnrollmentConfig:
    """Enrollment-related configuration."""

    use_beamformer: bool = True


@dataclass
class SpectralSubtractionConfig:
    """Realtime spectral subtraction post-processing configuration."""

    enabled: bool = False
    noise_profile_path: Path | None = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    n_fft: int = 512
    hop_length: int = DEFAULT_CHUNK_SIZE
    win_length: int = 512
    alpha: float = 0.95


@dataclass
class AutoResetConfig:
    """Automatic state poisoning detection and reset configuration."""

    enabled: bool = True
    input_floor: float = 0.01  # min input level to consider "active audio"
    ratio_threshold: float = 0.05  # output/input below this = suspected poisoning
    consecutive_chunks: int = 100  # ~800ms sustained suppression before triggering
    cooldown_chunks: int = 125  # ~1s cooldown after reset
    activity_window_chunks: int = 375  # ~3s lookback for recent output activity
    activity_threshold: float = 0.15  # output/input ratio above this = "model was active"


@dataclass
class NoiseGateConfig:
    """Post-model noise gate to suppress interferer leakage when target is silent."""

    enabled: bool = False
    energy_threshold: float = 0.005  # output peak below this → gate closes
    attack_chunks: int = 2  # ~16ms — ramp up to open
    hold_chunks: int = 15  # ~120ms — stay open between words
    release_chunks: int = 10  # ~80ms — ramp down to closed
    smooth_coeff: float = 0.3  # envelope smoothing (0=instant, 1=frozen)


@dataclass
class Config:
    """Complete configuration for real-time inference."""

    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    test: TestConfig = field(default_factory=TestConfig)
    name_detection: NameDetectionConfig = field(default_factory=NameDetectionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    enrollment: EnrollmentConfig = field(default_factory=EnrollmentConfig)
    spectral_subtraction: SpectralSubtractionConfig = field(default_factory=SpectralSubtractionConfig)
    auto_reset: AutoResetConfig = field(default_factory=AutoResetConfig)
    noise_gate: NoiseGateConfig = field(default_factory=NoiseGateConfig)

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
        logging_data = data.get("logging", {}) or {}
        controller_raw = data.get("controller", {}) or {}
        enrollment_data = data.get("enrollment", {}) or {}
        spectral_data = data.get("spectral_subtraction", {}) or {}
        auto_reset_data = data.get("auto_reset", {}) or {}
        noise_gate_data = data.get("noise_gate", {}) or {}

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
                input_gain=audio_data.get("input_gain", 1.0),
            ),
            debug=DebugConfig(
                verbose=debug_data.get("verbose", False),
                passthrough=debug_data.get("passthrough", False),
                save_dir=to_path(debug_data.get("save_dir")),
            ),
            optimization=OptimizationConfig(
                use_torch_compile=opt_data.get("use_torch_compile", False),
                use_coreml=opt_data.get("use_coreml", False),
                coreml_model_path=to_path(opt_data.get("coreml_model_path")),
            ),
            test=TestConfig(
                enabled=test_data.get("enabled", False),
                input_file=to_path(test_data.get("input_file")),
                output_dir=to_path(test_data.get("output_dir")),
                reference_file=to_path(test_data.get("reference_file")),
                report_dir=to_path(test_data.get("report_dir")),
                threshold_profile=test_data.get("threshold_profile") or None,
                warmup_chunks=test_data.get("warmup_chunks", 10),
                generate_plots=test_data.get("generate_plots", False),
            ),
            name_detection=NameDetectionConfig(
                enabled=name_detection_data.get("enabled", False),
                model_path=nd_model_path,
                target_word=name_detection_data.get("target_word", "matthew"),
            ),
            logging=LoggingConfig(
                enabled=logging_data.get("enabled", False),
                log_dir=to_path(logging_data.get("log_dir")) or REPO_ROOT / "logs/realtime",
            ),
            controller=ControllerConfig(
                bindings={str(k): str(v) for k, v in controller_raw.items()},
            ),
            enrollment=EnrollmentConfig(
                use_beamformer=enrollment_data.get("use_beamformer", True),
            ),
            spectral_subtraction=SpectralSubtractionConfig(
                enabled=spectral_data.get("enabled", False),
                noise_profile_path=to_path(spectral_data.get("noise_profile_path")),
                sample_rate=spectral_data.get("sample_rate", DEFAULT_SAMPLE_RATE),
                n_fft=spectral_data.get("n_fft", 512),
                hop_length=spectral_data.get("hop_length", DEFAULT_CHUNK_SIZE),
                win_length=spectral_data.get("win_length", 512),
                alpha=spectral_data.get("alpha", 0.95),
            ),
            auto_reset=AutoResetConfig(
                enabled=auto_reset_data.get("enabled", True),
                input_floor=auto_reset_data.get("input_floor", 0.01),
                ratio_threshold=auto_reset_data.get("ratio_threshold", 0.05),
                consecutive_chunks=auto_reset_data.get("consecutive_chunks", 100),
                cooldown_chunks=auto_reset_data.get("cooldown_chunks", 125),
                activity_window_chunks=auto_reset_data.get("activity_window_chunks", 375),
                activity_threshold=auto_reset_data.get("activity_threshold", 0.15),
            ),
            noise_gate=NoiseGateConfig(
                enabled=noise_gate_data.get("enabled", False),
                energy_threshold=noise_gate_data.get("energy_threshold", 0.005),
                attack_chunks=noise_gate_data.get("attack_chunks", 2),
                hold_chunks=noise_gate_data.get("hold_chunks", 15),
                release_chunks=noise_gate_data.get("release_chunks", 10),
                smooth_coeff=noise_gate_data.get("smooth_coeff", 0.3),
            ),
        )

    def get_checkpoint_path(self) -> Path:
        """Get checkpoint path with default fallback."""
        return self.model.checkpoint or DEFAULT_CHECKPOINT_PATH

    def get_model_config_path(self) -> Path:
        """Get model config path with default fallback."""
        return self.model.config or DEFAULT_CONFIG_PATH
