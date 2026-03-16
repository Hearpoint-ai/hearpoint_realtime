from __future__ import annotations

import argparse
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import sounddevice as sd

from src.ml.factory import EMBEDDING_MODEL_IDS

from .config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_YAML_CONFIG_PATH,
    SCRIPT_DIR,
    Config,
    _normalize_embedding_model_id,
)
from .engine import RealtimeInference
from .file_eval import FileBasedTest
from .metrics import _evaluate_thresholds, _load_threshold_profile, _write_report
from .perf_logger import PerformanceLogger


_BANNER = r"""
 __    __                                                    __             __                   __
|  \  |  \                                                  |  \           |  \                 |  \
| $  | $  ______    ______    ______    ______    ______   \$ _______  _| $_        ______   \$
| $__| $ /      \  |      \  /      \  /      \  /      \ |  \|       \|   $ \      |      \ |  \
| $    $|  $$$\  \$$$\|  $$$\|  $$$\|  $$$\| $| $$$$\\$$$       \$$$\| $
| $$$$| $    $ /      $| $   \$| $  | $| $  | $| $| $  | $ | $ __     /      $| $
| $  | $| $$$$|  $$$$| $      | $__/ $| $__/ $| $| $  | $ | $|  \ __|  $$$$| $
| $  | $ \$     \ \$    $| $      | $    $ \$    $| $| $  | $  \$  $|  \\$    $| $
 \$   \$  \$$$$  \$$$$ \$      | $$$$   \$$$  \$ \$   \$   \$$  \$ \$$$$ \$
                                        | $
                                        | $
                                         \$
"""

_TAGLINE = "  Real-time target-speech extraction  |  github.com/hearpoint"
_DIVIDER = "─" * 72


def _print_banner() -> None:
    print(_BANNER)
    print(_TAGLINE)
    print(_DIVIDER)
    print()


def main() -> None:
    # Ensure child processes use 'spawn' (default on macOS, explicit for portability)
    multiprocessing.set_start_method("spawn", force=True)

    _print_banner()

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
""",
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help=f"Path to model config JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (cpu, cuda, mps). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=None,
        help="Path to speaker embedding .npy file (overrides config.yaml)",
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
        help="Path to input audio file for file-based test mode",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Stop live mode after this many seconds and write report",
    )

    # Utility arguments
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
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
    if config.test.threshold_profile:
        profile_thresholds = _load_threshold_profile(config.test.threshold_profile)

    # Determine mode: file-based test or real-time
    if config.test.enabled:
        # File-based testing mode
        if config.test.input_file is None:
            parser.error("test input_file is required when test mode is enabled (set in config.yaml)")
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = config.test.output_dir or SCRIPT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{ts}.wav"

        tester = FileBasedTest(config)
        stats, plot_data = tester.process_file(
            config.test.input_file,
            output_path,
            warmup_chunks=config.test.warmup_chunks,
            reference_path=config.test.reference_file,
            generate_plots=config.test.generate_plots,
        )
        if config.test.generate_plots and plot_data is not None:
            from .plots import generate_plots as _generate_plots
            plot_out = Path(config.test.report_dir or "reports/eval") / "plots"
            _generate_plots(plot_data, stats, plot_out, ts=ts)
        mode = "file"
    else:
        # Real-time mode
        perf_logger: PerformanceLogger | None = None
        if config.logging.enabled:
            perf_logger = PerformanceLogger(config.logging.log_dir)
            perf_logger.start()
            print(f"Performance logging enabled: {perf_logger.log_path}")
        engine = RealtimeInference(config, logger=perf_logger)
        engine.warmup_chunks = config.test.warmup_chunks
        stats = engine.run(duration=args.duration)
        mode = "live"

    # Report and threshold evaluation
    failed: list[str] = []
    if profile_thresholds:
        failed = _evaluate_thresholds(stats, profile_thresholds, mode)

    if config.test.report_dir:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        report_path = Path(config.test.report_dir) / f"{ts}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_report(stats, report_path, failed)

    if failed:
        print(f"Exiting non-zero: {len(failed)} threshold(s) failed.")
        sys.exit(1)
