"""
Compare three eval JSON reports on SI-SDR improvement only.

The x-axis labels are intentionally hardcoded as:
- resemblyzer
- beamformer network
- knowledge-distillation
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a 3-model SI-SDR improvement comparison plot from eval reports."
    )
    parser.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to exactly 3 eval JSON report files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to reports/eval/plots/"
            "<timestamp>_model_compare_sisdr.png"
        ),
    )
    return parser


def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    reports: list[Path] = args.reports
    if len(reports) != 3:
        parser.error("Provide exactly 3 report files for comparison.")

    labels = ["resemblyzer", "beamformer network", "knowledge-distillation"]

    si_sdr_improvement_vals: list[float] = []

    for path in reports:
        if not path.exists():
            raise FileNotFoundError(f"Report not found: {path}")
        report = _read_report(path)

        imp_val = report.get("si_sdr_improvement")
        if imp_val is None:
            raise ValueError(
                f"Missing SI-SDR fields in {path}. "
                "Expected key: si_sdr_improvement."
            )
        si_sdr_improvement_vals.append(float(imp_val))

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = args.output or Path("reports/eval/plots") / f"{ts}_model_compare_sisdr.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_imp = ax.bar(
        x,
        si_sdr_improvement_vals,
        width=0.55,
        color="darkorange",
    )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("SI-SDR improvement (dB)")
    ax.set_title("Enrollment model comparison")

    for bar in bars_imp:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3 if h >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    print(f"Saved comparison plot: {out_path}")


if __name__ == "__main__":
    _main()
