#!/usr/bin/env python3
"""
Interactive demo CLI for HearPoint's real-time speech extraction system.

Controls:
    [T] Toggle passthrough / isolation
    [E] Enroll speaker (5-second capture from live mic)
    [S] Select enrolled speaker
    [Q] Quit
"""

import argparse
import json as _json
import logging
import multiprocessing
import select
import sys
import termios
import threading
import time
import tty
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# Suppress coremltools version-compatibility log warnings (sklearn, torch)
# Must happen before any import that transitively loads coremltools.
logging.getLogger("coremltools").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MEDIA_DIR = REPO_ROOT / "media"
ENROLLMENTS_DIR = MEDIA_DIR / "enrollments"
DATA_FILE = MEDIA_DIR / "data.json"

sys.path.insert(0, str(REPO_ROOT))

from src.ml.factory import EMBEDDING_MODEL_IDS, create_embedding_model, embedding_model_class_name
from src.models import Speaker
from src.persistence import MediaJsonStore
from src.realtime.perf_logger import PerformanceLogger
from src.realtime.realtime_inference import Config, RealtimeInference, _ensure_stereo

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


ENROLLMENT_DURATION = 5.0  # seconds


def _compute_embedding_in_process(audio_2xN: np.ndarray, sample_rate: int, model_id: str) -> np.ndarray:
    """Run embedding computation in a child process (avoids GIL contention).

    Must be a module-level function so it can be pickled by ProcessPoolExecutor.
    Forces CPU to avoid MPS/GPU conflicts with the parent process.
    """
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    torch.set_default_device("cpu")

    model = create_embedding_model(model_id)
    return model.compute_embedding_from_array(audio_2xN, sample_rate)


class DemoApp:
    def __init__(self, config: Config, embedding_model_id: str, logger: PerformanceLogger | None = None):
        self.engine = RealtimeInference(config, logger=logger)
        self.store = MediaJsonStore(media_root=MEDIA_DIR, data_file=DATA_FILE)
        self.embedding_model_id = embedding_model_id
        self._embedding_executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )

        self.current_speaker: str | None = None
        self.enrolling = False
        self.enroll_start_time: float | None = None
        self._speaker_list: list[Speaker] = self._load_speakers()
        self.naming = False
        self._name_input_buffer = ""
        self.status_message = "Ready"
        self.running = False

    def _load_speakers(self) -> list[Speaker]:
        speakers, _, _ = self.store.load()
        return speakers

    def _render(self) -> Panel:
        # ── Build speaker lines for right column ──────────────────────────
        spk_header = Text("Enrolled Speakers", style="bold bright_cyan")
        spk_divider = Text("─" * 22, style="dim")
        if self._speaker_list:
            spk_rows = []
            for i, spk in enumerate(self._speaker_list[:9], 1):
                if spk.name == self.current_speaker:
                    spk_rows.append(Text(f"▶ [{i}] {spk.name}", style="bold cyan"))
                else:
                    spk_rows.append(Text(f"  [{i}] {spk.name}", style="cyan"))
        else:
            spk_rows = [Text("(none — press E to enroll)", style="dim")]

        # Pad to a fixed length so the right column height is stable
        while len(spk_rows) < 9:
            spk_rows.append(Text(""))

        # right_col[i] maps to table row i
        right_col = [spk_header, spk_divider] + spk_rows

        def _r(i: int) -> Text:
            return right_col[i] if i < len(right_col) else Text("")

        # ── 3-column table: label | value | speakers ──────────────────────
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("key", style="bold", width=12)
        table.add_column("value", width=44)
        table.add_column("right", width=30, no_wrap=True)

        if self.engine.passthrough_mode:
            mode_text = Text("PASSTHROUGH", style="bold green")
        else:
            mode_text = Text("ISOLATION", style="bold red")

        with self.engine.input_level_lock:
            level = self.engine.recent_input_level
        level_db = 20 * np.log10(level + 1e-10)
        bars = int(max(0, min(28, (level_db + 60) * 28 / 60)))
        meter = f"[{'█' * bars}{'░' * (28 - bars)}] {level_db:5.1f} dB"

        if self.engine.processing_times:
            recent = self.engine.processing_times[-100:]
            recent_ms = np.array(recent) * 1000
            chunk_ms = self.engine.chunk_size / self.engine.sample_rate * 1000
            avg = np.mean(recent_ms)
            rtf = avg / chunk_ms
            p95 = np.percentile(recent_ms, 95)
            stats = [
                ("RTF",      f"{rtf:.3f}"),
                ("Latency",  f"{avg:.1f}ms avg"),
                ("Chunks",   f"{self.engine.chunks_processed:,}"),
                ("p95",      f"{p95:.1f}ms"),
                ("Drops",    f"{self.engine.drops_input}/{self.engine.drops_output}"),
                ("Underruns",str(self.engine.underruns)),
            ]
        else:
            stats = [
                ("RTF",      "—"), ("Latency",  "—"), ("Chunks",   "0"),
                ("p95",      "—"), ("Drops",    "0/0"), ("Underruns","0"),
            ]

        controls = Text()
        for key, label, key_style in [
            ("[T]", " Toggle  ", "bold bright_cyan"),
            ("[E]", " Enroll  ", "bold bright_cyan"),
            ("[N]", " Name  ", "bold bright_cyan"),
            ("[Q]", " Quit",    "bold red"),
        ]:
            controls.append(key, style=key_style)
            controls.append(label)

        if self.naming:
            status_text = Text(f"Enter name: {self._name_input_buffer}_", style="bold yellow")
        elif self.enrolling and self.enroll_start_time is not None:
            elapsed = time.time() - self.enroll_start_time
            remaining = max(0, ENROLLMENT_DURATION - elapsed)
            status_text = Text(f"Enrolling... speak now ({remaining:.0f}s remaining)", style="bold yellow")
        else:
            status_text = Text(f"Status: {self.status_message}")

        # Row layout (right column index in parens)
        table.add_row("Mode",    mode_text,                              _r(0))
        table.add_row("Speaker", self.current_speaker or "(none enrolled)", _r(1))
        table.add_row("Level",   meter,                                  _r(2))
        table.add_row("",        "",                                     _r(3))
        for row_i, (lbl, val) in enumerate(stats):
            table.add_row(lbl, val, _r(4 + row_i))
        table.add_row("",        "",       _r(10))
        table.add_row("",        controls, _r(11))
        table.add_row("",        "",       _r(12) if len(right_col) > 12 else Text(""))
        table.add_row("",        status_text, Text(""))

        title = Text(" HearPoint AI ", style="bold bright_cyan reverse")
        panel = Panel(table, title=title, border_style="bright_cyan", width=100)
        return Group(Text(_BANNER, style="bold bright_cyan", no_wrap=True), panel)

    def _handle_key(self, ch: str) -> None:
        if self.enrolling:
            return  # ignore keys during enrollment

        if self.naming:
            if ch in ("\r", "\n"):  # Enter — confirm
                word = self._name_input_buffer.strip()
                if word:
                    self.engine.set_target_word(word)
                    self.status_message = f"Target name set to: {word}"
                else:
                    self.status_message = "Name unchanged (empty input)"
                self.naming = False
                self._name_input_buffer = ""
            elif ch == "\x1b":  # Escape — cancel
                self.naming = False
                self._name_input_buffer = ""
                self.status_message = "Name change cancelled"
            elif ch == "\x7f":  # Backspace
                self._name_input_buffer = self._name_input_buffer[:-1]
            elif ch.isprintable():
                self._name_input_buffer += ch
            return

        # Number keys directly select a speaker
        if ch.isdigit() and ch != "0":
            idx = int(ch) - 1
            if idx < len(self._speaker_list):
                spk = self._speaker_list[idx]
                emb = np.load(spk.embedding_path)
                self.engine.set_embedding(emb)
                self.engine.set_passthrough(False)
                self.current_speaker = spk.name
                self.status_message = f"Selected: {spk.name}. Isolation active."
            return

        if ch in ("q", "Q", "\x03"):  # q or Ctrl+C
            self.running = False
        elif ch in ("t", "T"):
            self._toggle_mode()
        elif ch in ("e", "E"):
            self._start_enrollment()
        elif ch in ("n", "N"):
            self.naming = True
            self._name_input_buffer = ""
            self.status_message = "Type a name and press Enter"

    def _toggle_mode(self) -> None:
        if self.engine.embedding is None:
            self.status_message = "Enroll a speaker first [E]"
            return
        is_passthrough = self.engine.passthrough_mode
        self.engine.set_passthrough(not is_passthrough)
        self.status_message = f"Mode: {'passthrough' if not is_passthrough else 'isolation'}"

    def _start_enrollment(self) -> None:
        self.enrolling = True
        self.enroll_start_time = time.time()
        self.status_message = "Enrolling..."
        self.engine.start_enrollment_capture(ENROLLMENT_DURATION)

        def _enrollment_worker():
            time.sleep(ENROLLMENT_DURATION)
            audio = self.engine.stop_enrollment_capture()
            self.enrolling = False
            self.enroll_start_time = None

            if audio is None or len(audio) == 0:
                self.status_message = "Enrollment failed: no audio captured"
                return

            try:
                self.status_message = "Computing embedding..."
                # Ensure stereo [N, 2] for persistence and embedding
                audio = _ensure_stereo(audio)

                # [N, 2] → [2, N] channels-first (standard contract)
                audio_2xN = audio.T.astype(np.float32)

                # Offload CPU-bound embedding to a child process (avoids GIL contention)
                future = self._embedding_executor.submit(
                    _compute_embedding_in_process,
                    audio_2xN,
                    self.engine.sample_rate,
                    self.embedding_model_id,
                )
                embedding = future.result()

                # Set embedding on engine and switch to isolation
                self.engine.set_embedding(embedding)
                self.engine.set_passthrough(False)

                # Persist enrollment
                speaker_id = str(uuid.uuid4())
                ENROLLMENTS_DIR.mkdir(parents=True, exist_ok=True)
                dest_audio = ENROLLMENTS_DIR / f"{speaker_id}.wav"
                dest_embedding = ENROLLMENTS_DIR / f"{speaker_id}.npy"

                import soundfile as sf
                sf.write(str(dest_audio), audio, self.engine.sample_rate)
                np.save(dest_embedding, embedding)

                # Write sidecar metadata
                sidecar = {
                    "embedding_model_id": self.embedding_model_id,
                    "embedding_model_class": embedding_model_class_name(self.embedding_model_id),
                    "sample_rate": self.engine.sample_rate,
                }
                dest_embedding.with_suffix(".meta.json").write_text(_json.dumps(sidecar, indent=2))

                # Update data.json
                name = f"Speaker {datetime.now().strftime('%H:%M:%S')}"
                speakers, recordings, extractions = self.store.load()
                speaker = Speaker(
                    id=speaker_id,
                    name=name,
                    created_at=datetime.now(timezone.utc),
                    embedding_path=dest_embedding,
                    enrollment_audio_path=dest_audio,
                )
                speakers.append(speaker)
                self.store.save(speakers, recordings, extractions)

                self.current_speaker = name
                self.status_message = f"Enrolled: {name}. Isolation active."
                self._speaker_list = self._load_speakers()
            except Exception as e:
                self.status_message = f"Enrollment error: {e}"

        threading.Thread(target=_enrollment_worker, daemon=True).start()

    def _keypress_thread(self) -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            # cbreak mode: disables line buffering + echo for single-char reads,
            # but keeps OPOST enabled so Rich's ANSI output works correctly.
            tty.setcbreak(fd)
            while self.running:
                rlist, _, _ = select.select([fd], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch:
                        self._handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def run(self) -> None:
        console = Console()
        try:
            self.engine.start()
            self.running = True

            key_thread = threading.Thread(target=self._keypress_thread, daemon=True)
            key_thread.start()

            with Live(self._render(), refresh_per_second=5, console=console, screen=True) as live:
                while self.running:
                    time.sleep(0.2)
                    live.update(self._render())
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.engine.stop()
            self._embedding_executor.shutdown(wait=False)


_BANNER = r"""
 __    __                                                    __             __                   __ 
|  \  |  \                                                  |  \           |  \                 |  \
| $$  | $$  ______    ______    ______    ______    ______   \$$ _______  _| $$_        ______   \$$
| $$__| $$ /      \  |      \  /      \  /      \  /      \ |  \|       \|   $$ \      |      \ |  \
| $$    $$|  $$$$$$\  \$$$$$$\|  $$$$$$\|  $$$$$$\|  $$$$$$\| $$| $$$$$$$\\$$$$$$       \$$$$$$\| $$
| $$$$$$$$| $$    $$ /      $$| $$   \$$| $$  | $$| $$  | $$| $$| $$  | $$ | $$ __     /      $$| $$
| $$  | $$| $$$$$$$$|  $$$$$$$| $$      | $$__/ $$| $$__/ $$| $$| $$  | $$ | $$|  \ __|  $$$$$$$| $$
| $$  | $$ \$$     \ \$$    $$| $$      | $$    $$ \$$    $$| $$| $$  | $$  \$$  $$|  \\$$    $$| $$
 \$$   \$$  \$$$$$$$  \$$$$$$$ \$$      | $$$$$$$   \$$$$$$  \$$ \$$   \$$   \$$$$  \$$ \$$$$$$$ \$$
                                        | $$                                                        
                                        | $$                                                        
                                         \$$                                                        
"""


def main():
    # Ensure child processes use 'spawn' (default on macOS, explicit for portability)
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="HearPoint interactive real-time demo")
    parser.add_argument(
        "--embedding-model",
        choices=EMBEDDING_MODEL_IDS,
        default="resemblyzer",
        help="Speaker embedding model (default: resemblyzer)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "src" / "realtime" / "config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    # Force passthrough + no embedding for demo startup
    config.debug.passthrough = True
    config.model.embedding = None

    perf_logger: PerformanceLogger | None = None
    if config.logging.enabled:
        perf_logger = PerformanceLogger(config.logging.log_dir)
        perf_logger.start()

    app = DemoApp(config, args.embedding_model, logger=perf_logger)
    app.run()


if __name__ == "__main__":
    main()
