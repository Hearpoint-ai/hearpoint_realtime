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
import os
import select
import sys
import termios
import threading
import time
import tty
import uuid
import warnings
from collections import deque
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
from src.realtime.file_eval import FileBasedTest
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

        self._config = config
        self._focused_idx: int = 0

        self.current_speaker: str | None = None
        self.enrolling = False
        self.enroll_start_time: float | None = None
        self._speaker_list: list[Speaker] = self._load_speakers()
        self.naming = False
        self._name_input_buffer = ""
        self.status_message = "Ready"
        self.running = False
        self._last_ar_count = 0
        self._show_gate_debug = False
        self._input_history: deque[float] = deque(maxlen=200)
        self._output_history: deque[float] = deque(maxlen=200)

    def _load_speakers(self) -> list[Speaker]:
        speakers, _, _ = self.store.load()
        return speakers

    @staticmethod
    def _sparkline(values, width: int) -> str:
        """Block bar sparkline for waveform display."""
        blocks = "▁▂▃▄▅▆▇█"
        if not values:
            return blocks[0] * width
        vals = list(values)
        if len(vals) > width:
            step = len(vals) / width
            vals = [vals[int(i * step)] for i in range(width)]
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-10:
            chars = blocks[0] * len(vals)
        else:
            chars = "".join(blocks[min(int((v - mn) / (mx - mn) * 7.99), 7)] for v in vals)
        pad = width - len(chars)
        return (blocks[0] * pad) + chars


    def _render(self) -> Panel:
        # ── Dynamic terminal width ───────────────────────────────────────
        MIN_WIDTH = 80
        term_w = max(MIN_WIDTH, self._console.width if hasattr(self, '_console') else 100)
        inner_w = term_w - 4           # panel border + padding
        table_padding = 12             # 3 cols × 2 sides × 2 chars
        avail = inner_w - table_padding
        col_key = max(7, int(avail * 0.08))
        col_right = max(24, int(avail * 0.55))
        col_value = avail - col_key - col_right
        meter_max = max(8, col_value - 14)

        # ── Build speaker lines for right column ──────────────────────────
        spk_header = Text("Enrolled Speakers", style="bold bright_cyan")
        spk_divider = Text("─" * (col_right - 2), style="dim")
        if self._speaker_list:
            spk_rows = []
            focused = self._focused_idx % len(self._speaker_list)
            for i, spk in enumerate(self._speaker_list[:9], 1):
                is_active = spk.name == self.current_speaker
                is_focused = (i - 1) == focused
                if is_active:
                    prefix = "▶"
                    style = "bold cyan"
                elif is_focused:
                    prefix = "→"
                    style = "bold white"
                else:
                    prefix = " "
                    style = "cyan"
                spk_rows.append(Text(f"{prefix} [{i}] {spk.name}", style=style))
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
        table.add_column("key", style="bold", width=col_key)
        table.add_column("value", width=col_value)
        table.add_column("right", width=col_right, no_wrap=True)

        if self.engine.passthrough_mode:
            mode_text = Text("PASSTHROUGH", style="bold green")
        else:
            mode_text = Text("ISOLATION", style="bold red")

        with self.engine.input_level_lock:
            level = self.engine.recent_input_level
            out_level = self.engine.recent_output_level
        self._input_history.append(level)
        self._output_history.append(out_level)

        level_db = 20 * np.log10(level + 1e-10)
        bars = int(max(0, min(meter_max, (level_db + 60) * meter_max / 60)))
        meter = f"[{'█' * bars}{'░' * (meter_max - bars)}] {level_db:5.1f} dB"

        if self.engine.processing_times:
            recent = self.engine.processing_times[-100:]
            recent_ms = np.array(recent) * 1000
            chunk_ms = self.engine.chunk_size / self.engine.sample_rate * 1000
            avg = np.mean(recent_ms)
            rtf = avg / chunk_ms
            p95 = np.percentile(recent_ms, 95)
            stats = [
                ("RTF",      f"{rtf:.3f}"),
                ("Drops",    f"{self.engine.drops_input}/{self.engine.drops_output}"),
                ("Underruns",str(self.engine.underruns)),
            ]
            verbose_stats = [
                ("Chunks",       f"{self.engine.chunks_processed:,}"),
                ("p95 latency",  f"{p95:.1f}ms"),
            ]
        else:
            avg = 0.0
            stats = [
                ("RTF",      "—"), ("Drops",    "0/0"), ("Underruns","0"),
            ]
            verbose_stats = [
                ("Chunks",       "0"),
                ("p95 latency",  "—"),
            ]

        if self._show_gate_debug:
            _command_labels = {
                "next_speaker":       "Next Spk",
                "prev_speaker":       "Prev Spk",
                "select":             "Enter",
                "toggle_passthrough": "Toggle",
                "enroll":             "Enroll",
                "name":               "Name",
                "decrease_gain":      "Gain↓",
                "increase_gain":      "Gain↑",
                "reset_full":         "Full Reset",
                "clear_data":         "Clear Data",
                "quit":               "Quit",
                "esc":                "Quit",
            }
            _cmd_to_keys: dict[str, list[str]] = {"reset_full": ["F"], "clear_data": ["C"]}
            for _k, _cmd in self._config.controller.bindings.items():
                _cmd_to_keys.setdefault(_cmd, []).append(_k)

            _ctrl_entries: list[tuple[str, str, bool]] = []
            for _cmd, _label in _command_labels.items():
                _keys = _cmd_to_keys.get(_cmd, [])
                if not _keys:
                    continue
                _ctrl_entries.append(("/".join(_keys), _label, _cmd in ("quit", "esc")))

            # Lay out controls as a grid scaled to available width
            _COLS = max(2, min(4, col_value // 14))
            _COL_W = max(12, col_value // _COLS)
            controls = Table(show_header=False, box=None, padding=(0, 0))
            for _ in range(_COLS):
                controls.add_column(width=_COL_W, no_wrap=True)
            for _i in range(0, len(_ctrl_entries), _COLS):
                _row_items = _ctrl_entries[_i : _i + _COLS]
                _cells = []
                for _key_str, _label, _is_quit in _row_items:
                    _t = Text()
                    _t.append(f"[{_key_str}]", style="bold red" if _is_quit else "bold bright_cyan")
                    _t.append(f" {_label}")
                    _cells.append(_t)
                while len(_cells) < _COLS:
                    _cells.append(Text(""))
                controls.add_row(*_cells)
        else:
            _hint = Text()
            _hint.append("[V]", style="bold bright_cyan")
            _hint.append(" Show Menu")
            controls = _hint

        if self.naming:
            status_text = Text(f"Enter name: {self._name_input_buffer}_", style="bold yellow")
        elif self.enrolling and self.enroll_start_time is not None:
            elapsed = time.time() - self.enroll_start_time
            remaining = max(0, ENROLLMENT_DURATION - elapsed)
            status_text = Text(f"Enrolling... speak now ({remaining:.0f}s remaining)", style="bold yellow")
        else:
            status_text = Text(f"Status: {self.status_message}", style="dim")

        # Row layout (right column index in parens)
        ri = 0  # right column index
        table.add_row("Mode",    mode_text,                              _r(ri)); ri += 1
        table.add_row("",        "",                                     _r(ri)); ri += 1
        table.add_row("Speaker", self.current_speaker or "(none enrolled)", _r(ri)); ri += 1

        table.add_row("Level",   meter,                                  _r(ri)); ri += 1
        out_level_db = 20 * np.log10(out_level + 1e-10)
        out_bars = int(max(0, min(meter_max, (out_level_db + 60) * meter_max / 60)))
        out_meter = f"[{'█' * out_bars}{'░' * (meter_max - out_bars)}] {out_level_db:5.1f} dB"
        table.add_row("Output",  out_meter,                              _r(ri)); ri += 1
        gain = self.engine.output_gain
        _GAIN_STEPS = 20  # 0.0–10.0 in 0.5x steps
        gain_filled = round(gain * 2)  # each 0.5x step = 1 block
        gain_bar = f"[{'▮' * gain_filled}{' ' * (_GAIN_STEPS - gain_filled)}] {gain:.1f}x"
        table.add_row("Gain",    gain_bar,                               _r(ri)); ri += 1

        # Waveform sparkline
        spark_width = max(10, col_value - 2)
        waveform = self._sparkline(self._input_history, spark_width)
        table.add_row("Waveform", Text(waveform, style="green"),         _r(ri)); ri += 1

        # Voice activity indicator (from noise gate state — zero extra computation)
        if hasattr(self.engine, '_ng_diag_state'):
            gate_st = self.engine._ng_diag_state
            if gate_st in ("open", "hold"):
                vad_text = Text("● SPEECH", style="bold green")
            elif gate_st == "attack":
                vad_text = Text("● ONSET", style="bold yellow")
            else:
                vad_text = Text("○ SILENCE", style="dim")
        else:
            vad_text = Text("○ —", style="dim")
        table.add_row("Voice",   vad_text,                               _r(ri)); ri += 1

        table.add_row("",        "",                                     _r(ri)); ri += 1
        for lbl, val in stats:
            table.add_row(lbl, val, _r(ri)); ri += 1

        table.add_row("Latency", Text(f"{avg:.1f}ms avg" if avg else "—", style="dim" if not avg else ""), Text(""))

        # Gate debug rows (toggle with V key)
        if self._show_gate_debug and hasattr(self.engine, "_ng_diag_energy"):
            e = self.engine
            gate_state = e._ng_diag_state.upper()
            gate_style = (
                "bold green" if gate_state in ("OPEN", "HOLD")
                else "bold yellow" if gate_state == "ATTACK"
                else "bold red"
            )
            in_level = e.recent_input_level
            io_ratio = e._ng_diag_energy / (in_level + 1e-10) if in_level > 0 else 0.0
            for lbl, val in verbose_stats:
                table.add_row(lbl, val, Text(""))
            table.add_row("", "", Text(""))
            table.add_row("OutEnergy", f"{e._ng_diag_energy:.6f}", Text(""))
            table.add_row("Envelope",  f"{e._ng_diag_envelope:.6f}", Text(""))
            table.add_row("Threshold", f"{e._ng_threshold:.6f}", Text(""))
            table.add_row("Gate",      Text(gate_state, style=gate_style), Text(""))
            table.add_row("GateGain",  f"{e._ng_diag_gain:.2f}", Text(""))
            table.add_row("InLevel",   f"{in_level:.6f}", Text(""))
            table.add_row("IO Ratio",  f"{io_ratio:.4f}", Text(""))
            table.add_row("Resets",    f"{self.engine._ar_reset_count}", Text(""))

        table.add_row("",        "",       Text(""))
        table.add_row("",        controls, Text(""))
        table.add_row("",        status_text, Text(""))

        title = Text(" HearPoint AI ", style="bold bright_cyan reverse")
        panel = Panel(table, title=title, border_style="bright_cyan", width=term_w)
        if term_w >= 94:
            banner_display = Text(_BANNER, style="bold bright_cyan", no_wrap=True)
        else:
            banner_display = Text("  HearPoint.ai", style="bold bright_cyan")
        return Group(banner_display, panel)

    def _handle_key(self, ch: str) -> None:
        if self.enrolling:
            return  # ignore keys during enrollment

        if self.naming:
            if ch in ("\r", "\n"):  # Enter — confirm
                self._confirm_rename()
            elif ch == "\x1b":  # Escape — cancel
                self.naming = False
                self._name_input_buffer = ""
                self.status_message = "Rename cancelled"
            elif ch == "\x7f":  # Backspace
                self._name_input_buffer = self._name_input_buffer[:-1]
            elif ch.isprintable():
                self._name_input_buffer += ch
            return

        # Ctrl+C always quits regardless of bindings
        if ch == "\x03":
            self.running = False
            return

        # Direct keybinds (not routed through controller config)
        if ch in ("f", "F"):
            self.engine._reset_runtime_context()
            self.status_message = "Full state reset"
            return
        if ch in ("v", "V"):
            self._show_gate_debug = not self._show_gate_debug
            self.status_message = f"Menu {'ON' if self._show_gate_debug else 'OFF'}"
            return
        if ch in ("c", "C"):
            self._clear_data()
            return

        command = self._config.controller.bindings.get(ch)
        if command:
            self._dispatch_command(command)

    @staticmethod
    def _format_gain_bar(gain: float, bar_width: int = 20) -> str:
        filled = min(int(gain * bar_width / 10), bar_width)
        return f"[{'▮' * filled}{' ' * (bar_width - filled)}] {gain:.1f}x"

    def _dispatch_command(self, command: str) -> None:
        if command in ("quit", "esc"):
            self.running = False
        elif command == "toggle_passthrough":
            self._toggle_mode()
        elif command == "enroll":
            self._start_enrollment()
        elif command == "name":
            self.naming = True
            self._name_input_buffer = ""
            self.status_message = "Type a name and press Enter to confirm"
        elif command == "next_speaker":
            self._nav_speaker(+1)
        elif command == "prev_speaker":
            self._nav_speaker(-1)
        elif command == "select":
            self._select_focused_speaker()
        elif command == "decrease_gain":
            new_gain = max(0.0, self.engine.output_gain - 0.5)
            self.engine.set_output_gain(new_gain)
            self.status_message = f"Gain decreased ({new_gain:.1f}x)"
        elif command == "increase_gain":
            new_gain = min(10.0, self.engine.output_gain + 0.5)
            self.engine.set_output_gain(new_gain)
            self.status_message = f"Gain increased ({new_gain:.1f}x)"

    def _confirm_rename(self) -> None:
        word = self._name_input_buffer.strip()
        self.naming = False
        self._name_input_buffer = ""
        if not word:
            self.status_message = "Name unchanged (empty input)"
            return
        if not self._speaker_list:
            self.status_message = "No speaker to rename"
            return
        focused = self._focused_idx % len(self._speaker_list)
        spk = self._speaker_list[focused]
        old_name = spk.name
        speakers, recordings, extractions = self.store.load()
        for s in speakers:
            if s.id == spk.id:
                s.name = word
                break
        self.store.save(speakers, recordings, extractions)
        self._speaker_list = self._load_speakers()
        if self.current_speaker == old_name:
            self.current_speaker = word
        self.status_message = f"Renamed '{old_name}' to '{word}'"

    def _clear_data(self) -> None:
        """Wipe all enrolled speakers and reset data (equivalent to make wipe-data)."""
        import glob as _glob
        for pattern in ("*.wav", "*.npy", "*.meta.json"):
            for f in _glob.glob(str(ENROLLMENTS_DIR / pattern)):
                os.remove(f)
        DATA_FILE.write_text('{"speakers": [], "recordings": [], "extractions": []}')
        self._speaker_list = []
        self._focused_idx = 0
        self.current_speaker = None
        self.engine.embedding = None
        self.engine.set_passthrough(True)
        self.engine._reset_runtime_context()
        self.status_message = "Wiped all enrolled speakers and reset data"

    def _nav_speaker(self, delta: int) -> None:
        if not self._speaker_list:
            self.status_message = "No speakers enrolled"
            return
        self._focused_idx = (self._focused_idx + delta) % len(self._speaker_list)
        self.status_message = f"Focused: {self._speaker_list[self._focused_idx].name}"

    def _select_focused_speaker(self) -> None:
        if not self._speaker_list:
            self.status_message = "No speakers enrolled"
            return
        self._focused_idx = self._focused_idx % len(self._speaker_list)
        spk = self._speaker_list[self._focused_idx]
        emb = np.load(spk.embedding_path)
        self.engine.set_embedding(emb)
        self.engine.set_passthrough(False)
        self.current_speaker = spk.name
        self.status_message = f"Selected: {spk.name}. Isolation active."

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
            tty.setcbreak(fd)
            cbreak_attrs = termios.tcgetattr(fd)
            while self.running:
                # Re-apply cbreak if something (e.g. a spawned child process)
                # reset the terminal to cooked mode.
                if termios.tcgetattr(fd) != cbreak_attrs:
                    tty.setcbreak(fd)
                    cbreak_attrs = termios.tcgetattr(fd)
                rlist, _, _ = select.select([fd], [], [], 0.1)
                if rlist:
                    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                    if ch:
                        try:
                            self._handle_key(ch)
                        except Exception:
                            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def run(self) -> None:
        self._console = Console()
        try:
            self.engine.start()
            self.running = True

            key_thread = threading.Thread(target=self._keypress_thread, daemon=True)
            key_thread.start()

            with Live(self._render(), refresh_per_second=5, console=self._console, screen=True) as live:
                while self.running:
                    time.sleep(0.2)
                    ar_count = self.engine._ar_reset_count
                    if ar_count > self._last_ar_count:
                        self.status_message = f"[auto-reset] State reset #{ar_count} triggered"
                        self._last_ar_count = ar_count
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
        default=None,
        help="Speaker embedding model (default: from enrollment.use_beamformer in config.yaml)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "src" / "realtime" / "config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    embedding_model_id = args.embedding_model or (
        "beamformer_resemblyzer" if config.enrollment.use_beamformer else "resemblyzer"
    )

    if config.test.enabled:
        # File-based test mode: no audio devices needed
        if config.test.input_file is None:
            parser.error("test.input_file is required when test mode is enabled (set in config.yaml)")
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = config.test.output_dir or Path(REPO_ROOT / "reports/eval/output_audio")
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
            from src.realtime.plots import generate_plots as _generate_plots
            plot_out = Path(config.test.report_dir or "reports/eval") / "plots"
            _generate_plots(plot_data, stats, plot_out, ts=ts)
        return

    # Force passthrough + no embedding for demo startup
    config.debug.passthrough = True
    config.model.embedding = None

    perf_logger: PerformanceLogger | None = None
    if config.logging.enabled:
        perf_logger = PerformanceLogger(config.logging.log_dir)
        perf_logger.start()

    app = DemoApp(config, embedding_model_id, logger=perf_logger)
    app.run()


if __name__ == "__main__":
    main()
