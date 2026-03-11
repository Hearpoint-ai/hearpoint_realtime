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
import select
import sys
import termios
import threading
import time
import tty
import uuid
from datetime import datetime, timezone
from pathlib import Path

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
from src.realtime.realtime_inference import Config, RealtimeInference, _ensure_stereo

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


ENROLLMENT_DURATION = 5.0  # seconds


class DemoApp:
    def __init__(self, config: Config, embedding_model_id: str):
        self.engine = RealtimeInference(config)
        self.store = MediaJsonStore(media_root=MEDIA_DIR, data_file=DATA_FILE)
        self.embedding_model_id = embedding_model_id
        self.embedding_model = None  # lazy-loaded on first enrollment

        self.current_speaker: str | None = None
        self.mode = "passthrough"
        self.enrolling = False
        self.enroll_start_time: float | None = None
        self.selecting = False
        self._speaker_list: list[Speaker] = []
        self.status_message = "Ready"
        self.running = False

    def _load_speakers(self) -> list[Speaker]:
        speakers, _, _ = self.store.load()
        return speakers

    def _render(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("key", style="bold", width=12)
        table.add_column("value")

        # Mode
        if self.mode == "passthrough":
            mode_text = Text("PASSTHROUGH", style="bold green")
        else:
            mode_text = Text("ISOLATION", style="bold red")
        table.add_row("Mode", mode_text)

        # Speaker
        speaker_text = self.current_speaker or "(none enrolled)"
        table.add_row("Speaker", speaker_text)

        # Level meter
        with self.engine.input_level_lock:
            level = self.engine.recent_input_level
        level_db = 20 * np.log10(level + 1e-10)
        bars = int(max(0, min(30, (level_db + 60) / 2)))
        meter = f"[{'#' * bars}{'.' * (30 - bars)}] {level_db:5.1f} dB"
        table.add_row("Level", meter)

        table.add_row("", "")

        # Stats
        if self.engine.processing_times:
            recent = self.engine.processing_times[-100:]
            recent_ms = np.array(recent) * 1000
            chunk_ms = self.engine.chunk_size / self.engine.sample_rate * 1000
            avg = np.mean(recent_ms)
            rtf = avg / chunk_ms
            p95 = np.percentile(recent_ms, 95)

            table.add_row("RTF", f"{rtf:.3f}")
            table.add_row("Latency", f"{avg:.1f}ms avg")
            table.add_row("Chunks", f"{self.engine.chunks_processed:,}")
            table.add_row("p95", f"{p95:.1f}ms")
            drops = f"{self.engine.drops_input}/{self.engine.drops_output}"
            table.add_row("Drops", drops)
            table.add_row("Underruns", str(self.engine.underruns))
        else:
            table.add_row("RTF", "—")
            table.add_row("Latency", "—")
            table.add_row("Chunks", "0")
            table.add_row("p95", "—")
            table.add_row("Drops", "0/0")
            table.add_row("Underruns", "0")

        table.add_row("", "")

        # Controls
        controls = Text("[T] Toggle   [E] Enroll   [S] Select   [Q] Quit", style="dim")
        table.add_row("", controls)

        table.add_row("", "")

        # Status / enrollment countdown / selection list
        if self.enrolling and self.enroll_start_time is not None:
            elapsed = time.time() - self.enroll_start_time
            remaining = max(0, ENROLLMENT_DURATION - elapsed)
            status_text = Text(f"Enrolling... speak now ({remaining:.0f}s remaining)", style="bold yellow")
        elif self.selecting and self._speaker_list:
            lines = ["Select a speaker:"]
            for i, spk in enumerate(self._speaker_list[:9], 1):
                lines.append(f"  [{i}] {spk.name}")
            lines.append("  [Esc] Cancel")
            status_text = Text("\n".join(lines), style="cyan")
        else:
            status_text = Text(f"Status: {self.status_message}")

        table.add_row("", status_text)

        return Panel(table, title="HearPoint Real-Time Demo", border_style="blue", width=55)

    def _handle_key(self, ch: str) -> None:
        if self.enrolling:
            return  # ignore keys during enrollment

        if self.selecting:
            if ch == "\x1b":  # Escape
                self.selecting = False
                self.status_message = "Selection cancelled"
            elif ch.isdigit() and 1 <= int(ch) <= len(self._speaker_list):
                idx = int(ch) - 1
                spk = self._speaker_list[idx]
                emb = np.load(spk.embedding_path)
                self.engine.set_embedding(emb)
                self.engine.set_passthrough(False)
                self.current_speaker = spk.name
                self.mode = "isolation"
                self.selecting = False
                self.status_message = f"Selected: {spk.name}. Isolation active."
            return

        if ch in ("q", "Q", "\x03"):  # q or Ctrl+C
            self.running = False
        elif ch in ("t", "T"):
            self._toggle_mode()
        elif ch in ("e", "E"):
            self._start_enrollment()
        elif ch in ("s", "S"):
            self._start_selection()

    def _toggle_mode(self) -> None:
        if self.engine.embedding is None:
            self.status_message = "Enroll a speaker first [E]"
            return
        is_passthrough = self.engine.passthrough_mode
        self.engine.set_passthrough(not is_passthrough)
        self.mode = "passthrough" if not is_passthrough else "isolation"
        self.status_message = f"Mode: {self.mode}"

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

                # Lazy-load embedding model
                if self.embedding_model is None:
                    self.status_message = "Loading embedding model..."
                    self.embedding_model = create_embedding_model(self.embedding_model_id)
                    self.status_message = "Computing embedding..."

                # [N, 2] → [2, N] channels-first (standard contract)
                audio_2xN = audio.T.astype(np.float32)
                embedding = self.embedding_model.compute_embedding_from_array(
                    audio_2xN, self.engine.sample_rate
                )

                # Set embedding on engine and switch to isolation
                self.engine.set_embedding(embedding)
                self.engine.set_passthrough(False)
                self.mode = "isolation"

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
            except Exception as e:
                self.status_message = f"Enrollment error: {e}"

        threading.Thread(target=_enrollment_worker, daemon=True).start()

    def _start_selection(self) -> None:
        self._speaker_list = self._load_speakers()
        if not self._speaker_list:
            self.status_message = "No enrolled speakers. Press [E] to enroll."
            return
        self.selecting = True

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

            with Live(self._render(), refresh_per_second=5, console=console) as live:
                while self.running:
                    time.sleep(0.2)
                    live.update(self._render())
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.engine.stop()


def main():
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

    app = DemoApp(config, args.embedding_model)
    app.run()


if __name__ == "__main__":
    main()
