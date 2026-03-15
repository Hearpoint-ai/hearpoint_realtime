"""Zero-latency performance logger using a background-thread queue pattern.

The hot processing thread calls log() which does a non-blocking put_nowait()
into a SimpleQueue. A daemon thread drains the queue and writes JSONL to disk.
"""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path


_SENTINEL = object()


class PerformanceLogger:
    """Writes performance snapshots and session summaries to a JSONL file.

    The log() method is O(1) and non-blocking — safe to call from the audio
    processing thread. All file I/O happens in a background daemon thread.
    """

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._log_path: Path | None = None

    def start(self) -> None:
        """Open the log file and start the background writer thread."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_path = self._log_dir / f"realtime_{ts}.jsonl"
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="perf-logger")
        self._thread.start()

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    def log(self, record: dict) -> None:
        """Non-blocking enqueue. Drops the record silently if the thread is gone."""
        try:
            self._queue.put_nowait(record)
        except Exception:
            pass

    def stop(self) -> None:
        """Signal the writer to flush remaining records and exit, then join."""
        self._queue.put_nowait(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _writer_loop(self) -> None:
        if self._log_path is None:
            return
        with self._log_path.open("w", encoding="utf-8") as fh:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                try:
                    if item.get("type") == "summary":
                        fh.write(json.dumps(item, default=str, indent=2) + "\n")
                    else:
                        fh.write(json.dumps(item, default=str) + "\n")
                    fh.flush()
                except Exception:
                    pass
