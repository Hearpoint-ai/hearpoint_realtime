import json
import queue
import sys
import threading
import time
import types
import unittest

import numpy as np

from src.realtime.vosk_worker import vosk_worker


class VoskWorkerTests(unittest.TestCase):
    def test_reset_grace_period_prevents_immediate_retrigger(self):
        fake_vosk = types.ModuleType("vosk")
        recognizers = []

        class FakeModel:
            def __init__(self, path):
                self.path = path

        class FakeRecognizer:
            def __init__(self, model, sample_rate):
                self.sample_rate = sample_rate
                self.reset_count = 0
                recognizers.append(self)

            def Reset(self):
                self.reset_count += 1

            def AcceptWaveform(self, data):
                return False

            def Result(self):
                return json.dumps({"text": "matthew"})

            def PartialResult(self):
                return json.dumps({"partial": "matthew"})

        fake_vosk.Model = FakeModel
        fake_vosk.KaldiRecognizer = FakeRecognizer

        original_vosk = sys.modules.get("vosk")
        sys.modules["vosk"] = fake_vosk
        try:
            audio_queue = queue.Queue()
            control_queue = queue.Queue()
            detection_event = threading.Event()
            stop_event = threading.Event()

            worker = threading.Thread(
                target=vosk_worker,
                args=(audio_queue, control_queue, detection_event, stop_event, ".", 16000, "matthew"),
                daemon=True,
            )
            worker.start()

            control_queue.put({"type": "reset", "grace_period_s": 0.2})
            audio_queue.put(np.ones((64, 1), dtype=np.float32))
            time.sleep(0.05)
            self.assertFalse(detection_event.is_set())

            time.sleep(0.25)
            audio_queue.put(np.ones((64, 1), dtype=np.float32))

            deadline = time.time() + 1.0
            while time.time() < deadline and not detection_event.is_set():
                time.sleep(0.01)
            self.assertTrue(detection_event.is_set())
            self.assertGreaterEqual(recognizers[0].reset_count, 1)

            stop_event.set()
            worker.join(timeout=1.0)
            self.assertFalse(worker.is_alive())
        finally:
            if original_vosk is None:
                sys.modules.pop("vosk", None)
            else:
                sys.modules["vosk"] = original_vosk


if __name__ == "__main__":
    unittest.main()
