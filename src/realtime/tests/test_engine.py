import queue
import threading
import unittest

import numpy as np
import torch

from src.realtime.engine import ControlCommand, RealtimeInference


class BlockingModel:
    def __init__(self):
        self.stft_pad_size = 2
        self.calls = []
        self._state_counter = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def init_buffers(self, batch_size, device):
        self._state_counter += 1
        return {"id": self._state_counter}

    def predict(self, input_buffer, embedding, state, pad=True, lookahead_audio=None):
        self.entered.set()
        self.release.wait(timeout=1.0)
        self.calls.append(
            {
                "embedding": embedding.detach().cpu().numpy().copy(),
                "state_id": state["id"],
            }
        )
        output = torch.ones((1, 2, input_buffer.shape[-1]), dtype=torch.float32, device=input_buffer.device)
        return output, state


class EngineControlTests(unittest.TestCase):
    def _make_engine(self, model=None, name_detection_enabled=False):
        engine = RealtimeInference.__new__(RealtimeInference)
        engine.sample_rate = 16000
        engine.chunk_size = 4
        engine.input_channels = 1
        engine.output_channels = 2
        engine.buffer_size_chunks = 2
        engine.output_gain = 1.0
        engine.debug = False
        engine.save_debug_dir = None
        engine.passthrough_mode = False
        engine.input_queue = queue.Queue(maxsize=64)
        engine.output_queue = queue.Queue(maxsize=64)
        engine._control_queue = queue.Queue()
        engine.input_accumulator = np.zeros((0, engine.input_channels), dtype=np.float32)
        engine.input_level_lock = threading.Lock()
        engine.recent_input_level = 0.0
        engine.processing_times = []
        engine.inference_times = []
        engine.prep_times = []
        engine.post_times = []
        engine.chunks_processed = 0
        engine.drops_input = 0
        engine.drops_output = 0
        engine.underruns = 0
        engine.warmup_chunks = 0
        engine.nan_count = 0
        engine._sq_sum_in = 0.0
        engine._sq_sum_out = 0.0
        engine._total_post_warmup_samples = 0
        engine._clipped_post_warmup = 0
        engine.device = torch.device("cpu")
        engine._input_buffer = torch.zeros((1, 2, engine.chunk_size), dtype=torch.float32)
        engine._lookahead_buffer = torch.zeros((1, 2, 2), dtype=torch.float32)
        engine.running = True
        engine._process_thread = object()
        engine._play_transparency_sound_async = lambda: None
        engine.name_detection_enabled = name_detection_enabled
        engine.name_detection_armed = name_detection_enabled
        engine._name_detection_grace_period_s = 1.0
        engine._name_detection_event = threading.Event() if name_detection_enabled else None
        engine._name_detection_queue = queue.Queue() if name_detection_enabled else None
        engine._name_detection_control_queue = queue.Queue() if name_detection_enabled else None
        engine.model = model or BlockingModel()
        engine.stft_pad_size = engine.model.stft_pad_size
        engine.embedding = engine._make_embedding_tensor(np.array([1.0, 0.0], dtype=np.float32))
        engine.state = engine.model.init_buffers(batch_size=1, device=engine.device)
        return engine

    def test_embedding_switch_is_applied_between_chunks(self):
        model = BlockingModel()
        engine = self._make_engine(model=model)
        old_state_id = engine.state["id"]
        old_embedding = engine.embedding[:, 0].detach().cpu().numpy().copy()
        new_embedding = np.array([0.0, 1.0], dtype=np.float32)

        worker = threading.Thread(
            target=engine._process_chunk,
            args=(np.ones((4, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)),
        )
        worker.start()
        self.assertTrue(model.entered.wait(timeout=1.0))

        engine.set_embedding(new_embedding)
        self.assertTrue(np.allclose(engine.embedding[:, 0].detach().cpu().numpy(), old_embedding))

        model.release.set()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["state_id"], old_state_id)
        self.assertTrue(np.allclose(model.calls[0]["embedding"], old_embedding))

        engine._apply_pending_control_commands()
        engine._process_chunk(np.ones((4, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32))

        self.assertEqual(len(model.calls), 2)
        self.assertNotEqual(engine.state["id"], old_state_id)
        self.assertTrue(np.allclose(model.calls[1]["embedding"], new_embedding.reshape(1, -1)))
        self.assertEqual(model.calls[1]["state_id"], engine.state["id"])

    def test_switch_flushes_queued_audio_and_prefills_silence(self):
        engine = self._make_engine()
        engine.input_queue.put(np.ones((8, 1), dtype=np.float32))
        engine.output_queue.put(np.ones((4, 2), dtype=np.float32))
        engine.input_accumulator = np.ones((6, 1), dtype=np.float32)

        engine._apply_control_command(
            ControlCommand(kind="set_embedding", payload=np.array([0.0, 1.0], dtype=np.float32))
        )

        self.assertEqual(len(engine.input_accumulator), 0)
        self.assertTrue(engine.input_queue.empty())
        self.assertEqual(engine.output_queue.qsize(), engine.buffer_size_chunks * 2)
        while not engine.output_queue.empty():
            self.assertTrue(np.allclose(engine.output_queue.get_nowait(), 0.0))

    def test_manual_rearm_after_name_detection_passthrough(self):
        engine = self._make_engine(name_detection_enabled=True)
        engine._name_detection_event.set()

        engine._handle_name_detection_trigger()
        self.assertTrue(engine.passthrough_mode)
        self.assertFalse(engine.name_detection_armed)
        self.assertFalse(engine._name_detection_event.is_set())

        engine._name_detection_event.set()
        engine._handle_name_detection_trigger()
        self.assertTrue(engine.passthrough_mode)
        self.assertFalse(engine._name_detection_event.is_set())

        engine._apply_control_command(ControlCommand(kind="set_passthrough", payload=False, manual=True))
        self.assertFalse(engine.passthrough_mode)
        self.assertTrue(engine.name_detection_armed)
        self.assertFalse(engine._name_detection_event.is_set())
        reset_command = engine._name_detection_control_queue.get_nowait()
        self.assertEqual(reset_command["type"], "reset")

    def test_processing_thread_rechecks_accumulator_after_control_reset(self):
        engine = self._make_engine()
        engine.running = True
        engine.input_queue.put(np.ones((8, 1), dtype=np.float32))
        engine.state = engine.model.init_buffers(batch_size=1, device=engine.device)

        original_apply_pending = engine._apply_pending_control_commands
        call_count = {"count": 0}

        def apply_pending_and_stop():
            call_count["count"] += 1
            if call_count["count"] == 2:
                engine.input_accumulator = np.zeros((0, engine.input_channels), dtype=np.float32)
                engine.running = False
                return
            return original_apply_pending()

        engine._apply_pending_control_commands = apply_pending_and_stop

        worker = threading.Thread(target=engine._processing_thread, daemon=True)
        worker.start()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(engine.model.calls), 0)


if __name__ == "__main__":
    unittest.main()
