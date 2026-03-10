import time
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---------------------------
# Configuration
# ---------------------------

SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # Try swapping with a larger model

# How long to collect audio (seconds)
TEST_DURATION = 10
# Optional grammar list for lighter decoding

# ---------------------------
# Audio setup
# ---------------------------

audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio callback status:", status)
    # Convert to 16‑bit PCM
    audio_q.put((indata.copy() * 32767).astype("int16").tobytes())

# ---------------------------
# Main benchmarking loop
# ---------------------------

def main():
    print("Loading model:", MODEL_PATH)
    model_load_t0 = time.perf_counter()
    model = Model(MODEL_PATH)
    model_load_t1 = time.perf_counter()
    print(f"Model load time: {model_load_t1 - model_load_t0:.2f} s\n")

    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

    latencies = []
    print(f"Recording for {TEST_DURATION} s — speak normally.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,   # 0.5 s blocks; adjust to test responsiveness
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        start = time.perf_counter()
        while (time.perf_counter() - start) < TEST_DURATION:
            try:
                data = audio_q.get(timeout=2)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            ok = recognizer.AcceptWaveform(data)
            t1 = time.perf_counter()

            latencies.append(t1 - t0)

            if ok:
                res = json.loads(recognizer.Result())
                txt = res.get("text", "")
                print(f"[Final] {txt}")
            else:
                res = json.loads(recognizer.PartialResult())
                part = res.get("partial", "")
                if part:
                    print(f"\r[Partial] {part}", end="")

    # ---------------------------
    # Report results
    # ---------------------------
    if latencies:
        avg_ms = sum(latencies) / len(latencies) * 1000
        p95_ms = sorted(latencies)[int(0.95 * len(latencies))] * 1000
        print("\n--- Benchmark results ---")
        print(f"Model: {MODEL_PATH}")
        print(f"Blocks processed: {len(latencies)}")
        print(f"Average AcceptWaveform latency: {avg_ms:.2f} ms")
        print(f"95th‑percentile latency: {p95_ms:.2f} ms")
    else:
        print("No latencies recorded (no audio blocks).")

    print("\nDone.\n")

if __name__ == "__main__":
    main()