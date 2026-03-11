import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---------------------------
# Configuration
# ---------------------------

SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # Change to your model folder
GRAMMAR = ["matthew"]  # Restricted grammar: the recognizer only knows this word

# ---------------------------
# Audio capture setup
# ---------------------------

audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback from sounddevice.InputStream."""
    if status:
        print(f"Audio stream status: {status}")
    # Convert float32 [-1,1] to int16 bytes
    audio_q.put((indata.copy() * 32767).astype("int16").tobytes())

# ---------------------------
# Main
# ---------------------------

def main():
    print("Loading model...")
    model = Model(MODEL_PATH)

    print("Initializing recognizer (grammar = {})".format(GRAMMAR))
    recognizer = KaldiRecognizer(model, SAMPLE_RATE, json.dumps(GRAMMAR))

    print("\nListening... Say 'Hady' to trigger.\n"
          "Press Ctrl+C to stop at any time.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype="float32",
            channels=1,
            callback=audio_callback,
        ):
            while True:
                data = audio_q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    transcript = result.get("text", "").lower().strip()
                    if transcript:
                        print(f"[Final]: {transcript}")  # Debug: finalized chunk
                        if "matthew" in transcript.split():
                            print("NAME DETECTED")
                    else:
                        print("[Final]: (silence or unknown)")
                else:
                    partial = json.loads(recognizer.PartialResult())
                    part_text = partial.get("partial", "").lower().strip()
                    if part_text:
                        # Debugging: show what it's currently hearing
                        print(f"\r[Partial]: {part_text}", end="", flush=True)

    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    main()