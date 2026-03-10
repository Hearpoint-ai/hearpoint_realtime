import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---------------------------
# Configuration
# ---------------------------

SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # Adjust to your installed model path
target = "matthew"  # Word to detect in the transcript

# ---------------------------
# Audio stream setup
# ---------------------------

audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback from sounddevice.InputStream."""
    if status:
        print(status)
    audio_q.put((indata.copy() * 32767).astype("int16").tobytes())

# ---------------------------
# Main
# ---------------------------

def main():
    print("Loading model...")
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    transparency = False

    print("Live transcription started. Speak into your microphone.")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=8000, dtype="float32",
            channels=1, callback=audio_callback
        ):
            while True:
                data = audio_q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    transcript = result.get("text", "").lower().strip()
                    print(f">> {transcript}")
                    if target in transcript.split() and not transparency:
                        transparency = True
                        print(f"***NAME DETECTED***")
                else:
                    # Partial results while speaking
                    partial = json.loads(recognizer.PartialResult())
                    if text := partial.get("partial"):
                        # Overwrite current line with partial transcript
                        print(f"\r{text}", end="", flush=True)
                        if target in text.split() and not transparency:
                            transparency = True
                            print(f"***NAME DETECTED***")

    except KeyboardInterrupt:
        print("\nExiting transcription.")

if __name__ == "__main__":
    main()