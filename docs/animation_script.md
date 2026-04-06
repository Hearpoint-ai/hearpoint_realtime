# HearPoint: Real-Time Speaker Isolation — Animation Script
# For use with tma.live

---

## SCENE 1 — THE PROBLEM (0:00–0:30)

**[Visual: Crowded room. Multiple overlapping sound waves in different colors colliding into one messy waveform.]**

Imagine you're on a call in a noisy room. Your microphone picks up everything — your voice, background music, other conversations — all mixed into one continuous stream of audio samples.

**[Visual: Raw waveform on screen, indistinguishable voices labeled as "you", "background", "noise".]**

A microphone captures sound as a series of numbers — 16,000 of them every second. Each number is just an air pressure value. There's no label saying which sound came from where.

The challenge: how do you pull one specific voice out of that mix, in real time, with less than 20 milliseconds of delay?

---

## SCENE 2 — CHUNKING THE STREAM (0:30–1:00)

**[Visual: Long audio stream being sliced into equal 8ms rectangles.]**

The system processes audio in small chunks — 128 samples at a time, or about 8 milliseconds of audio.

**[Visual: Chunk counter. Each chunk slides into a processing box.]**

Every 8ms, a new chunk arrives and must be fully processed before the next one appears. Miss the deadline, and audio dropouts occur.

**[Visual: Timer counting down 8ms, chunk being processed, output emerging.]**

This real-time constraint means every step in the pipeline has to be fast. The model runs on a GPU, completing each chunk in 2–4ms — well inside the 8ms budget.

---

## SCENE 3 — FROM TIME TO FREQUENCY: THE STFT (1:00–1:50)

**[Visual: A chunk of 128 samples shown as a waveform. Then a sliding window of 280 samples sweeps across it.]**

Before the neural network can analyze the audio, we transform it from the time domain into the frequency domain using the Short-Time Fourier Transform, or STFT.

**[Visual: Hann window applied — the edges of the 280-sample window taper to zero smoothly.]**

We apply a Hann window — a smooth bell-shaped curve — to 280 samples at a time. This prevents sharp edges from creating artificial high-frequency artifacts.

**[Visual: FFT applied. Time-domain waveform morphs into a vertical column of frequency magnitudes. Then multiple columns appear, forming a 2D spectrogram.]**

The Fast Fourier Transform decomposes each windowed segment into its frequency components — how much energy is present at each pitch, from 0 Hz up to 8,000 Hz.

**[Visual: The spectrogram fills in. Low frequencies at bottom, high at top. Brighter = more energy.]**

The result is a spectrogram: a 2D grid where the horizontal axis is time, the vertical axis is frequency, and brightness represents energy. Human speech concentrates in specific frequency bands. So does background noise. They overlap — but they have different patterns.

**[Visual: STFT formula appears: X(f, t) = Σ x(n) · w(n−t) · e^(−i2πfn/N)]**

Mathematically, each point in the spectrogram is a complex number — a magnitude telling us how loud that frequency is, and a phase telling us the timing of that frequency's wave.

---

## SCENE 4 — THE NEURAL NETWORK: TFGRIDNET (1:50–3:10)

**[Visual: The spectrogram grid. An "AI" block descends onto it.]**

The spectrogram is fed into TFGridNet — a neural network purpose-built for this task.

### What "TF" means

**[Visual: Two axes of the spectrogram highlighted — time (horizontal) and frequency (vertical).]**

TF stands for Time-Frequency. The network operates directly on the spectrogram grid, processing both dimensions simultaneously.

### What "GridNet" means

**[Visual: Two arrows sweep across the grid — one horizontal (time), one vertical (frequency).]**

GridNet describes two LSTMs that traverse the grid in perpendicular directions.

**[Visual: "Intra LSTM" sweeps along each row — processing how one frequency evolves over time.]**

The Intra LSTM processes time: for each frequency band, it looks at how that band's energy changes across frames. This captures the rhythm and dynamics of speech.

**[Visual: "Inter LSTM" sweeps along each column — processing all frequencies at one moment in time.]**

The Inter LSTM processes frequency: at each moment in time, it looks across all frequency bands simultaneously. This captures the harmonic structure that defines a specific voice.

**[Visual: The two LSTMs repeating in 3 stacked blocks.]**

This pairing of time-sweep and frequency-sweep is repeated 3 times, each layer refining the network's understanding of what belongs to the target speaker versus what doesn't.

### Speaker Conditioning

**[Visual: A waveform of a short enrollment clip → a compact embedding vector → being injected into the spectrogram.]**

Here's the key ingredient: the network is told *whose* voice to keep.

Before real-time processing begins, a short enrollment recording from the target speaker is encoded into a 256-dimensional embedding — a compact numerical fingerprint of that person's voice.

**[Visual: Embedding vector multiplied element-wise into the spectrogram features — FiLM layer.]**

This embedding is projected and multiplied into the network's internal representations. Mathematically, this is called Feature-wise Linear Modulation (FiLM). It biases every layer to recognize the target speaker's patterns and suppress everything else.

### The Mask

**[Visual: TFGridNet outputs a grid of values between 0 and 1 — the mask — overlaid on the input spectrogram.]**

The network's final output is a mask — one value per frequency-time bin, ranging from 0 to 1.

**[Visual: Mask multiplied element-wise into the spectrogram. Bright regions from the target speaker are preserved; noise regions dim out.]**

This mask is multiplied into the input spectrogram:

**[Visual: Equation: Output(f,t) = Mask(f,t) × Input(f,t)]**

Regions where the target speaker is dominant get a mask value near 1 — kept. Regions dominated by noise or other voices get a mask near 0 — suppressed.

Crucially, only the *magnitude* is masked. The *phase* — the timing information — is carried over from the input unchanged. This prevents artifacts and preserves naturalness.

---

## SCENE 5 — BACK TO AUDIO: THE ISTFT (3:10–3:40)

**[Visual: Filtered spectrogram glowing with only the target speaker's frequencies highlighted.]**

Once the mask is applied, we have a clean spectrogram. Now we need to convert it back into audio.

**[Visual: Inverse FFT arrows reversing the spectrogram back into waveform segments.]**

The Inverse STFT (ISTFT) converts each spectrogram column back into a short audio segment.

**[Visual: Overlapping audio segments. The overlap-add operation shown: segments aligned, overlapping regions added together.]**

Because the analysis windows overlapped, the reconstructed segments also overlap. They're summed together using overlap-add. The Hann window is carefully designed so that overlapping windows sum to exactly 1 — meaning no samples are lost or double-counted.

**[Visual: Clean output waveform emerging. Only the target voice's pattern visible.]**

The result is a clean output waveform, one chunk at a time, ready to be played through the speaker.

---

## SCENE 6 — STATEFULNESS ACROSS CHUNKS (3:40–4:10)

**[Visual: Sequence of chunks. Between each chunk, a small "memory" box passes state forward.]**

There's one more important detail: the LSTMs have memory.

**[Visual: LSTM cell diagram. Hidden state h_t and cell state c_t passed from one chunk to the next.]**

An LSTM maintains a hidden state (h) and a cell state (c) — vectors that summarize what it has heard so far. At the end of each 8ms chunk, these states are saved and reinjected at the start of the next chunk.

**[Visual: Continuous conversation with state flowing through chunks like a thread.]**

This means the network isn't treating each chunk in isolation. It accumulates context across seconds of audio — learning the current acoustic environment, adapting to the speaker's speech patterns.

---

## SCENE 7 — LOOKAHEAD AND LATENCY (4:10–4:35)

**[Visual: Timeline. Current chunk highlighted. 4ms of future audio peeking forward.]**

The system uses a 4ms lookahead — it actually reads slightly ahead before processing the current chunk. This small glimpse into the future improves accuracy at chunk boundaries, at the cost of 4ms of added latency.

**[Visual: Latency breakdown bar: 8ms capture + 4ms lookahead + 2ms model = ~14ms total.]**

Total end-to-end latency is roughly 14–20 milliseconds. That's fast enough to feel synchronous — comparable to the delay in a typical phone call.

---

## SCENE 8 — FULL PIPELINE RECAP (4:35–5:00)

**[Visual: Full pipeline animated end-to-end in one sweep.]**

To recap: raw audio arrives in 8ms chunks. Each chunk is transformed into a time-frequency spectrogram via STFT. TFGridNet applies a speaker-conditioned mask to suppress everything that isn't the target voice. The ISTFT reconstructs clean audio. Stateful LSTMs carry context across chunks. And the whole loop completes in under 8ms — in real time.

**[Visual: Clean voice waveform plays back. Background noise is gone.]**

The result: one voice, isolated, in real time.

---

*Script end — approximately 5 minutes at natural narration pace.*
