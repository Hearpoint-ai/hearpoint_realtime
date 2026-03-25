# !/usr/local/bin/python3.8

import argparse
import os

import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf

base_path = os.getcwd()
output_path = os.path.join(base_path, "subtracted_signal.wav")


def resample(input_signal, old_sample_rate, new_sample_rate):
    resampled_signal = signal.resample_poly(input_signal, new_sample_rate, old_sample_rate)
    return resampled_signal.astype(input_signal.dtype)


def _channel_view(audio, dimension):
    if audio.ndim == 1:
        return audio
    return audio[:, dimension]


def _apply_spectral_subtraction(magnitude, phase_complex, noise_mean):
    output_mag = np.clip(magnitude - noise_mean[:, np.newaxis], a_min=0.0, a_max=None)
    return output_mag * phase_complex


def stft(audio, dimension, n_fft=2048, hop_length=None, win_length=None, center=True):
    signal_ch = _channel_view(audio, dimension)
    return librosa.stft(
        signal_ch,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
    )


def spectral_subtraction(
    noise_profile_n,
    input_signal_y,
    dimension,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    center=True,
):
    N = stft(
        noise_profile_n,
        dimension,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
    )
    mN = np.abs(N)

    Y = stft(
        input_signal_y,
        dimension,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
    )
    mY = np.abs(Y)
    pY = np.angle(Y)
    poY = np.exp(1.0j * pY)

    noise_mean = np.mean(mN, axis=1, dtype="float64")
    X = _apply_spectral_subtraction(mY, poY, noise_mean)

    return librosa.istft(X, hop_length=hop_length, win_length=win_length, center=center)


def _to_2d(audio):
    if audio.ndim == 1:
        return audio[:, np.newaxis]
    if audio.ndim != 2:
        raise ValueError("Expected mono or stereo audio array.")
    return audio


class StreamingSpectralSubtractor:
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        hop_length=128,
        win_length=512,
        channels=1,
        noise_mean=None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.channels = channels
        self.window = np.hanning(self.win_length).astype(np.float64)
        if self.win_length != self.n_fft:
            pad = self.n_fft - self.win_length
            self.window = np.pad(self.window, (0, max(pad, 0)))
        self._eps = 1e-8

        self._analysis_buffer = np.zeros((0, channels), dtype=np.float64)
        self._synth_num = np.zeros((self.n_fft, channels), dtype=np.float64)
        self._synth_den = np.zeros((self.n_fft, channels), dtype=np.float64)
        self._pending_output = np.zeros((0, channels), dtype=np.float64)
        self._noise_frames = 0

        bins = self.n_fft // 2 + 1
        if noise_mean is None:
            self.noise_mean = np.zeros((bins, channels), dtype=np.float64)
        else:
            nm = np.asarray(noise_mean, dtype=np.float64)
            if nm.ndim == 1:
                nm = np.repeat(nm[:, np.newaxis], channels, axis=1)
            if nm.shape != (bins, channels):
                raise ValueError("noise_mean has incompatible shape.")
            self.noise_mean = nm
            self._noise_frames = 1

    def _normalize_chunk(self, chunk):
        arr = np.asarray(chunk, dtype=np.float64)
        arr2d = _to_2d(arr)
        if arr2d.shape[1] != self.channels:
            if arr2d.shape[1] == 1 and self.channels == 2:
                arr2d = np.repeat(arr2d, 2, axis=1)
            else:
                raise ValueError(f"Expected {self.channels} channels, got {arr2d.shape[1]}.")
        return arr2d

    def _append_pending(self, block):
        if block.size == 0:
            return
        self._pending_output = np.vstack((self._pending_output, block))

    def _pop_pending(self, samples):
        if self._pending_output.shape[0] < samples:
            return np.zeros((0, self.channels), dtype=np.float64)
        out = self._pending_output[:samples]
        self._pending_output = self._pending_output[samples:]
        return out

    def _process_frame(self, frame, channel):
        windowed = frame * self.window
        spectrum = np.fft.rfft(windowed, n=self.n_fft)
        magnitude = np.abs(spectrum)
        phase_complex = np.exp(1.0j * np.angle(spectrum))
        denoised_spec = np.clip(magnitude - self.noise_mean[:, channel], 0.0, None) * phase_complex
        time_frame = np.fft.irfft(denoised_spec, n=self.n_fft)
        time_frame = time_frame * self.window
        self._synth_num[:, channel] += time_frame
        self._synth_den[:, channel] += self.window * self.window

    def _step_ola(self):
        hop = self.hop_length
        den = np.maximum(self._synth_den[:hop], self._eps)
        ready = self._synth_num[:hop] / den
        self._append_pending(ready)
        self._synth_num = np.vstack(
            (self._synth_num[hop:], np.zeros((hop, self.channels), dtype=np.float64))
        )
        self._synth_den = np.vstack(
            (self._synth_den[hop:], np.zeros((hop, self.channels), dtype=np.float64))
        )

    def set_noise_profile(self, noise_audio):
        noise = self._normalize_chunk(noise_audio)
        if noise.shape[0] < self.n_fft:
            pad = self.n_fft - noise.shape[0]
            noise = np.vstack((noise, np.zeros((pad, self.channels))))
        bins = self.n_fft // 2 + 1
        mag_sum = np.zeros((bins, self.channels), dtype=np.float64)
        frames = 0
        start = 0
        while start + self.n_fft <= noise.shape[0]:
            frame = noise[start : start + self.n_fft]
            for ch in range(self.channels):
                spectrum = np.fft.rfft(frame[:, ch] * self.window, n=self.n_fft)
                mag_sum[:, ch] += np.abs(spectrum)
            frames += 1
            start += self.hop_length
        if frames == 0:
            raise ValueError("Noise profile is too short to estimate spectral statistics.")
        self.noise_mean = mag_sum / frames
        self._noise_frames = frames

    def update_noise_profile(self, noise_chunk, alpha=0.95):
        noise = self._normalize_chunk(noise_chunk)
        if noise.shape[0] < self.n_fft:
            return
        start = 0
        updated = False
        while start + self.n_fft <= noise.shape[0]:
            frame = noise[start : start + self.n_fft]
            for ch in range(self.channels):
                spectrum = np.fft.rfft(frame[:, ch] * self.window, n=self.n_fft)
                mag = np.abs(spectrum)
                if self._noise_frames == 0:
                    self.noise_mean[:, ch] = mag
                else:
                    self.noise_mean[:, ch] = alpha * self.noise_mean[:, ch] + (1.0 - alpha) * mag
            updated = True
            self._noise_frames += 1
            start += self.hop_length
        return updated

    def process_chunk(self, chunk):
        if self._noise_frames == 0:
            raise RuntimeError("Noise profile is not initialized. Call set_noise_profile first.")
        x = self._normalize_chunk(chunk)
        input_len = x.shape[0]
        self._analysis_buffer = np.vstack((self._analysis_buffer, x))

        while self._analysis_buffer.shape[0] >= self.n_fft:
            frame = self._analysis_buffer[: self.n_fft]
            for ch in range(self.channels):
                self._process_frame(frame[:, ch], ch)
            self._step_ola()
            self._analysis_buffer = self._analysis_buffer[self.hop_length :]

        out = self._pop_pending(input_len)
        if out.shape[0] < input_len:
            pad = input_len - out.shape[0]
            out = np.vstack((out, np.zeros((pad, self.channels), dtype=np.float64)))
        return out[:, 0] if self.channels == 1 else out

    def flush(self):
        if self._analysis_buffer.shape[0] > 0:
            pad = self.n_fft - self._analysis_buffer.shape[0]
            self._analysis_buffer = np.vstack(
                (self._analysis_buffer, np.zeros((pad, self.channels), dtype=np.float64))
            )
            frame = self._analysis_buffer[: self.n_fft]
            for ch in range(self.channels):
                self._process_frame(frame[:, ch], ch)
            self._step_ola()
            self._analysis_buffer = np.zeros((0, self.channels), dtype=np.float64)

        tail_num = self._synth_num.copy()
        tail_den = np.maximum(self._synth_den, self._eps)
        tail = tail_num / tail_den
        self._append_pending(tail)
        self._synth_num.fill(0.0)
        self._synth_den.fill(0.0)

        out = self._pending_output
        self._pending_output = np.zeros((0, self.channels), dtype=np.float64)
        return out[:, 0] if self.channels == 1 else out


def process_chunks(noise_profile_audio, chunk_iter, sample_rate=16000, n_fft=512, hop_length=128):
    noise = _to_2d(np.asarray(noise_profile_audio, dtype=np.float64))
    stream = StreamingSpectralSubtractor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        channels=noise.shape[1],
    )
    stream.set_noise_profile(noise)
    for chunk in chunk_iter:
        yield stream.process_chunk(chunk)
    tail = stream.flush()
    if tail.size:
        yield tail


def validate_streaming_equivalence(
    noise_profile_audio,
    noisy_audio,
    chunk_size=128,
    n_fft=512,
    hop_length=128,
):
    noise = _to_2d(np.asarray(noise_profile_audio, dtype=np.float64))
    noisy = _to_2d(np.asarray(noisy_audio, dtype=np.float64))
    stream = StreamingSpectralSubtractor(
        n_fft=n_fft, hop_length=hop_length, win_length=n_fft, channels=noisy.shape[1]
    )
    stream.set_noise_profile(noise)

    out_chunks = []
    for start in range(0, noisy.shape[0], chunk_size):
        out_chunks.append(stream.process_chunk(noisy[start : start + chunk_size]))
    tail = stream.flush()
    if tail.size:
        out_chunks.append(tail)

    streaming = np.concatenate([_to_2d(np.asarray(c, dtype=np.float64)) for c in out_chunks], axis=0)
    if streaming.shape[0] < noisy.shape[0]:
        pad = noisy.shape[0] - streaming.shape[0]
        streaming = np.vstack((streaming, np.zeros((pad, streaming.shape[1]))))
    streaming = streaming[: noisy.shape[0]]

    offline_channels = []
    for ch in range(noisy.shape[1]):
        offline_channels.append(
            spectral_subtraction(
                noise,
                noisy,
                ch,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                center=False,
            )
        )
    min_len = min(len(ch) for ch in offline_channels)
    offline = np.stack([ch[:min_len] for ch in offline_channels], axis=1)
    stream_trim = streaming[:min_len]
    mae = float(np.mean(np.abs(offline - stream_trim)))
    return {"offline_len": int(min_len), "stream_len": int(streaming.shape[0]), "mae": mae}


def run(noise_profile, noisy_input, FS, output_file=output_path):
    n, fs_n = sf.read(noise_profile)
    y, fs_y = sf.read(noisy_input)
    profile_dimensions = n.ndim
    input_dimensions = y.ndim

    if fs_n != FS:
        n = resample(n, fs_n, FS)
    if fs_y != FS:
        y = resample(y, fs_y, FS)

    assert profile_dimensions <= 2, "Only mono and stereo files supported."
    assert input_dimensions <= 2, "Only mono and stereo files supported."

    if profile_dimensions > input_dimensions:
        num_channels = profile_dimensions
        y = np.array([y, y], ndmin=num_channels)
        y = np.moveaxis(y, 0, 1)
    else:
        num_channels = input_dimensions
        if profile_dimensions != input_dimensions:
            n = np.array([n, n], ndmin=num_channels)
            n = np.moveaxis(n, 0, 1)

    for channel in range(num_channels):
        single_channel_output = spectral_subtraction(n, y, channel)
        if channel == 0:
            output_x = np.zeros((len(single_channel_output), num_channels))
        output_x[:, channel] = single_channel_output

    if num_channels > 1:
        output_x = np.moveaxis(output_x, 0, 1)
        output_x = librosa.to_mono(output_x)

    sf.write(output_file, output_x, FS, format="WAV")


def _parse_args():
    parser = argparse.ArgumentParser(description="Spectral subtraction denoising.")
    parser.add_argument("--noise", required=True, help="Path to noise profile WAV.")
    parser.add_argument("--input", required=True, help="Path to noisy input WAV.")
    parser.add_argument(
        "--output",
        default=output_path,
        help="Path to output WAV (default: ./subtracted_signal.wav).",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Target sample rate. Defaults to the input WAV sample rate.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.sr is None:
        _, input_sr = sf.read(args.input)
        target_sr = input_sr
    else:
        target_sr = args.sr
    run(args.noise, args.input, target_sr, output_file=args.output)
    print(f"Wrote denoised file to: {args.output}")


if __name__ == "__main__":
    main()
