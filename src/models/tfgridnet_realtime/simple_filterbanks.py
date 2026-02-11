from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from espnet2.enh.layers.stft import Stft, IStft


class SimpleSTFTEncoder(nn.Module):
    def __init__(self, kernel_size: int, stride: int, window_type: str = "hann"):
        super().__init__()
        self.n_fft = kernel_size
        self.stride = stride
        self.stft = Stft(
            n_fft=self.n_fft,
            hop_length=self.stride,
            win_length=self.n_fft,
            window=window_type,
            center=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError(f"Expected [batch, channels, samples], got {inputs.shape}")
        batch, channels, samples = inputs.shape
        # Stft handles [B, T] or [B, C, T]; we flatten channels manually for clarity.
        stft = self.stft(inputs.reshape(batch * channels, samples))
        freq_bins = self.n_fft // 2 + 1
        if stft.dim() != 3:
            raise ValueError(f"Expected STFT output with 3 dims, got {stft.shape}")
        if stft.shape[1] == freq_bins and stft.shape[2] != freq_bins:
            # [B*C, F, T]
            pass
        elif stft.shape[2] == freq_bins and stft.shape[1] != freq_bins:
            # [B*C, T, F] -> transpose
            stft = stft.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected STFT shape {stft.shape} for n_fft={self.n_fft}")
        # Reshape back to [B, C, F, T]
        stft = stft.view(batch, channels, freq_bins, stft.shape[-1])
        # Stack real and imaginary parts along the frequency axis to match TFGridNet expectations.
        stft = torch.cat([stft.real, stft.imag], dim=2)  # [B, C, 2*freq, T]
        return stft


class SimpleSTFTDecoder(nn.Module):
    def __init__(self, kernel_size: int, stride: int, window_type: str = "hann"):
        super().__init__()
        self.n_fft = kernel_size
        self.stride = stride
        self.istft = IStft(
            n_fft=self.n_fft,
            hop_length=self.stride,
            win_length=self.n_fft,
            window=window_type,
            center=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 4:
            raise ValueError(f"Expected [batch, channels, freq*2, time], got {inputs.shape}")
        batch, channels, doubled_freq, time = inputs.shape
        if doubled_freq % 2 != 0:
            raise ValueError("Frequency dimension must be even to split real/imag parts.")
        freq = doubled_freq // 2
        real = inputs[:, :, :freq]
        imag = inputs[:, :, freq:]
        stft = torch.stack([real, imag], dim=-1)  # [B, C, freq, T, 2]
        stft = stft.view(batch * channels, freq, time, 2)
        stft_complex = torch.view_as_complex(stft)
        # Expected output length that mirrors the encoder (center=False)
        expected_len = self.stride * (time - 1) + self.n_fft
        waveform = self.istft(stft_complex, length=expected_len)
        waveform = waveform.view(batch, channels, -1)
        return waveform


def make_enc_dec(
    kind: str,
    *,
    n_filters: int,
    kernel_size: int,
    stride: int,
    window_type: str = "hann",
) -> Tuple[nn.Module, nn.Module]:
    if kind.lower() != "stft":
        raise ValueError("Only 'stft' encoders are supported in the simplified backend.")
    encoder = SimpleSTFTEncoder(kernel_size=kernel_size, stride=stride, window_type=window_type)
    decoder = SimpleSTFTDecoder(kernel_size=kernel_size, stride=stride, window_type=window_type)
    return encoder, decoder
