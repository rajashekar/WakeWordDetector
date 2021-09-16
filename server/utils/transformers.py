from torchaudio.transforms import MelSpectrogram
from typing import Iterable

import torch.nn as nn
import torch


num_mels = 40  # https://en.wikipedia.org/wiki/Mel_scale
num_fft = 512  # window length - Fast Fourier Transform
hop_length = 200  # making hops of size hop_length each time to sample the next window


# Calculate mel spectrograms from audio
def audio_transform(audio_data, device, sample_rate, skip_log=False):
    # Transformations
    # Mel-scale spectrogram is a combination of Spectrogram and mel scale conversion
    # 1. compute FFT - for each window to transform from time domain to frequency domain
    # 2. Generate Mel Scale - Take entire freq spectrum & seperate to n_mels evenly spaced
    #    frequencies. (not by distance on freq domain but distance as it is heard by human ear)
    # 3. Generate Spectrogram - For each window, decompose the magnitude of the signal
    #    into its components, corresponding to the frequencies in the mel scale.
    mel_spectrogram = MelSpectrogram(
        n_mels=num_mels, sample_rate=sample_rate, n_fft=num_fft, hop_length=hop_length, norm="slaney"
    )
    mel_spectrogram.to(device)
    if skip_log:
        log_mels = mel_spectrogram(audio_data.float())
    else:
        log_mels = mel_spectrogram(audio_data.float()).add_(1e-7).log_().contiguous()
    # returns (channel, n_mels, time)
    return log_mels.to(device)


# Calculate Zero Mean Unit Variance
class ZmuvTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(1))
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("mean2", torch.zeros(1))

    def update(self, data, mask=None):
        with torch.no_grad():
            if mask is not None:
                data = data * mask
                mask_size = mask.sum().item()
            else:
                mask_size = data.numel()
            self.mean = (data.sum() + self.mean * self.total) / (self.total + mask_size)
            self.mean2 = ((data ** 2).sum() + self.mean2 * self.total) / (self.total + mask_size)
            self.total += mask_size

    def initialize(self, iterable: Iterable[torch.Tensor]):
        for ex in iterable:
            self.update(ex)

    @property
    def std(self):
        return (self.mean2 - self.mean ** 2).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std
