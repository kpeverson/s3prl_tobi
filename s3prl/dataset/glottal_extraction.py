
import librosa
import numpy as np
import scipy.signal as signal
import torch
import torchaudio

class GlottalExtractor:

    def __init__(self, sr, lpc_window_size, lpc_window_stride, lpc_order, lpc_window="hamming", lpf_cutoff=None, lpf_order=4, half_band_signal=False, energy_threshold=1e-4):
        self.sr = sr
        self.lpc_window_size = int(sr * lpc_window_size)
        self.lpc_window_stride = int(sr * lpc_window_stride)
        self.lpc_order = lpc_order
        self.lpc_window = lpc_window
        self.lpf_cutoff = lpf_cutoff
        self.lpf_order = lpf_order
        self.half_band_signal = half_band_signal
        self.energy_threshold = energy_threshold
        if self.half_band_signal:
            # divide energy_threshold by 2 since frames are half as long
            self.energy_threshold /= 2

    def half_band(self, x):
        """
        input:
            x: torch.Tensor, (T,) speech signal
        return:
            x: torch.Tensor, (T,) half-band speech signal via downsampling
        """
        if self.half_band_signal:
            x = torchaudio.functional.resample(x.unsqueeze(0), orig_freq=self.sr, new_freq=self.sr//2).squeeze(0)
        return x
    
    def undo_half_band(self, x):
        """
        input:
            x: np.ndarray, (T,) half-band speech signal
        return:
            x: np.ndarray, (T,) upsampled speech signal
        """
        if self.half_band_signal:
            x = torchaudio.functional.resample(torch.from_numpy(x).unsqueeze(0), orig_freq=self.sr//2, new_freq=self.sr).squeeze(0).numpy()
        return x

    def lpf(self, x):
        if self.lpf_cutoff is not None:
            sos = signal.butter(
                self.lpf_order,
                self.lpf_cutoff,
                "low",
                fs=self.sr,
                output="sos",
            )
            x = signal.sosfiltfilt(sos, x)
        return x

    def inverse_filter(self, x_frame, a, idx):
        if np.sum(x_frame**2) < self.energy_threshold:
            return x_frame
        x_frame_hat = signal.lfilter(
            np.hstack([[0], -1 * a[1:]]), [1], x_frame
        )
        glottal = x_frame - x_frame_hat
        return glottal

    def extract(self, x, idx):
        """
        input:
            x: torch.Tensor (will convert to np.ndarray), (T,) speech signal
            idx: int, index of the signal in the batch (for logging purposes)
        return:
            glottal_source: np.ndarray, (T,) glottal source signal
        """
        x = self.half_band(x)
        x = x.numpy()

        glottal_source = np.zeros_like(x)
        frames = librosa.util.frame(
            x, frame_length=self.lpc_window_size, hop_length=self.lpc_window_stride
        ).T

        if self.lpc_window == "hamming":
            window = np.hamming(self.lpc_window_size)
        else:
            raise ValueError(f"Unsupported window type: {self.lpc_window}")

        for i, frame in enumerate(frames):
            frame = frame * window
            a = librosa.lpc(frame, order=self.lpc_order)
            glottal_frame = self.inverse_filter(frame, a, idx)
            glottal_source[i*self.lpc_window_stride : i*self.lpc_window_stride + self.lpc_window_size] += glottal_frame

        glottal_source = self.undo_half_band(glottal_source)
        glottal_source = self.lpf(glottal_source)

        return glottal_source