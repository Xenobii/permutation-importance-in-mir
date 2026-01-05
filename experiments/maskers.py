from typing import Tuple, Optional
import numpy as np
import pretty_midi

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Constants from basic_pitch (duplicated to avoid circular imports)
AUDIO_SAMPLE_RATE = 22050
FFT_HOP = 256
AUDIO_WINDOW_LENGTH = 2  # seconds
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP
N_OVERLAPPING_FRAMES = 30


def midi_to_hz(midi: torch.Tensor) -> torch.Tensor:
    return 440.0 * (2.0 ** ((midi - 69) / 12))


class FeatureMasker(nn.Module):
    def __init__(
        self,
        spec_type: str = "cqt",
        bins_per_octave: int = 3 * 12,
        f_min: float = 27.7,
        f_max: Optional[float] = None,
    ):
        """
        Base class for spectrogram feature masking
        """
        super().__init__()
        self.spec_type = spec_type
        self.bins_per_octave = bins_per_octave
        self.f_min = f_min
        self.f_max = f_max

    def load_label(self, midi_path: str, audio_length_samples: Optional[int] = None):
        """
        Loads the MIDI as a binary piano roll and registers it as a buffer.
        Windowing is deferred to _align_label_to_input() where actual input dimensions are known.
        """
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        if audio_length_samples is not None:
            duration = audio_length_samples / AUDIO_SAMPLE_RATE
        else:
            duration = midi_data.get_end_time()

        piano_roll = midi_data.get_piano_roll(fs=ANNOTATIONS_FPS)
        piano_roll = (piano_roll > 0).astype(np.float32)

        # make piano_roll match the audio duration in annotation frames
        n_frames_expected = int(np.ceil(duration * ANNOTATIONS_FPS))
        if piano_roll.shape[1] < n_frames_expected:
            piano_roll = np.pad(piano_roll, ((0, 0), (0, n_frames_expected - piano_roll.shape[1])), mode="constant")
        elif piano_roll.shape[1] > n_frames_expected:
            piano_roll = piano_roll[:, :n_frames_expected]

        # Store raw piano roll - windowing happens in _align_label_to_input
        label_tensor = torch.from_numpy(piano_roll).float()  # [N, T_total]
        self.register_buffer("label_raw", label_tensor)

    def _align_label_to_input(self, B: int, T: int) -> torch.Tensor:
        """
        Window the raw piano roll to match input spectrogram dimensions.
        
        Args:
            B: Number of batches (windows) from the input spectrogram
            T: Number of time frames per window from the input spectrogram
            
        Returns:
            label: Tensor of shape [B, N, T] aligned with input
        """
        if not hasattr(self, "label_raw"):
            raise RuntimeError("No label_raw buffer found. Call load_label(...) first.")
        
        N, T_total = self.label_raw.shape
        
        # Calculate windowing parameters (same as basic_pitch inference)
        overlap_len_samples = N_OVERLAPPING_FRAMES * FFT_HOP
        hop_size_samples = AUDIO_N_SAMPLES - overlap_len_samples
        
        # Convert to annotation frames
        hop_size_frames = hop_size_samples // FFT_HOP
        overlap_frames = overlap_len_samples // FFT_HOP
        
        # Pad start to match basic_pitch's padding
        pad_start_frames = overlap_frames // 2
        piano_roll = torch.nn.functional.pad(self.label_raw, (pad_start_frames, 0))
        
        # Pad end if needed to accommodate all windows
        T_padded = piano_roll.shape[1]
        total_needed = (B - 1) * hop_size_frames + T
        if total_needed > T_padded:
            piano_roll = torch.nn.functional.pad(piano_roll, (0, total_needed - T_padded))
        
        # Extract windows to match input batches
        windows = []
        for i in range(B):
            start = i * hop_size_frames
            end = start + T
            windows.append(piano_roll[:, start:end])
        
        return torch.stack(windows, dim=0)  # [B, N, T]

    def note_to_cqt_bin(
        self,
        midi_notes: torch.Tensor,  # [N]
        F: int
    ) -> torch.Tensor:
        """
        Maps MIDI notes to CQT bin indices
        """
        freqs = 440.0 * (2.0 ** ((midi_notes - 69) / 12))
        bins = self.bins_per_octave * torch.log2(freqs / self.f_min)
        bins = bins.round().long()

        return bins.clamp(min=0, max=F - 1)

    def note_to_mel_bin(
        self,
        midi_notes: torch.Tensor,  # [N]
        F: int
    ) -> torch.Tensor:
        """
        Approximate MIDI → mel-bin mapping
        """
        freqs = 440.0 * (2.0 ** ((midi_notes - 69) / 12))

        mel = 2595.0 * torch.log10(1.0 + freqs / 700.0)
        mel_min = 2595.0 * torch.log10(1.0 + self.f_min / 700.0)

        if self.f_max is None:
            raise ValueError("f_max must be set for mel masking")

        mel_max = 2595.0 * torch.log10(1.0 + self.f_max / 700.0)

        bins = (mel - mel_min) / (mel_max - mel_min) * (F - 1)
        return bins.round().long().clamp(0, F - 1)

    def note_to_bin(
        self,
        midi_notes: torch.Tensor,
        F: int
    ) -> torch.Tensor:
        if self.spec_type == "cqt":
            return self.note_to_cqt_bin(midi_notes, F)
        elif self.spec_type == "log-mel":
            return self.note_to_mel_bin(midi_notes, F)
        else:
            raise ValueError(f"Unknown spec_type: {self.spec_type}")

    @staticmethod
    def build_fundamental_mask(
        y: torch.Tensor,
        note_bins: torch.Tensor,
        F: int
    ) -> torch.Tensor:
        """
        Returns:
            fund_mask: BoolTensor [B, T, F]
        """
        B, N, T = y.shape
        device = y.device

        # [B, N, T] -> [B, T, N]
        y_bt_n = y.permute(0, 2, 1).bool()

        bins = note_bins.view(1, 1, N).expand(B, T, N)
        fund_mask = torch.zeros(B, T, F, device=device, dtype=torch.bool)

        fund_mask.scatter_(dim=2, index=bins, src=y_bt_n)

        return fund_mask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def plot(
        self,
        x: torch.Tensor,
        batch: int = 0,
        figsize: Tuple[int, int] = (12, 5),
        cmap: str = "magma",
        label_cmap: str = "gray_r",
        show: bool = True,
    ):
        """
        Plot spectrogram x (B, T, F) and piano-roll label (B, N, T) for given batch index.
        Returns (fig, (ax_spec, ax_label)).
        """
        if x.dim() != 3:
            raise ValueError("x must be a 3D tensor with shape [B, T, F]")
        if not hasattr(self, "label_raw"):
            raise RuntimeError("No label_raw buffer found. Call load_label(...) before plotting.")

        B, T, F = x.shape
        # Align label to input dimensions for plotting
        label = self._align_label_to_input(B, T)  # [B, N, T]

        x_np = x[batch].detach().cpu().numpy()   # [T, F]
        label_np = label[batch].detach().cpu().numpy()  # [N, T]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        im1 = ax1.imshow(x_np.T, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")
        ax1.set_title("Spectrogram (freq bins x time frames)")
        ax1.set_xlabel("Time frames")
        ax1.set_ylabel("Frequency bins")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(label_np, aspect="auto", origin="lower", cmap=label_cmap, interpolation="nearest")
        ax2.set_title("Piano roll (MIDI pitch x time frames)")
        ax2.set_xlabel("Time frames")
        ax2.set_ylabel("MIDI pitch")
        fig.tight_layout()

        if show:
            plt.show()

        return fig, (ax1, ax2)


class FundamentalMasking(FeatureMasker):
    def __init__(
        self,
        spec_type: str = "cqt",
        bins_per_octave: int = 3 * 12,
        f_min: float = 27.7,
        f_max: Optional[float] = None,
    ):
        """
        Mask the fundamental frequencies of the active notes
        """
        super().__init__(spec_type, bins_per_octave, f_min, f_max)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device

        # Align label to input dimensions
        label = self._align_label_to_input(B, T)  # [B, N, T]
        label = label.to(device)
        N = label.shape[1]

        midi = torch.arange(N, device=device)
        note_bins = self.note_to_bin(midi, F)

        fund_mask = self.build_fundamental_mask(label, note_bins, F)

        x_out = x.clone()
        x_out[fund_mask] = 0.0

        self.plot(x_out, 1)

        return x_out


class HarmonicMasking(FeatureMasker):
    def __init__(
        self,
        spec_type: str = "cqt",
        bins_per_octave: int = 3 * 12,
        f_min: float = 27.7,
        f_max: Optional[float] = None,
    ):
        """
        Keep only the fundamental frequencies of active notes
        """
        super().__init__(spec_type, bins_per_octave, f_min, f_max)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device

        # Align label to input dimensions
        label = self._align_label_to_input(B, T)  # [B, N, T]
        label = label.to(device)
        N = label.shape[1]

        midi = torch.arange(N, device=device)
        note_bins = self.note_to_bin(midi, F)

        fund_mask = self.build_fundamental_mask(label, note_bins, F)

        x_out = torch.zeros_like(x)
        x_out[fund_mask] = x[fund_mask]

        return x_out


class SoftFundamentalMasking(FeatureMasker):
    def __init__(
        self,
        spec_type: str = "cqt",
        bins_per_octave: int = 3 * 12,
        f_min: float = 27.7,
        f_max: Optional[float] = None,
        temperature: float = 0.1,
        eps: float = 1e-8,
    ):
        """
        Softly emphasize fundamental frequencies using softmax attenuation
        """
        super().__init__(spec_type, bins_per_octave, f_min, f_max)
        self.temperature = temperature
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device

        # Align label to input dimensions
        label = self._align_label_to_input(B, T)  # [B, N, T]
        label = label.to(device)
        N = label.shape[1]

        midi = torch.arange(N, device=device)
        note_bins = self.note_to_bin(midi, F)

        fund_mask = self.build_fundamental_mask(label, note_bins, F)

        # Build softmax scores
        scores = torch.full_like(x, float("-inf"))
        scores[fund_mask] = x[fund_mask] / self.temperature

        # Stable softmax over frequency
        weights = torch.softmax(scores, dim=-1)

        # Attenuate
        x_out = x * weights

        return x_out
