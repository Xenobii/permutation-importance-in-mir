import torch
import torch.nn as nn
from typing import Tuple, Optional



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
        self.spec_type = spec_type
        self.bins_per_octave = bins_per_octave
        self.f_min = f_min
        self.f_max = f_max


    def load_label(self, midi_path: str):
        """
        Loads the label as a buffer
        """
        # self.register_buffer("label", label)

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
        _, N, _ = self.label.shape
        device = x.device

        midi = torch.arange(N, device=device)
        note_bins = self.note_to_bin(midi, F)

        fund_mask = self.build_fundamental_mask(self.label, note_bins, F)

        x_out = x.clone()
        x_out[fund_mask] = 0.0

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
        _, N, _ = self.label.shape
        device = x.device

        midi = torch.arange(N, device=device)
        note_bins = self.note_to_bin(midi, F)

        fund_mask = self.build_fundamental_mask(self.label, note_bins, F)

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
        _, N, _ = self.label.shape
        device = x.device

        midi = torch.arange(N, device=device)
        note_bins = self.note_to_bin(midi, F)

        fund_mask = self.build_fundamental_mask(self.label, note_bins, F)

        # Build softmax scores
        scores = torch.full_like(x, float("-inf"))
        scores[fund_mask] = x[fund_mask] / self.temperature

        # Stable softmax over frequency
        weights = torch.softmax(scores, dim=-1)

        # Attenuate
        x_out = x * weights

        return x_out
