import torch
import torch.nn as nn
import torchaudio


class CQTPermuter(nn.Module):
    def __init__(self, seed: int=0):
        """
        Base class for CQT frequency-axis permutations.
        """
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        """
        return NotImplementedError


class CQTRandPerm(CQTPermuter):
    def __init__(self, p: float=0.1, seed: int=0):
        """
        Random per-frame bin swapping.

        p: probability of swapping a given bin with another bin
        """
        super().__init__(seed)
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device

        base = torch.arange(F, device=device)
        noise = torch.rand(B, T, F, generator=self.gen, device=device)

        # Noisy identity permutation
        scores = base + (noise < self.p) * torch.rand_like(noise)
        perm_idx = torch.argsort(scores, dim=-1)

        return torch.gather(x, dim=-1, index=perm_idx)


class CQTHighFreqPerm(CQTPermuter):
    def __init__(self, start_bin: int, seed: int = 0):
        """
        Permutes only high-frequency bins (>= start_bin),
        independently per frame.
        """
        super().__init__(seed)
        self.start_bin = start_bin

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device
        assert 0 <= self.start_bin < F

        # Identity permutation
        perm_idx = torch.arange(F, device=device).expand(B, T, F).clone()

        hf_len = F - self.start_bin
        hf_perm = torch.argsort(
            torch.rand(B, T, hf_len, generator=self.gen, device=device),
            dim=-1
        ) + self.start_bin

        perm_idx[..., self.start_bin:] = hf_perm

        return torch.gather(x, dim=-1, index=perm_idx)


class CQTMicrotonalPerm(CQTPermuter):
    def __init__(self, bins_per_semitone: int, seed: int = 0):
        """
        Permutes bins within each semitone group.

        bins_per_semitone: number of CQT bins per semitone
        """
        super().__init__(seed)
        self.bins_per_semitone = bins_per_semitone

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device

        bps = self.bins_per_semitone
        n_semitones = F // bps
        assert n_semitones * bps == F, "F must be divisible by bins_per_semitone"

        # Generate one permutation per semitone
        perm = torch.argsort(
            torch.rand(n_semitones, bps, generator=self.gen, device=device),
            dim=-1
        )

        # Convert to absolute frequency indices
        perm_idx = (
            perm
            + torch.arange(n_semitones, device=device)[:, None] * bps
        ).reshape(-1)

        return x[..., perm_idx]