import os
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Subset
import torchaudio
import torchaudio.functional as AF  # kept for consistency

from utils import (safe_load_wav, mono, resample,
    mix_snr, crop, build_cum_prob, sample_from_cum_prob)

class NoiseAugment:
    """Draw random SNR uniformly in [5, 20] dB, then mix one noise type (or keep clean)."""
    def __init__(
        self,
        root_path: str,
        noise_mix: Optional[Dict[str, float]] = None,
        snr_range: Tuple[float, float] = (5.0, 20.0),
        target_sr: int = 16000,
    ):
        self.target_sr = int(target_sr)
        self.snr_low, self.snr_high = map(float, snr_range)

        requested = noise_mix or {"clean": 1.0}  # default

        noise_dir = os.path.join(root_path, "_background_noise_")
        # swapped: load three environmental noises instead of pink/white
        self._dishes = self.load(os.path.join(noise_dir, "doing_the_dishes.wav"))
        self._tap    = self.load(os.path.join(noise_dir, "running_tap.wav"))
        self._bike   = self.load(os.path.join(noise_dir, "exercise_bike.wav"))

        keys = ["clean"]
        if self._dishes is not None: keys.append("dishes")
        if self._tap is not None:    keys.append("tap")
        if self._bike is not None:   keys.append("biking")
        self.keys = keys

        probs = [float(requested.get(k, 0.0)) for k in self.keys]
        if sum(probs) == 0:
            probs = [1.0] + [0.0] * (len(self.keys) - 1)
        self.cumulative = build_cum_prob(probs)

    def load(self, path: str) -> Optional[torch.Tensor]:
        if not os.path.isfile(path):
            return None
        wav, sr = safe_load_wav(path)
        wav = mono(wav)
        return resample(wav, sr, self.target_sr).contiguous()

    def apply(self, wav: torch.Tensor, sr: int):
        wav = mono(wav)
        if int(sr) != self.target_sr:
            wav = resample(wav, int(sr), self.target_sr)

        mode = sample_from_cum_prob(self.keys, self.cumulative)

        if mode == "dishes" and self._dishes is not None:
            seg = crop(self._dishes, wav.numel())
            wav = mix_snr(wav, seg, random.uniform(self.snr_low, self.snr_high))
        elif mode == "tap" and self._tap is not None:
            seg = crop(self._tap, wav.numel())
            wav = mix_snr(wav, seg, random.uniform(self.snr_low, self.snr_high))
        elif mode == "biking" and self._bike is not None:
            seg = crop(self._bike, wav.numel())
            wav = mix_snr(wav, seg, random.uniform(self.snr_low, self.snr_high))
        # else "clean": do nothing

        return wav, self.target_sr

    def __call__(self, wav: torch.Tensor, sr: int):
        return self.apply(wav, sr)
