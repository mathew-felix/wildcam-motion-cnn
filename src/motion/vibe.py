from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ViBeConfig:
    num_samples: int = 20
    min_matches: int = 2
    radius: int = 20
    subsampling_factor: int = 16  # update prob = 1 / subsampling_factor


class ViBe:
    """
    Simple ViBe background subtraction (grayscale, uint8).
    Foreground mask: 255=FG, 0=BG.
    """

    def __init__(self, cfg: ViBeConfig, rng_seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(rng_seed)
        self.samples: np.ndarray | None = None  # (N, H, W) uint8

    @staticmethod
    def _clip_coords(y: np.ndarray, x: np.ndarray, h: int, w: int):
        y = np.clip(y, 0, h - 1)
        x = np.clip(x, 0, w - 1)
        return y, x

    def initialize(self, first_gray: np.ndarray) -> None:
        if first_gray.dtype != np.uint8:
            raise ValueError("ViBe expects uint8 grayscale frames.")
        if first_gray.ndim != 2:
            raise ValueError("ViBe expects 2D grayscale frames.")

        h, w = first_gray.shape
        n = self.cfg.num_samples
        self.samples = np.empty((n, h, w), dtype=np.uint8)

        yy, xx = np.indices((h, w))
        offsets = np.array(
            [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),  (0, 0),  (0, 1),
             (1, -1),  (1, 0),  (1, 1)],
            dtype=np.int32
        )

        for i in range(n):
            idx = self.rng.integers(0, len(offsets), size=(h, w))
            oy = offsets[idx, 0]
            ox = offsets[idx, 1]
            ny, nx = self._clip_coords(yy + oy, xx + ox, h, w)
            self.samples[i] = first_gray[ny, nx]

    def apply(self, gray: np.ndarray) -> np.ndarray:
        if self.samples is None:
            self.initialize(gray)

        assert self.samples is not None
        n, h, w = self.samples.shape

        diff = np.abs(self.samples.astype(np.int16) - gray.astype(np.int16)[None, :, :])
        matches = (diff < self.cfg.radius)
        count = matches.sum(axis=0)

        is_bg = (count >= self.cfg.min_matches)
        fg_mask = (~is_bg).astype(np.uint8) * 255

        if self.cfg.subsampling_factor > 0:
            update = is_bg & (self.rng.integers(0, self.cfg.subsampling_factor, size=(h, w)) == 0)
        else:
            update = is_bg

        ys, xs = np.where(update)
        if ys.size > 0:
            ridx = self.rng.integers(0, n, size=ys.size)
            self.samples[ridx, ys, xs] = gray[ys, xs]

            offsets8 = np.array(
                [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)],
                dtype=np.int32
            )
            oidx = self.rng.integers(0, len(offsets8), size=ys.size)
            oy = offsets8[oidx, 0]
            ox = offsets8[oidx, 1]
            ny, nx = self._clip_coords(ys + oy, xs + ox, h, w)

            ridx2 = self.rng.integers(0, n, size=ys.size)
            self.samples[ridx2, ny, nx] = gray[ys, xs]

        return fg_mask
