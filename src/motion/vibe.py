from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class ViBeConfig:
    """
    Configuration for ViBe background subtraction.

    ViBe maintains a set of N sample pixels per location. A pixel is classified as
    background if enough samples are "close" (within `radius`) to the current pixel.

    This config also supports two optional additions:
      1) AT  = Adaptive Thresholding: dynamic `min_matches` based on illumination
      2) ALR = Adaptive Learning Rate: dynamic update probability based on motion intensity
    """

    # -----------------------------
    # Core ViBe parameters
    # -----------------------------
    num_samples: int = 20
    # Minimum number of close samples required to classify pixel as BACKGROUND.
    # Larger min_matches => more strict background decision => more foreground.
    min_matches: int = 2
    # Distance threshold (in grayscale intensity) for a sample to "match" current pixel.
    radius: int = 20
    # Background model update probability is ~ 1 / subsampling_factor.
    # Larger subsampling_factor => fewer updates (slower model adaptation).
    subsampling_factor: int = 16  # baseline: update prob = 1/subsampling_factor

    # -----------------------------
    # AT: Adaptive thresholding (dynamic min_matches)
    # -----------------------------
    adaptive_min_matches: bool = False

    # We map mean intensity -> min_matches using linear interpolation:
    #   mean <= intensity_min  -> min_matches_min
    #   mean >= intensity_max  -> min_matches_max
    # This makes the detector behave differently for dark (night/IR) vs bright (day).
    intensity_min: float = 40.0
    intensity_max: float = 160.0
    min_matches_min: int = 1
    min_matches_max: int = 4

    # EMA smoothing for mean intensity to reduce flicker in min_matches decisions.
    # 0.0 disables smoothing. Typical useful range: 0.1–0.3.
    intensity_ema_alpha: float = 0.2

    # -----------------------------
    # ALR: Adaptive learning rate / adaptive background update
    # -----------------------------
    adaptive_update: bool = False

    # Motion intensity is measured as foreground ratio in [0,1] for a frame:
    #   fg_ratio = (#foreground_pixels) / (H*W)
    #
    # We map fg_ratio -> dynamic update subsampling factor:
    #   fg_ratio <= motion_low  -> update_sf_min (fast updates, model adapts quickly)
    #   fg_ratio >= motion_high -> update_sf_max (slow updates, avoid absorbing motion)
    motion_low: float = 0.001
    motion_high: float = 0.05

    # Bounds for the dynamic subsampling factor:
    # - update_sf_min: faster update (more frequent model refresh)
    # - update_sf_max: slower update (prevents absorbing moving objects / dynamic bg)
    update_sf_min: int = 8
    update_sf_max: int = 64

    # EMA smoothing for fg_ratio to reduce flicker in update rate.
    motion_ema_alpha: float = 0.2


# =============================================================================
# ViBe model
# =============================================================================
class ViBe:
    """
    ViBe background subtraction (grayscale uint8).
    Output mask uses 255 = foreground, 0 = background.

    Core idea:
      For each pixel location (y,x), maintain `num_samples` past sample values.
      A pixel is background if at least `min_matches` samples lie within `radius`.

    Added options:
      (1) AT: adaptive min_matches based on frame illumination (mean intensity)
      (2) ALR: adaptive update probability based on motion intensity (fg_ratio)
    """

    def __init__(self, cfg: ViBeConfig, rng_seed: int = 0):
        self.cfg = cfg

        # Random generator used for:
        # - initial sampling from neighbors
        # - stochastic model update
        self.rng = np.random.default_rng(rng_seed)

        # Background model samples:
        # shape = (N, H, W), dtype = uint8
        self.samples: np.ndarray | None = None

        # -----------------------------
        # Adaptive state (EMA smoothing)
        # -----------------------------
        self._ema_mean_intensity: float | None = None
        self._ema_motion_intensity: float | None = None

        # -----------------------------
        # Instrumentation (optional)
        # Useful if you want to log behavior into CSV for debugging / analysis.
        # -----------------------------
        self.last_min_matches_used: int = int(cfg.min_matches)
        self.last_mean_intensity_used: float = 0.0
        self.last_motion_intensity_used: float = 0.0
        self.last_update_sf_used: int = int(cfg.subsampling_factor)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _clip_coords(y: np.ndarray, x: np.ndarray, h: int, w: int):
        """
        Clip coordinate arrays to valid image boundaries.
        Used when sampling neighbor pixels.
        """
        y = np.clip(y, 0, h - 1)
        x = np.clip(x, 0, w - 1)
        return y, x

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def initialize(self, first_gray: np.ndarray) -> None:
        """
        Initialize the background model using the first grayscale frame.

        For each sample layer i (0..N-1), we fill samples by picking a random
        neighbor pixel from the first frame (classic ViBe initialization).
        """
        if first_gray.dtype != np.uint8:
            raise ValueError("ViBe expects uint8 grayscale frames.")
        if first_gray.ndim != 2:
            raise ValueError("ViBe expects 2D grayscale frames.")

        h, w = first_gray.shape
        n = int(self.cfg.num_samples)

        # Allocate sample bank
        self.samples = np.empty((n, h, w), dtype=np.uint8)

        # Pixel coordinate grids
        yy, xx = np.indices((h, w))

        # 8-neighborhood + center
        offsets = np.array(
            [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),  (0, 0),  (0, 1),
             (1, -1),  (1, 0),  (1, 1)],
            dtype=np.int32
        )

        # Build each sample layer by sampling random neighbors from the first frame
        for i in range(n):
            idx = self.rng.integers(0, len(offsets), size=(h, w))
            oy = offsets[idx, 0]
            ox = offsets[idx, 1]
            ny, nx = self._clip_coords(yy + oy, xx + ox, h, w)
            self.samples[i] = first_gray[ny, nx]

        # Reset adaptive state
        self._ema_mean_intensity = float(first_gray.mean())
        self._ema_motion_intensity = 0.0

        # Initialize instrumentation
        self.last_mean_intensity_used = float(self._ema_mean_intensity)
        self.last_motion_intensity_used = 0.0
        self.last_min_matches_used = int(self.cfg.min_matches)
        self.last_update_sf_used = int(self.cfg.subsampling_factor)

    # -------------------------------------------------------------------------
    # AT: Adaptive thresholding
    # -------------------------------------------------------------------------
    def _compute_dynamic_min_matches(self, gray: np.ndarray) -> int:
        """
        Compute a per-frame `min_matches` based on illumination (mean intensity).

        Motivation:
          - Night/IR frames often have different noise/contrast characteristics than day frames.
          - Fixed `min_matches` can be too conservative in dark scenes (misses) or too sensitive in bright scenes (FP).

        Implementation:
          - Compute mean intensity of the frame.
          - Smooth it with EMA to reduce flicker.
          - Linearly map mean intensity [intensity_min, intensity_max] to
            [min_matches_min, min_matches_max].
        """
        mean_i = float(gray.mean())

        # EMA smoothing to prevent unstable min_matches changes frame-to-frame
        a = float(self.cfg.intensity_ema_alpha)
        if self._ema_mean_intensity is None or a <= 0.0:
            ema = mean_i
        else:
            ema = (1.0 - a) * self._ema_mean_intensity + a * mean_i

        self._ema_mean_intensity = ema
        self.last_mean_intensity_used = ema

        lo = float(self.cfg.intensity_min)
        hi = float(self.cfg.intensity_max)
        if hi <= lo + 1e-9:
            # Degenerate config: fall back to fixed baseline value
            return int(self.cfg.min_matches)

        # Normalize intensity into [0,1]
        t = (ema - lo) / (hi - lo)
        t = max(0.0, min(1.0, t))

        mm_lo = int(self.cfg.min_matches_min)
        mm_hi = int(self.cfg.min_matches_max)

        # Linear interpolation + clamp
        mm = mm_lo + t * (mm_hi - mm_lo)
        mm_int = int(round(mm))
        mm_int = max(1, min(mm_int, int(self.cfg.num_samples)))
        return mm_int

    # -------------------------------------------------------------------------
    # ALR: Adaptive learning rate / adaptive update
    # -------------------------------------------------------------------------
    def _compute_dynamic_update_sf(self, fg_ratio: float) -> int:
        """
        Compute dynamic subsampling_factor (update period) based on motion intensity.

        Motivation:
          - If a lot of pixels are foreground (high motion), updating the background too fast
            can "absorb" moving animals into the background model.
          - If the scene is stable (low motion), faster updates help adapt to slow changes
            (lighting drift, background changes).

        Implementation:
          - Smooth fg_ratio with EMA to reduce flicker.
          - Linearly map fg_ratio [motion_low, motion_high] to [update_sf_min, update_sf_max].
            Higher fg_ratio => larger sf => fewer updates (slower learning).
        """
        fg_ratio = float(max(0.0, min(1.0, fg_ratio)))

        # EMA smoothing for stability
        a = float(self.cfg.motion_ema_alpha)
        if self._ema_motion_intensity is None or a <= 0.0:
            ema = fg_ratio
        else:
            ema = (1.0 - a) * self._ema_motion_intensity + a * fg_ratio

        self._ema_motion_intensity = ema
        self.last_motion_intensity_used = float(ema)

        low = float(self.cfg.motion_low)
        high = float(self.cfg.motion_high)
        if high <= low + 1e-12:
            # Degenerate config: fall back to baseline subsampling_factor
            return int(self.cfg.subsampling_factor)

        # Normalize motion into [0,1]
        t = (ema - low) / (high - low)
        t = max(0.0, min(1.0, t))

        sf_min = int(self.cfg.update_sf_min)
        sf_max = int(self.cfg.update_sf_max)

        # Safety clamps
        sf_min = max(1, sf_min)
        if sf_max < sf_min:
            sf_max = sf_min

        # Linear interpolation + clamp
        sf = sf_min + t * (sf_max - sf_min)
        sf_int = int(round(sf))
        sf_int = max(1, sf_int)
        return sf_int

    # -------------------------------------------------------------------------
    # Main API
    # -------------------------------------------------------------------------
    def apply(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply ViBe to one grayscale frame.

        Returns:
          fg_mask: uint8 mask (255=foreground, 0=background)

        Steps:
          1) (Optional) compute dynamic min_matches via AT
          2) Count sample matches within `radius` for each pixel
          3) Classify pixels as background if count >= min_matches
          4) (Optional) compute motion intensity and dynamic update rate via ALR
          5) Stochastically update model using background pixels only
        """
        if self.samples is None:
            # First call initializes the sample bank
            self.initialize(gray)

        assert self.samples is not None
        n, h, w = self.samples.shape

        # ---- (1) Choose threshold for background classification ----
        if bool(self.cfg.adaptive_min_matches):
            min_matches_used = self._compute_dynamic_min_matches(gray)
        else:
            min_matches_used = int(self.cfg.min_matches)
        self.last_min_matches_used = int(min_matches_used)

        # ---- (2) Compute number of matching samples per pixel ----
        # Using int16 avoids underflow/overflow for abs difference.
        diff = np.abs(self.samples.astype(np.int16) - gray.astype(np.int16)[None, :, :])
        matches = (diff < int(self.cfg.radius))
        count = matches.sum(axis=0)

        # ---- (3) Background / foreground decision ----
        is_bg = (count >= min_matches_used)
        fg_mask = (~is_bg).astype(np.uint8) * 255

        # Motion intensity: fraction of foreground pixels in the frame.
        # This is a cheap signal to estimate "how much motion is happening".
        fg_ratio = float((~is_bg).mean())  # in [0,1]

        # ---- (4) Choose learning/update rate ----
        if bool(self.cfg.adaptive_update):
            sf_used = self._compute_dynamic_update_sf(fg_ratio)
        else:
            sf_used = int(self.cfg.subsampling_factor)

        self.last_update_sf_used = int(sf_used)

        # ---- (5) Update background model (only on BG pixels) ----
        # Probabilistic update: for BG pixels, update with probability 1/sf_used.
        if sf_used > 0:
            update = is_bg & (self.rng.integers(0, sf_used, size=(h, w)) == 0)
        else:
            update = is_bg

        ys, xs = np.where(update)
        if ys.size > 0:
            # Replace a random sample with the current pixel value
            ridx = self.rng.integers(0, n, size=ys.size)
            self.samples[ridx, ys, xs] = gray[ys, xs]

            # Also update a random neighbor (classic ViBe: spatial diffusion of updates)
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
