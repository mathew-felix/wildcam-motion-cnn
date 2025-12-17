from __future__ import annotations

"""
ViBe + MHI utilities.

This module does NOT replace your ViBe implementation. Instead it adds:
  - MotionHistoryImage (MHI): short-term temporal memory of motion masks
  - ClipGate: a simple hysteresis gate that uses ViBe to START and MHI to CONTINUE

Why this design:
  - ViBe foreground masks can flicker (especially night/IR or subtle animal motion)
  - MHI smooths over short gaps by decaying motion history over a window (tau)
  - Using MHI to *start* recording often increases false triggers (wind/grass)
  - Best practice: START with a strict instantaneous signal, CONTINUE with a lenient temporal signal
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


# =============================================================================
# Motion History Image (MHI)
# =============================================================================

@dataclass
class MHIConfig:
    """
    Parameters for Motion History Image.

    tau_sec:
      Length of the temporal window (seconds). Motion remains "active" for ~tau_sec,
      then decays to zero if no new motion is observed.

    recent_thresh:
      When computing a "recent motion" score, we treat pixels with
      mhi >= recent_thresh * tau_frames as "recent".
      Example: recent_thresh=0.3 means last 30% of the window is considered recent.
    """
    enabled: bool = True
    tau_sec: float = 1.0
    recent_thresh: float = 0.3


class MotionHistoryImage:
    """
    Simple MHI implementation using a per-pixel countdown buffer.

    Internal representation:
      mhi[y,x] is an integer in [0, tau_frames]
        - set to tau_frames when motion occurs at (y,x)
        - decremented by 1 each frame otherwise
    """
    def __init__(self, cfg: MHIConfig, frame_shape: Tuple[int, int], fps: float):
        self.cfg = cfg
        self.h, self.w = int(frame_shape[0]), int(frame_shape[1])
        self.fps = float(max(1e-6, fps))

        # Convert seconds to frames
        self.tau_frames = max(1, int(round(float(cfg.tau_sec) * self.fps)))

        # int16 is enough for typical tau_frames (<= a few hundred)
        self.mhi = np.zeros((self.h, self.w), dtype=np.int16)

    def reset(self) -> None:
        """Clear motion history."""
        self.mhi.fill(0)

    def update(self, motion_mask: np.ndarray) -> None:
        """
        Update MHI with a binary motion mask.

        motion_mask:
          boolean or uint8 mask where True/Nonzero indicates motion pixels.
        """
        if not self.cfg.enabled:
            return

        # Decay history everywhere (one frame step)
        self.mhi -= 1
        np.maximum(self.mhi, 0, out=self.mhi)

        # Refresh pixels where motion is present
        mm = motion_mask.astype(bool)
        self.mhi[mm] = self.tau_frames

    def active_ratio(self) -> float:
        """Fraction of pixels with any motion history remaining (mhi > 0)."""
        if not self.cfg.enabled:
            return 0.0
        return float(np.count_nonzero(self.mhi > 0)) / float(max(1, self.h * self.w))

    def recent_ratio(self) -> float:
        """Fraction of pixels with *recent* motion (mhi close to tau_frames)."""
        if not self.cfg.enabled:
            return 0.0
        thr = int(round(self.cfg.recent_thresh * self.tau_frames))
        thr = max(1, min(thr, self.tau_frames))
        return float(np.count_nonzero(self.mhi >= thr)) / float(max(1, self.h * self.w))


# =============================================================================
# Clip-level gating (hysteresis)
# =============================================================================

@dataclass
class GateConfig:
    """
    Hysteresis gate configuration.

    start_metric / start_threshold / start_consecutive:
      When not recording, we require the start condition to be satisfied for
      `start_consecutive` frames to begin recording. This reduces single-frame FPs.

    stay_metric / stay_threshold:
      Once recording, we keep recording as long as the stay condition is met.
      This is where MHI shines: it smooths over brief motion dropouts.

    pre_roll_sec:
      If a start trigger occurs, we can retroactively mark the previous pre_roll_sec
      frames as recorded (more realistic clip behavior). This is optional.

    post_roll_sec:
      After the stay condition stops being met, continue recording for post_roll_sec
      frames before turning off (another realistic clip behavior).

    min_blob_area / max_blobs:
      Optional noise suppression that should be applied BEFORE MHI update/gating.
      Recommended to prevent wind/grass speckles from accumulating in MHI.
      (Filtering itself is implemented in the eval script because it uses cv2.)
    """
    enabled: bool = True

    # START logic (strict)
    start_metric: str = "vibe_fg_pixels"      # "vibe_fg_pixels" or "vibe_fg_ratio"
    start_threshold: float = 800.0           # pixels (if vibe_fg_pixels) or ratio (if vibe_fg_ratio)
    start_consecutive: int = 2

    # STAY logic (lenient)
    stay_metric: str = "mhi_active_ratio"    # "mhi_active_ratio" or "mhi_recent_ratio" or "vibe_fg_ratio"
    stay_threshold: float = 0.002            # ratio by default

    # Clip shaping
    pre_roll_sec: float = 0.0
    post_roll_sec: float = 1.0


class ClipGate:
    """
    Stateful hysteresis gate.

    Usage:
      gate = ClipGate(cfg, fps)
      recording, backfill = gate.step(scores)
      - recording: whether current frame should be kept
      - backfill: number of previous frames (including current) to set True on START (pre-roll)
    """
    def __init__(self, cfg: GateConfig, fps: float):
        self.cfg = cfg
        self.fps = float(max(1e-6, fps))

        self.pre_roll_frames = max(0, int(round(cfg.pre_roll_sec * self.fps)))
        self.post_roll_frames = max(0, int(round(cfg.post_roll_sec * self.fps)))

        self.recording = False
        self._start_hits = 0
        self._post_roll_left = 0

    @staticmethod
    def _get_metric(scores: Dict[str, float], name: str) -> float:
        if name not in scores:
            raise KeyError(f"Gate metric '{name}' missing from scores dict. Available: {list(scores.keys())}")
        return float(scores[name])

    def reset(self) -> None:
        self.recording = False
        self._start_hits = 0
        self._post_roll_left = 0

    def step(self, scores: Dict[str, float]) -> Tuple[bool, Optional[int]]:
        """
        Advance the gate by one frame.

        Returns:
          (recording_now, backfill_k)

          backfill_k:
            If we transitioned IDLE -> RECORDING this frame, backfill_k is the number of
            frames to mark as recorded including this one (pre-roll).
            Otherwise None.
        """
        if not self.cfg.enabled:
            return True, None  # degenerate: keep everything

        start_val = self._get_metric(scores, self.cfg.start_metric)
        stay_val = self._get_metric(scores, self.cfg.stay_metric)

        # -------------------------
        # IDLE -> maybe START
        # -------------------------
        if not self.recording:
            if start_val >= float(self.cfg.start_threshold):
                self._start_hits += 1
            else:
                self._start_hits = 0

            if self._start_hits >= max(1, int(self.cfg.start_consecutive)):
                # Start recording
                self.recording = True
                self._post_roll_left = self.post_roll_frames
                self._start_hits = 0

                # Backfill previous frames (pre-roll) + current
                backfill_k = self.pre_roll_frames + 1
                return True, backfill_k

            return False, None

        # -------------------------
        # RECORDING -> maybe STOP (with post-roll)
        # -------------------------
        if stay_val >= float(self.cfg.stay_threshold):
            # Motion still present: refresh post-roll
            self._post_roll_left = self.post_roll_frames
            return True, None

        # No stay signal: consume post-roll
        if self._post_roll_left > 0:
            self._post_roll_left -= 1
            return True, None

        # Post-roll finished: stop
        self.recording = False
        return False, None


def compute_scores(
    fg_mask_binary: np.ndarray,
    mhi: MotionHistoryImage,
) -> Dict[str, float]:
    """
    Compute common gating scores from:
      - fg_mask_binary (boolean motion mask after your postprocess)
      - MHI state

    Returns a dict with keys used by ClipGate.
    """
    mm = fg_mask_binary.astype(bool)
    fg_pixels = float(np.count_nonzero(mm))
    h, w = mm.shape[:2]
    fg_ratio = fg_pixels / float(max(1, h * w))

    return {
        "vibe_fg_pixels": fg_pixels,
        "vibe_fg_ratio": fg_ratio,
        "mhi_active_ratio": float(mhi.active_ratio()),
        "mhi_recent_ratio": float(mhi.recent_ratio()),
    }
