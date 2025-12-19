from __future__ import annotations

"""
Evaluate ViBe / ViBe+MHI / ViBe+CNN / ViBe+MHI+CNN variants and write TWO CSV outputs.

Outputs
-------
1) detection_log_csv (per-frame):
   - video_id, frame_id, ground_truth
   - vibe_mask_ratio: ViBe foreground ratio (proxy for motion amount)
   - mhi_energy: motion intensity proxy from MHI (or 0 if MHI disabled)
   - cnn_confidence: MobileNet probability (or 0 if CNN disabled)
   - final_decision: 1 if the pipeline keeps/saves this frame, else 0
   - inference_time_ms: end-to-end processing time for this frame
   - trigger_source: compact label explaining why the decision happened

2) benchmark_summary_csv (per-video):
   - video_id, total_frames
   - frames_processed: how many frames actually ran through the CNN (efficiency)
   - TP / FP / FN computed AFTER warmup_sec, in the [start_sec, end_sec] range
   - Precision, Recall, F1_Score, Avg_FPS
   - Storage_Saved_Pct = (1 - saved_frames/total_frames) * 100

Ablations (controlled by YAML)
------------------------------
- ViBe: toggle vibe.adaptive_min_matches (AT) and vibe.adaptive_update (ALR)
- MHI: toggle mhi.enabled
- CNN: toggle cnn.enabled
- Gate behavior: set gate.start_metric / thresholds and gate.stay_metric / thresholds

Run
---
python scripts/eval_vibe_mhi_cnn.py --config configs/ablations/vibe.yaml

Adaptive improvements (optional, controlled by YAML)
----------------------------------------------------
You asked for an adaptive mode that targets the real pain points:
  1) Run CNN only when motion is weak/ambiguous (where it helps most)
  2) Limit MHI influence (require motion-confirmed activation + faster effective decay)
  3) Reduce "FP snowballing" at night by using CNN-negative feedback to temporarily
     harden thresholds and speed up background adaptation.

This is implemented via the optional top-level YAML block:

  adaptive:
    enabled: true
    # motion ambiguity range for CNN scheduling
    cnn_motion_low_fg_ratio: 0.00025
    cnn_motion_high_fg_ratio: 0.01
    # MHI is only allowed to contribute after a "confirmed motion" arm
    mhi_requires_arm: true
    mhi_arm_fg_ratio: 0.002
    mhi_arm_hold_sec: 1.5
    # extra effective decay when no confirmed motion recently (acts like shorter tau)
    mhi_extra_decay_night_sec: 0.8
    # night/FP guard
    fp_guard:
      enabled: true
      night_luma_threshold: 70.0
      negative_p: 0.25
      negative_streak_to_activate: 3
      hold_sec: 2.0
      min_matches_boost: 1
      update_sf_div: 2.0

All old configs still work: if "adaptive.enabled" is false/missing, behavior is
identical to the previous evaluator.
"""

import argparse
import csv as csv_mod
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.motion.vibe import ViBe, ViBeConfig  # noqa: E402
from src.motion.vibe_mhi import (  # noqa: E402
    MHIConfig,
    MotionHistoryImage,
    compute_scores,
)
from src.motion.vibe_mhi_cnn import (  # noqa: E402
    CNNConfig,
    CNNRunner,
    PresenceConfig,
    PresenceLatch,
    HybridGateConfig,
    HybridClipGate,
    boxes_from_mask,
)


# -----------------------------------------------------------------------------
# Adaptive policy configs (optional)
# -----------------------------------------------------------------------------


@dataclass
class FPGuardConfig:
    """Settings to prevent FP "snowballing" in low-light / night clips.

    The core idea is simple:
      - detect low-light using mean luma (grayscale intensity)
      - if CNN repeatedly says "no animal" while motion is present,
        temporarily harden motion sensitivity and update background faster.
    """

    enabled: bool = False

    # Low-light detector
    night_luma_threshold: float = 70.0

    # CNN-negative feedback
    negative_p: float = 0.25
    negative_streak_to_activate: int = 3
    hold_sec: float = 2.0

    # How we "harden" ViBe while the guard is active
    #   - increase min_matches (more strict => fewer foreground pixels)
    #   - decrease subsampling_factor (faster background updates)
    min_matches_boost: int = 1
    update_sf_div: float = 2.0

    # While guard is active, downweight MHI contribution (reduces long tails)
    mhi_scale_when_active: float = 0.5


@dataclass
class AdaptiveConfig:
    """Top-level adaptive improvements requested by you."""

    enabled: bool = False

    # CNN scheduling: run CNN primarily when motion is weak/ambiguous
    cnn_motion_low_fg_ratio: float = 0.00025
    cnn_motion_high_fg_ratio: float = 0.01

    stride_rec_weak: int = 6
    stride_rec_nomotion: int = 8
    stride_idle_weak: int = 12

    # MHI influence: require a confirmed-motion "arm" before MHI can keep the clip alive
    mhi_requires_arm: bool = True
    mhi_arm_fg_ratio: float = 0.002
    mhi_arm_hold_sec: float = 1.5

    # Extra effective decay at night when confirmed motion disappears (acts like shorter tau)
    mhi_extra_decay_night_sec: float = 0.8

    # FP guard block
    # IMPORTANT (Python dataclasses): nested config objects must use default_factory
    # to avoid a shared mutable default across instances.
    fp_guard: FPGuardConfig = field(default_factory=FPGuardConfig)


def _mean_luma(gray_u8: np.ndarray) -> float:
    """Mean luma in [0,255] from a grayscale uint8 image."""
    return float(np.mean(gray_u8)) if gray_u8.size else 0.0


def _motion_state(fg_ratio: float, low: float, high: float) -> str:
    """Quantize motion strength using fg_ratio thresholds."""
    if fg_ratio < low:
        return "none"
    if fg_ratio < high:
        return "weak"
    return "strong"


def _safe_set_vibe_param(vibe: Any, name: str, value: Any) -> bool:
    """Best-effort runtime tweak of ViBe parameters (works for common implementations).

    We never assume a specific ViBe class layout.
    Returns True if we managed to update something.
    """
    if vibe is None:
        return False

    # Most common: vibe.cfg.<name>
    if hasattr(vibe, "cfg") and hasattr(vibe.cfg, name):
        try:
            setattr(vibe.cfg, name, value)
            return True
        except Exception:
            pass

    # Alternate: vibe.<name>
    if hasattr(vibe, name):
        try:
            setattr(vibe, name, value)
            return True
        except Exception:
            pass

    return False


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML not installed. Install with: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must load to a dict at top-level.")
    return cfg


def resolve_path(p: str, bases: List[Path]) -> str:
    if not p:
        return p
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return str(pp)
    if pp.exists():
        return str(pp)
    for b in bases:
        cand = (b / pp).resolve()
        if cand.exists():
            return str(cand)
    return str(pp)


def resolve_out_path(cfg_dir: Path, out_path: str) -> str:
    op = Path(out_path)
    if op.is_absolute():
        return str(op)
    return str((cfg_dir / op).resolve())


def maybe_tqdm(iterable, enabled: bool, total: Optional[int] = None, desc: str = ""):
    if not enabled:
        return iterable
    from tqdm import tqdm
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, file=sys.stdout)


# -----------------------------------------------------------------------------
# Ground-truth timeline
# -----------------------------------------------------------------------------

@dataclass
class EventTimeline:
    intervals: List[Tuple[int, int]]
    _ptr: int = 0

    @staticmethod
    def from_events(events: List[Dict[str, float]], fps: float) -> "EventTimeline":
        intervals: List[Tuple[int, int]] = []
        for e in events:
            s = float(e["start_sec"])
            t = float(e["end_sec"])
            start_f = int(math.floor(s * fps))
            end_f = int(math.ceil(t * fps)) - 1
            if end_f >= start_f:
                intervals.append((start_f, end_f))
        intervals.sort(key=lambda x: x[0])
        return EventTimeline(intervals=intervals)

    @staticmethod
    def from_frame_intervals(intervals: List[Tuple[int, int]]) -> "EventTimeline":
        """Build a timeline directly from frame-index intervals."""
        out = [(int(s), int(e)) for (s, e) in intervals if int(e) >= int(s)]
        out.sort(key=lambda x: x[0])
        return EventTimeline(intervals=out)

    def reset(self) -> None:
        self._ptr = 0

    def is_positive(self, frame_idx: int) -> bool:
        while self._ptr < len(self.intervals) and frame_idx > self.intervals[self._ptr][1]:
            self._ptr += 1
        if self._ptr >= len(self.intervals):
            return False
        s, e = self.intervals[self._ptr]
        return s <= frame_idx <= e


def _frames_to_intervals(frames: List[int]) -> List[Tuple[int, int]]:
    """Convert a list of positive frame indices into contiguous [start,end] intervals."""
    if not frames:
        return []
    fr = sorted({int(x) for x in frames})
    intervals: List[Tuple[int, int]] = []
    s = e = fr[0]
    for f in fr[1:]:
        if f == e + 1:
            e = f
        else:
            intervals.append((s, e))
            s = e = f
    intervals.append((s, e))
    return intervals


def _extract_gt_intervals(ann: Dict[str, Any], fps: float) -> List[Tuple[int, int]]:
    """Best-effort extraction of GT positive intervals from an annotation JSON.

    Preferred format (already used elsewhere in your project):
      {"events": [{"start_sec": ..., "end_sec": ...}, ...]}

    Fallbacks supported:
      - {"intervals": [[start_frame, end_frame], ...]} (or gt_intervals)
      - {"positive_frames": [f1, f2, ...]} (or gt_frames/pos_frames)
      - {"frames": {"123": <anything truthy>, ...}} (frame-index keyed dict)
      - {"frames": [{"frame_id": 123, ...}, ...]} (list with frame_id/frame_idx)
    """
    events = ann.get("events", None)
    if isinstance(events, list) and events:
        return EventTimeline.from_events(events, fps).intervals

    for k in ("intervals", "gt_intervals", "events_frames"):
        v = ann.get(k, None)
        if isinstance(v, list) and v:
            out: List[Tuple[int, int]] = []
            for it in v:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    out.append((int(it[0]), int(it[1])))
                elif isinstance(it, dict) and ("start" in it and "end" in it):
                    out.append((int(it["start"]), int(it["end"])))
            if out:
                out = [(s, e) for (s, e) in out if e >= s]
                out.sort(key=lambda x: x[0])
                return out

    for k in ("positive_frames", "pos_frames", "gt_frames", "frames_pos"):
        v = ann.get(k, None)
        if isinstance(v, list) and v:
            return _frames_to_intervals([int(x) for x in v])

    frames = ann.get("frames", None)
    if isinstance(frames, dict) and frames:
        pos = []
        for kk, vv in frames.items():
            try:
                fi = int(kk)
            except Exception:
                continue
            if vv is None:
                continue
            # treat empty containers as negative
            if isinstance(vv, (list, dict)) and len(vv) == 0:
                continue
            pos.append(fi)
        return _frames_to_intervals(pos)

    if isinstance(frames, list) and frames:
        pos = []
        for it in frames:
            if not isinstance(it, dict):
                continue
            fid = it.get("frame_id", it.get("frame_idx", it.get("frame", None)))
            if fid is None:
                continue
            try:
                pos.append(int(fid))
            except Exception:
                continue
        return _frames_to_intervals(pos)

    return []


def _segments_from_bool(mask: np.ndarray, start_frame_abs: int) -> List[Tuple[int, int]]:
    """Convert a boolean array into absolute-frame [start,end] segments where mask==True."""
    if mask.size == 0:
        return []
    segs: List[Tuple[int, int]] = []
    in_seg = False
    s = 0
    for i, v in enumerate(mask.tolist()):
        if v and not in_seg:
            in_seg = True
            s = i
        elif (not v) and in_seg:
            in_seg = False
            segs.append((start_frame_abs + s, start_frame_abs + i - 1))
    if in_seg:
        segs.append((start_frame_abs + s, start_frame_abs + int(mask.size) - 1))
    return segs


def _clip_interval(a: Tuple[int, int], lo: int, hi: int) -> Optional[Tuple[int, int]]:
    s, e = int(a[0]), int(a[1])
    s2 = max(s, int(lo))
    e2 = min(e, int(hi))
    if e2 < s2:
        return None
    return (s2, e2)


def compute_event_metrics(
    *,
    gt_intervals: List[Tuple[int, int]],
    pred_segments: List[Tuple[int, int]],
    score_start_abs: int,
    score_end_abs: int,
    fps: float,
) -> Tuple[float, float, float, float]:
    """Compute event-level metrics within the scored range.

    Returns:
      event_recall,
      animal_coverage,
      mean_trigger_delay_sec,
      mean_extra_tail_sec

    Notes:
    - Metrics are computed after warmup (using score_start_abs/score_end_abs).
    - Trigger delay / extra tail are averaged over DETECTED gt events.
    - Animal coverage is averaged over ALL gt events (missed events contribute 0).
    """
    if fps <= 0:
        fps = 30.0

    gt_clipped: List[Tuple[int, int]] = []
    for g in gt_intervals:
        c = _clip_interval(g, score_start_abs, score_end_abs)
        if c is not None:
            gt_clipped.append(c)

    if not gt_clipped:
        return 0.0, 0.0, 0.0, 0.0

    pred_clipped: List[Tuple[int, int]] = []
    for p in pred_segments:
        c = _clip_interval(p, score_start_abs, score_end_abs)
        if c is not None:
            pred_clipped.append(c)

    detected = 0
    cov_sum = 0.0
    delays: List[float] = []
    tails: List[float] = []

    for (gs, ge) in gt_clipped:
        best_ov = 0
        best_p: Optional[Tuple[int, int]] = None
        for (ps, pe) in pred_clipped:
            ov = max(0, min(ge, pe) - max(gs, ps) + 1)
            if ov > best_ov:
                best_ov = ov
                best_p = (ps, pe)

        gt_len = max(1, ge - gs + 1)
        cov = float(best_ov) / float(gt_len)
        cov_sum += cov

        if best_ov > 0 and best_p is not None:
            detected += 1
            ps, pe = best_p
            delay_frames = max(0, int(ps) - int(gs))
            tail_frames = max(0, int(pe) - int(ge))
            delays.append(float(delay_frames) / float(fps))
            tails.append(float(tail_frames) / float(fps))

    event_recall = float(detected) / float(len(gt_clipped)) if gt_clipped else 0.0
    animal_coverage = float(cov_sum) / float(len(gt_clipped)) if gt_clipped else 0.0
    mean_delay = float(np.mean(delays)) if delays else 0.0
    mean_tail = float(np.mean(tails)) if tails else 0.0
    return event_recall, animal_coverage, mean_delay, mean_tail
# -----------------------------------------------------------------------------
# Pre/post processing
# -----------------------------------------------------------------------------

def resize_keep_aspect(frame_bgr: np.ndarray, resize_width: Optional[int]) -> np.ndarray:
    if resize_width is None or int(resize_width) <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    rw = int(resize_width)
    if w == rw:
        return frame_bgr
    scale = rw / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (rw, nh), interpolation=cv2.INTER_AREA)


def preprocess_pair(frame_bgr: np.ndarray, resize_width: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    frame_rs = resize_keep_aspect(frame_bgr, resize_width)
    gray = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2GRAY)
    return frame_rs, gray


def postprocess_mask(mask: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    out = mask
    k = int(cfg.get("median_blur_ksize", 0) or 0)
    if k >= 3 and (k % 2 == 1):
        out = cv2.medianBlur(out, k)

    er = int(cfg.get("erode_iters", 0) or 0)
    di = int(cfg.get("dilate_iters", 0) or 0)
    if er or di:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if er:
            out = cv2.erode(out, kernel, iterations=er)
        if di:
            out = cv2.dilate(out, kernel, iterations=di)
    return out


def filter_components(mask: np.ndarray, min_area: int, max_blobs: int) -> np.ndarray:
    if (min_area is None or int(min_area) <= 0) and (max_blobs is None or int(max_blobs) <= 0):
        return mask

    mm = (mask > 0).astype(np.uint8)
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mm, connectivity=8)

    comps = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if min_area and area < int(min_area):
            continue
        comps.append((area, lab))

    if not comps:
        return np.zeros_like(mask)

    comps.sort(key=lambda t: t[0], reverse=True)
    if max_blobs and int(max_blobs) > 0:
        comps = comps[: int(max_blobs)]

    keep_labels = set(lab for _area, lab in comps)
    out = np.zeros_like(mm)
    for lab in keep_labels:
        out[labels == lab] = 1

    return (out * 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

DETECTION_LOG_COLUMNS = [
    "video_id",
    "frame_id",
    "ground_truth",
    "vibe_mask_ratio",
    "mhi_energy",
    "cnn_confidence",
    "final_decision",
    "inference_time_ms",
    "trigger_source",
]

BENCHMARK_SUMMARY_COLUMNS = [
    "video_id",
    "total_frames",
    "frames_processed",
    "TP",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1_Score",
    "Avg_FPS",
    "Storage_Saved_Pct",
    # Event-level metrics (computed from GT/predicted timelines)
    "Event_Recall",
    "Animal_Coverage",
    "Mean_Trigger_Delay_Sec",
    "Mean_Extra_Tail_Sec",
]


def get_mhi_energy(scores: Dict[str, Any], mhi: Optional[MotionHistoryImage]) -> float:
    """
    Best-effort extraction of an "MHI energy" scalar.
    We prefer an explicit key if compute_scores provides it, otherwise fall back
    to mhi_active_ratio (a stable proxy for motion intensity over time).
    """
    for k in ("mhi_energy", "mhi_intensity", "mhi_mean", "mhi_sum"):
        if k in scores:
            try:
                return float(scores[k])
            except Exception:
                pass

    # Safe proxy
    if "mhi_active_ratio" in scores:
        try:
            return float(scores["mhi_active_ratio"])
        except Exception:
            pass

    # Last resort: try to compute a mean over an internal buffer if exposed
    if mhi is not None:
        for attr in ("mhi", "buf", "history", "img"):
            if hasattr(mhi, attr):
                arr = getattr(mhi, attr)
                try:
                    arr = np.asarray(arr)
                    if arr.size:
                        return float(np.mean(arr))
                except Exception:
                    pass

    return 0.0


def classify_trigger_source(
    *,
    final_decision: bool,
    gate_reason: str,
    motion_present: bool,
    ran_cnn: bool,
    cnn_conf: float,
    presence_score: float,
    mhi_active_ratio: float,
    p_set: float,
) -> str:
    """
    Produce a compact categorical label for ablations / plots.
    Examples requested by you: "ViBe_only", "MHI_reject", "CNN_confirm".
    """
    r = str(gate_reason or "").strip().lower()

    if final_decision:
        if "start_cnn" in r or "cnn" in r and "start" in r:
            return "CNN_confirm"
        if "start_motion" in r or "motion" in r and "start" in r:
            return "ViBe_only"
        if r == "stay":
            if presence_score > 0 and presence_score >= mhi_active_ratio:
                return "CNN_confirm"
            return "MHI_stay"
        if r == "post_roll":
            return "post_roll"
        return "recording"

    # Not recording
    if not motion_present:
        return "ViBe_only"

    # Motion present but gate still rejects
    if ran_cnn:
        return "CNN_reject" if float(cnn_conf) < float(p_set) else "MHI_reject"

    return "MHI_reject"


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------

def eval_one_video(
    video_id: str,
    video_path: str,
    annotation_path: str,
    manifest_fps: Optional[float],
    cfg: Dict[str, Any],
    cnn: Optional[CNNRunner],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[{video_id}] video not found: {video_path}")
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"[{video_id}] annotation not found: {annotation_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[{video_id}] failed to open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 1e-6:
        video_fps = float(manifest_fps or 30.0)

    # Ground-truth intervals (best-effort; supports several annotation schemas)
    ann = load_json(annotation_path)
    gt_intervals = _extract_gt_intervals(ann, fps=video_fps)
    timeline = EventTimeline.from_frame_intervals(gt_intervals)
    timeline.reset()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    eval_cfg = cfg["eval"]
    start_sec = float(eval_cfg.get("start_sec", 0.0) or 0.0)
    end_sec = eval_cfg.get("end_sec", None)
    warmup_sec = float(eval_cfg.get("warmup_sec", 0.0) or 0.0)

    start_frame = max(0, int(round(start_sec * video_fps)))
    end_frame = (total_frames - 1) if end_sec is None else min(total_frames - 1, int(round(float(end_sec) * video_fps)))
    if total_frames <= 0:
        end_frame = max(end_frame, start_frame)

    warmup_frames = int(round(warmup_sec * video_fps))
    score_start = min(end_frame + 1, start_frame + warmup_frames)

    # timeline already initialized above

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pre = cfg["preprocess"]
    post = cfg["postprocess"]
    ui = cfg["ui"]

    use_tqdm = bool(ui.get("use_tqdm", False)) and (not bool(ui.get("show_debug", False)))

    vibe_cfg = ViBeConfig(**cfg["vibe"])
    vibe = ViBe(vibe_cfg, rng_seed=int(cfg.get("rng_seed", 0) or 0))

    # Snapshot a few ViBe knobs so adaptive FP-guard can temporarily tweak and then restore.
    # (This keeps your ablations fair: unless fp_guard activates, ViBe behaves exactly as configured.)
    base_min_matches = getattr(getattr(vibe, "cfg", vibe), "min_matches", None)
    base_subsampling_factor = getattr(getattr(vibe, "cfg", vibe), "subsampling_factor", None)

    mhi_cfg = MHIConfig(**cfg.get("mhi", {}))
    mhi_enabled = bool(getattr(mhi_cfg, "enabled", True))
    mhi: Optional[MotionHistoryImage] = None

    gate_cfg = HybridGateConfig(**cfg.get("gate", {}))
    gate = HybridClipGate(gate_cfg, fps=video_fps)

    comp_cfg = cfg.get("components", {})
    min_blob_area = int(comp_cfg.get("min_blob_area", 0) or 0)
    max_blobs = int(comp_cfg.get("max_blobs", 0) or 0)

    cnn_cfg = CNNConfig(**cfg.get("cnn", {}))
    p_set = float(getattr(cnn_cfg, "p_set", 0.5))

    # Optional adaptive mode (all keys are optional; missing -> defaults)
    ad_raw = cfg.get("adaptive", {}) if isinstance(cfg.get("adaptive", {}), dict) else {}
    fp_raw = ad_raw.get("fp_guard", {}) if isinstance(ad_raw.get("fp_guard", {}), dict) else {}
    adaptive_cfg = AdaptiveConfig(
        **{k: v for k, v in ad_raw.items() if k != "fp_guard"},
        fp_guard=FPGuardConfig(**fp_raw),
    )

    presence_cfg = PresenceConfig(
        enabled=bool(cnn_cfg.enabled),
        tau_sec=float(cnn_cfg.presence_tau_sec),
        p_set=float(cnn_cfg.p_set),
        p_start_high=float(cnn_cfg.p_start_high),
        refresh_requires_motion=bool(cnn_cfg.refresh_requires_motion),
    )
    presence = PresenceLatch(presence_cfg, fps=video_fps)

    stride_rec = max(1, int(cnn_cfg.stride_recording))
    stride_idle_motion = max(1, int(cnn_cfg.stride_idle_motion))
    idle_full_stride_frames = 0
    if float(cnn_cfg.idle_fullframe_stride_sec) > 0:
        idle_full_stride_frames = max(1, int(round(float(cnn_cfg.idle_fullframe_stride_sec) * video_fps)))

    sum_cnn_p = 0.0
    cnn_runs = 0

    n_range = max(0, end_frame - start_frame + 1)
    pred_all = np.zeros((n_range,), dtype=bool)
    gt_all = np.zeros((n_range,), dtype=bool)

    detection_rows: List[Dict[str, Any]] = []

    processed = 0
    t_total0 = time.perf_counter()

    last_p_frame = 0.0

    frame_iter = maybe_tqdm(
        range(start_frame, end_frame + 1),
        enabled=use_tqdm,
        total=n_range,
        desc=f"Frames: {video_id}",
    )

    # ---------------------------------------------------------------------
    # Adaptive state (per-video)
    # ---------------------------------------------------------------------
    mhi_armed = False
    mhi_arm_left = 0
    frames_since_confirmed_motion = 1_000_000

    fp_guard_left = 0
    fp_guard_active = False  # derived from fp_guard_left each frame
    neg_streak = 0

    mhi_arm_hold_frames = max(1, int(round(float(adaptive_cfg.mhi_arm_hold_sec) * video_fps)))
    fp_hold_frames = max(1, int(round(float(adaptive_cfg.fp_guard.hold_sec) * video_fps)))

    for i, frame_idx in enumerate(frame_iter):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        t_frame0 = time.perf_counter()

        # Derive active FP-guard state from remaining hold frames (safe: always defined).
        fp_guard_active = bool(fp_guard_left > 0)

        frame_rs, gray = preprocess_pair(frame_bgr, resize_width=pre.get("resize_width", None))

        if mhi is None:
            mhi = MotionHistoryImage(mhi_cfg, frame_shape=gray.shape[:2], fps=video_fps)

        fgmask = vibe.apply(gray)
        fgmask = postprocess_mask(fgmask, post)

        crop_mask_01 = (fgmask > 0).astype(np.uint8)

        fgmask_filt = filter_components(fgmask, min_area=min_blob_area, max_blobs=max_blobs)
        motion_binary = (fgmask_filt > 0)

        gt = bool(timeline.is_positive(frame_idx))
        gt_all[i] = gt

        # --- cheap scene context ---
        # mean luma (night detector) and raw fg ratio (motion strength)
        mean_luma = _mean_luma(gray)
        is_night = bool(adaptive_cfg.enabled and adaptive_cfg.fp_guard.enabled and (mean_luma < float(adaptive_cfg.fp_guard.night_luma_threshold)))

        fg_ratio_raw = float(np.mean(motion_binary)) if motion_binary.size else 0.0
        motion_state = _motion_state(
            fg_ratio_raw,
            low=float(adaptive_cfg.cnn_motion_low_fg_ratio),
            high=float(adaptive_cfg.cnn_motion_high_fg_ratio),
        )

        # --- MHI update (adaptive: require confirmed motion to "arm" MHI) ---
        confirmed_motion = fg_ratio_raw >= float(adaptive_cfg.mhi_arm_fg_ratio)
        if confirmed_motion:
            mhi_armed = True
            mhi_arm_left = int(mhi_arm_hold_frames)
            frames_since_confirmed_motion = 0
        else:
            frames_since_confirmed_motion = int(frames_since_confirmed_motion) + 1
            if mhi_arm_left > 0:
                mhi_arm_left -= 1
            # If we're not currently recording and the arm timer expires, disarm.
            if (mhi_arm_left <= 0) and (not gate.recording):
                mhi_armed = False

        # Always call update() so the internal MHI decays over time.
        # When not armed, we update with an all-zero mask to avoid reinforcing noise.
        mhi_update_mask = motion_binary
        if bool(adaptive_cfg.enabled) and bool(mhi_enabled) and bool(adaptive_cfg.mhi_requires_arm):
            if not (mhi_armed or confirmed_motion):
                mhi_update_mask = np.zeros_like(motion_binary, dtype=bool)

        mhi.update(mhi_update_mask)
        scores = compute_scores(motion_binary, mhi)

        # motion_present based on vibe_fg_ratio (cheap)
        fg_ratio = float(scores.get("vibe_fg_ratio", fg_ratio_raw))
        motion_present = fg_ratio >= float(cnn_cfg.motion_min_fg_ratio)

        # FP-guard state machine: if active, we temporarily harden the system in night clips.
        fp_guard_active = bool(fp_guard_left > 0)
        if fp_guard_active:
            fp_guard_left -= 1
            if fp_guard_left <= 0:
                # Restore original ViBe settings
                if base_min_matches is not None:
                    _safe_set_vibe_param(vibe, "min_matches", int(base_min_matches))
                if base_subsampling_factor is not None:
                    _safe_set_vibe_param(vibe, "subsampling_factor", int(base_subsampling_factor))

        # ---- CNN scheduling (FP-safe) ----
        p_frame = float(last_p_frame)
        ran_cnn = False
        full_frame = False

        if bool(cnn_cfg.enabled) and (cnn is not None):
            run_cnn = False

            if bool(adaptive_cfg.enabled):
                # Research-style adaptive policy:
                #   - STRONG motion => trust motion (skip CNN)
                #   - WEAK motion   => run CNN (where it helps most)
                #   - NO motion     => only run CNN while recording (to keep stationary animals)
                if gate.recording:
                    if motion_state == "weak":
                        run_cnn = (i % max(1, int(adaptive_cfg.stride_rec_weak))) == 0
                    elif motion_state == "none":
                        run_cnn = (i % max(1, int(adaptive_cfg.stride_rec_nomotion))) == 0
                    else:
                        run_cnn = False
                else:
                    if motion_state == "weak":
                        run_cnn = (i % max(1, int(adaptive_cfg.stride_idle_weak))) == 0
                    else:
                        run_cnn = False

                # In adaptive mode we do NOT do idle full-frame spot-checks by default
                # because they can create FP explosions and distort ablations.
                full_frame = False

            else:
                # Legacy scheduling (pre-adaptive): fixed strides + optional idle full-frame spot-check
                if gate.recording:
                    run_cnn = (i % stride_rec) == 0
                else:
                    if motion_present:
                        run_cnn = (i % stride_idle_motion) == 0
                    else:
                        # idle full-frame spot-check (OFF by default)
                        if idle_full_stride_frames > 0:
                            run_cnn = (i % idle_full_stride_frames) == 0
                            full_frame = True

            # Optional: block CNN-start when no motion (extra safety)
            if (not gate.recording) and bool(cnn_cfg.start_requires_motion) and (not motion_present):
                # Still allow full-frame spot-check if explicitly enabled
                if not full_frame:
                    run_cnn = False

            if run_cnn:
                ran_cnn = True
                if full_frame:
                    boxes = [(0, 0, frame_rs.shape[1], frame_rs.shape[0])]
                    p_frame = cnn.predict_frame_prob(frame_rs, boxes, fallback_full_frame=True)
                else:
                    boxes = boxes_from_mask(
                        crop_mask_01,
                        frame_shape_hw=frame_rs.shape[:2],
                        min_area=int(cnn_cfg.min_crop_area),
                        max_boxes=int(cnn_cfg.max_crops),
                        expand_ratio=float(cnn_cfg.expand_ratio),
                        pad_px=int(cnn_cfg.pad_px),
                    )
                    p_frame = cnn.predict_frame_prob(
                        frame_rs,
                        boxes_xyxy=boxes,
                        fallback_full_frame=bool(cnn_cfg.full_frame_if_no_crops),
                    )

                last_p_frame = float(p_frame)
                cnn_runs += 1
                sum_cnn_p += float(p_frame)

        # ---- Night FP-guard: CNN-negative feedback => temporary hardening ----
        if bool(adaptive_cfg.enabled) and bool(adaptive_cfg.fp_guard.enabled) and bool(is_night):
            # Only consider activating the guard when we *actually* ran the CNN and there is motion.
            # (This avoids biasing metrics by unconditionally calling CNN at night.)
            if bool(ran_cnn) and bool(motion_present) and (motion_state != "none"):
                if float(p_frame) < float(adaptive_cfg.fp_guard.negative_p):
                    neg_streak += 1
                elif float(p_frame) >= float(p_set):
                    neg_streak = 0
                else:
                    neg_streak = max(0, int(neg_streak) - 1)

                if (not fp_guard_active) and (int(neg_streak) >= int(adaptive_cfg.fp_guard.negative_streak_to_activate)):
                    # Activate for a short window
                    fp_guard_left = int(fp_hold_frames)
                    fp_guard_active = True
                    neg_streak = 0

                    # AT/ALR tuning: stricter match threshold + faster background learning
                    if base_min_matches is not None:
                        boosted = max(1, int(base_min_matches) + int(adaptive_cfg.fp_guard.min_matches_boost))
                        _safe_set_vibe_param(vibe, "min_matches", boosted)
                    if base_subsampling_factor is not None:
                        div = float(adaptive_cfg.fp_guard.update_sf_div)
                        if not (div > 0):
                            div = 1.0
                        new_sf = max(1, int(round(float(base_subsampling_factor) / div)))
                        _safe_set_vibe_param(vibe, "subsampling_factor", new_sf)

        # Presence latch refresh policy: if FP-guard is active, do not refresh on motion.
        # This helps the latch decay quickly during noise bursts.
        motion_present_for_latch = bool(motion_present) and (not bool(fp_guard_active))
        presence_score = float(presence.step(float(p_frame), motion_present=bool(motion_present_for_latch)))

        # STAY score combines (effective) MHI + presence latch.
        # Adaptive rules:
        #   - MHI contributes only after a confirmed-motion "arm" (prevents night noise tails)
        #   - additional effective decay at night when confirmed motion disappears
        mhi_active_ratio_raw = float(scores.get("mhi_active_ratio", 0.0)) if bool(mhi_enabled) else 0.0
        mhi_active_ratio_eff = float(mhi_active_ratio_raw)

        if bool(adaptive_cfg.enabled) and bool(mhi_enabled) and bool(adaptive_cfg.mhi_requires_arm):
            if not (bool(mhi_armed) or bool(confirmed_motion)):
                mhi_active_ratio_eff = 0.0

        if (
            bool(adaptive_cfg.enabled)
            and bool(is_night)
            and float(adaptive_cfg.mhi_extra_decay_night_sec) > 0
            and int(frames_since_confirmed_motion) > 0
        ):
            decay_frames = max(1.0, float(adaptive_cfg.mhi_extra_decay_night_sec) * float(video_fps))
            mhi_active_ratio_eff = float(mhi_active_ratio_eff) * float(math.exp(-float(frames_since_confirmed_motion) / decay_frames))

        if bool(fp_guard_active):
            mhi_active_ratio_eff = float(mhi_active_ratio_eff) * float(adaptive_cfg.fp_guard.mhi_scale_when_active)

        stay_score = max(float(mhi_active_ratio_eff), float(presence_score))

        # Expose to gate
        scores["cnn_p_frame"] = float(p_frame)
        scores["cnn_presence_score"] = float(presence_score)
        scores["stay_score"] = float(stay_score)

        # If FP-guard is active, harden both START and STAY by downweighting motion metrics.
        # This is a lightweight way to "raise thresholds" without mutating gate configuration.
        if bool(fp_guard_active):
            if "vibe_fg_pixels" in scores:
                scores["vibe_fg_pixels"] = float(scores["vibe_fg_pixels"]) * 0.6
            if "vibe_fg_ratio" in scores:
                scores["vibe_fg_ratio"] = float(scores["vibe_fg_ratio"]) * 0.6
            scores["stay_score"] = float(scores.get("stay_score", 0.0)) * 0.8

        recording, backfill_k = gate.step(scores, cnn_p_frame=float(p_frame))
        pred_all[i] = bool(recording)

        # Backfill pre-roll frames (keep detection log consistent too)
        if backfill_k is not None and backfill_k > 1:
            j0 = max(0, i - (backfill_k - 1))
            pred_all[j0 : i + 1] = True
            # Update already-written rows
            for jj in range(j0, i):
                if jj < len(detection_rows):
                    detection_rows[jj]["final_decision"] = 1
                    detection_rows[jj]["trigger_source"] = "pre_roll_backfill"

        # Trigger label
        gate_reason = getattr(gate, "last_reason", "")
        trigger_source = classify_trigger_source(
            final_decision=bool(recording),
            gate_reason=str(gate_reason),
            motion_present=bool(motion_present),
            ran_cnn=bool(ran_cnn),
            cnn_conf=float(p_frame),
            presence_score=float(presence_score),
            mhi_active_ratio=float(mhi_active_ratio_eff),
            p_set=float(p_set),
        )

        # Prefix for easier slicing in plots
        if bool(fp_guard_active):
            trigger_source = f"fp_guard:{trigger_source}"
        elif bool(is_night):
            trigger_source = f"night:{trigger_source}"

        inference_time_ms = float((time.perf_counter() - t_frame0) * 1000.0)

        det_row = {
            "video_id": str(video_id),
            "frame_id": int(frame_idx),
            "ground_truth": int(gt),
            "vibe_mask_ratio": float(fg_ratio),
            # We log the *effective* MHI energy (after adaptive gating), since that's what drives decisions.
            "mhi_energy": float(mhi_active_ratio_eff),
            "cnn_confidence": float(p_frame),
            "final_decision": int(bool(recording)),
            "inference_time_ms": float(inference_time_ms),
            "trigger_source": str(trigger_source),
        }
        detection_rows.append(det_row)

        if bool(ui.get("show_debug", False)):
            vis_mask = cv2.cvtColor(fgmask_filt.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            stacked = np.hstack([frame_rs, vis_mask])
            cv2.putText(
                stacked,
                f"rec={int(recording)} p={p_frame:.2f} pres={presence_score:.2f} fg={fg_ratio:.4f} reason={gate_reason}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(f"ViBe+MHI+CNN Debug: {video_id}", stacked)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        processed += 1

    cap.release()
    if bool(ui.get("show_debug", False)):
        cv2.destroyAllWindows()

    pred_all = pred_all[:processed]
    gt_all = gt_all[:processed]
    detection_rows = detection_rows[:processed]

    elapsed = max(1e-12, time.perf_counter() - t_total0)
    avg_fps = processed / elapsed if processed > 0 else 0.0

    # Score region (skip warmup)
    score_i0 = max(0, int(score_start - start_frame))
    pred_sc = pred_all[score_i0:]
    gt_sc = gt_all[score_i0:]

    tp = int(np.sum(pred_sc & gt_sc))
    fp = int(np.sum(pred_sc & (~gt_sc)))
    fn = int(np.sum((~pred_sc) & gt_sc))

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    frames_saved = int(np.sum(pred_all))
    denom_total = float(max(1, total_frames if total_frames > 0 else processed))
    storage_saved_pct = (1.0 - (frames_saved / denom_total)) * 100.0

    # -----------------------------------------------------------------
    # Event-level metrics (scored range)
    # -----------------------------------------------------------------
    abs_end_frame = int(start_frame + processed - 1)
    score_start_abs = int(start_frame + score_i0)
    score_end_abs = int(abs_end_frame)

    pred_segments = _segments_from_bool(pred_all, start_frame_abs=int(start_frame))
    event_recall, animal_cov, mean_delay_sec, mean_tail_sec = compute_event_metrics(
        gt_intervals=list(timeline.intervals),
        pred_segments=pred_segments,
        score_start_abs=score_start_abs,
        score_end_abs=score_end_abs,
        fps=float(video_fps),
    )

    summary_row = {
        "video_id": str(video_id),
        "total_frames": int(total_frames if total_frames > 0 else processed),
        "frames_processed": int(cnn_runs),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_Score": float(f1),
        "Avg_FPS": float(avg_fps),
        "Storage_Saved_Pct": float(storage_saved_pct),
        "Event_Recall": float(event_recall),
        "Animal_Coverage": float(animal_cov),
        "Mean_Trigger_Delay_Sec": float(mean_delay_sec),
        "Mean_Extra_Tail_Sec": float(mean_tail_sec),
    }

    return summary_row, detection_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/eval_vibe_mhi_cnn_fp_safe.yaml",
        help="Path to YAML config.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent

    cfg = load_yaml(str(cfg_path))

    manifest_path = resolve_path(cfg["manifest_path"], bases=[cfg_dir, ROOT])

    # Output paths (backward compatible with older configs)
    summary_csv = cfg.get("benchmark_summary_csv", None) or cfg.get("output_csv", "benchmark_summary.csv")
    summary_csv = resolve_out_path(cfg_dir, str(summary_csv))

    detection_csv = cfg.get("detection_log_csv", None)
    if detection_csv:
        detection_csv = resolve_out_path(cfg_dir, str(detection_csv))
    else:
        sp = Path(summary_csv)
        detection_csv = str((sp.parent / f"{sp.stem}_detection_log.csv").resolve())

    manifest = load_json(manifest_path)
    items = manifest["items"]

    only_ids = cfg.get("only_ids", None)
    only_ids = set(only_ids) if only_ids else None

    ui = cfg["ui"]
    use_tqdm_videos = bool(ui.get("use_tqdm", False)) and (not bool(ui.get("show_debug", False)))

    cnn_cfg = CNNConfig(**cfg.get("cnn", {}))
    cnn = CNNRunner(cnn_cfg) if bool(cnn_cfg.enabled) else None

    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(detection_csv) or ".", exist_ok=True)

    # Prepare writers
    with open(summary_csv, "w", newline="", encoding="utf-8") as f_sum, open(
        detection_csv, "w", newline="", encoding="utf-8"
    ) as f_det:
        w_sum = csv_mod.DictWriter(f_sum, fieldnames=BENCHMARK_SUMMARY_COLUMNS, extrasaction="ignore")
        w_det = csv_mod.DictWriter(f_det, fieldnames=DETECTION_LOG_COLUMNS, extrasaction="ignore")
        w_sum.writeheader()
        w_det.writeheader()

        video_iter = maybe_tqdm(items, enabled=use_tqdm_videos, total=len(items), desc="Videos")
        manifest_dir = Path(manifest_path).resolve().parent

        wrote_any = False

        for it in video_iter:
            vid = it["id"]
            if only_ids and vid not in only_ids:
                continue

            video_path = resolve_path(it["video_path"], bases=[manifest_dir, cfg_dir, ROOT])
            ann_path = resolve_path(it["annotation_path"], bases=[manifest_dir, cfg_dir, ROOT])

            summary_row, det_rows = eval_one_video(
                video_id=vid,
                video_path=video_path,
                annotation_path=ann_path,
                manifest_fps=it.get("fps", None),
                cfg=cfg,
                cnn=cnn,
            )

            w_sum.writerow(summary_row)
            for r in det_rows:
                w_det.writerow(r)

            wrote_any = True

        if not wrote_any:
            print("No videos evaluated (check only_ids / manifest).")
            return

    print(f"Wrote benchmark summary: {summary_csv}")
    print(f"Wrote detection log:     {detection_csv}")


if __name__ == "__main__":
    main()
