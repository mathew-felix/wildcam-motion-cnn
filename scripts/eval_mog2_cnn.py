from __future__ import annotations

"""
Evaluate MOG2 / MOG2+CNN variants and write TWO CSV outputs with the SAME schema
as `eval_vibe_mhi_cnn.py` so that plotting + benchmarking stays consistent.

Why this file exists
--------------------
Your ViBe evaluator already reports *event-level* behavior-preservation metrics:
  - Event_Recall
  - Animal_Coverage
  - Mean_Trigger_Delay_Sec
  - Mean_Extra_Tail_Sec

Your MOG2(+CNN) evaluator previously wrote only frame-level TP/FP/FN + FPS + storage,
so your aggregate `benchmark_summary_all.csv` had missing columns for the MOG2
baselines. This updated evaluator adds the missing metrics and standardizes the
CSV column names.

Outputs
-------
1) detection_log_csv (per-frame):
   - video_id, frame_id, ground_truth
   - vibe_mask_ratio      (for MOG2: we log mog2_fg_ratio here to match schema)
   - mhi_energy           (for MOG2: we log stay_score here as the "temporal keep" signal)
   - cnn_confidence
   - final_decision
   - inference_time_ms
   - trigger_source       (compact label explaining why decision happened)

2) benchmark_summary_csv (per-video):
   - video_id, total_frames
   - frames_processed     (# of CNN runs, 0 for pure MOG2)
   - TP / FP / FN (computed AFTER warmup_sec, within [start_sec, end_sec])
   - Precision, Recall, F1_Score, Avg_FPS, Storage_Saved_Pct
   - Event_Recall, Animal_Coverage, Mean_Trigger_Delay_Sec, Mean_Extra_Tail_Sec

Run
---
python scripts/eval_mog2_cnn.py --config configs/ablations/mog2_cnn.yaml

Backwards-compatible config keys
--------------------------------
This script supports BOTH:
  - New style (recommended): benchmark_summary_csv + detection_log_csv
  - Old style: output_dir + run_name (+ optional output_csv)

If neither benchmark_summary_csv nor output_csv is set, we write to:
  <output_dir>/<run_name>_benchmark_summary_<timestamp>.csv
and derive detection_log_csv accordingly.
"""

import argparse
import csv as csv_mod
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.motion.mog2_cnn import (  # noqa: E402
    CNNConfig,
    CNNRunner,
    HybridClipGate,
    HybridGateConfig,
    MOG2Config,
    MOG2Motion,
    PresenceConfig,
    PresenceLatch,
    boxes_from_mask,
)


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
    for k in bases:
        cand = (k / pp).resolve()
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
# Ground-truth timeline + event metrics (copied/aligned with eval_vibe_mhi_cnn)
# -----------------------------------------------------------------------------

@dataclass
class EventTimeline:
    intervals: List[Tuple[int, int]]
    _ptr: int = 0

    @staticmethod
    def from_events(events: List[Dict[str, float]], fps: float) -> "EventTimeline":
        import math
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
    """Best-effort GT extraction (supports several annotation schemas)."""
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
            out = [(s, e) for (s, e) in out if e >= s]
            out.sort(key=lambda x: x[0])
            if out:
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
    """Return (Event_Recall, Animal_Coverage, Mean_Trigger_Delay_Sec, Mean_Extra_Tail_Sec)."""
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
# Logging schema (matches eval_vibe_mhi_cnn)
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
    "Event_Recall",
    "Animal_Coverage",
    "Mean_Trigger_Delay_Sec",
    "Mean_Extra_Tail_Sec",
]


def _trigger_source_from_reason(
    *,
    recording: bool,
    reason: str,
    motion_present: bool,
    ran_cnn: bool,
    cnn_p: float,
    p_set: float,
) -> str:
    r = (reason or "").strip().lower()

    if recording:
        if "start_cnn" in r or ("cnn" in r and "start" in r):
            return "CNN_confirm"
        if "start_motion" in r or ("motion" in r and "start" in r):
            return "MOG2_motion"
        if r == "stay":
            return "stay"
        if "post" in r:
            return "post_roll"
        return "recording"

    # Not recording
    if not motion_present:
        return "idle"
    if ran_cnn:
        return "CNN_reject" if float(cnn_p) < float(p_set) else "reject"
    return "reject"


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------

def eval_one_video(
    *,
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

    ann = load_json(annotation_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[{video_id}] failed to open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 1e-6:
        video_fps = float(manifest_fps or 30.0)

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

    # GT intervals (robust extraction)
    gt_intervals = _extract_gt_intervals(ann, fps=float(video_fps))
    timeline = EventTimeline.from_frame_intervals(gt_intervals)
    timeline.reset()

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pre = cfg["preprocess"]
    post = cfg["postprocess"]
    ui = cfg["ui"]

    use_tqdm = bool(ui.get("use_tqdm", False)) and (not bool(ui.get("show_debug", False)))

    # MOG2
    mog2_cfg = MOG2Config(**cfg.get("mog2", {}))
    mog2 = MOG2Motion(mog2_cfg)

    # Gate + presence latch
    gate_cfg = HybridGateConfig(**cfg.get("gate", {}))
    gate = HybridClipGate(gate_cfg, fps=video_fps)

    cnn_cfg = CNNConfig(**cfg.get("cnn", {}))
    p_set = float(getattr(cnn_cfg, "p_set", 0.5))

    presence_cfg = PresenceConfig(
        enabled=bool(cnn_cfg.enabled),
        tau_sec=float(cnn_cfg.presence_tau_sec),
        p_set=float(cnn_cfg.p_set),
        p_start_high=float(cnn_cfg.p_start_high),
        refresh_requires_motion=bool(cnn_cfg.refresh_requires_motion),
    )
    presence = PresenceLatch(presence_cfg, fps=video_fps)

    # Motion EMA used as an "energy" (stand-in for a temporal stay signal)
    motion_alpha = float(cfg.get("motion_ema_alpha", 0.2) or 0.2)
    motion_alpha = max(0.0, min(1.0, motion_alpha))
    motion_ema = 0.0

    # Components filter
    comp_cfg = cfg.get("components", {})
    min_blob_area = int(comp_cfg.get("min_blob_area", 0) or 0)
    max_blobs = int(comp_cfg.get("max_blobs", 0) or 0)

    # CNN schedule
    stride_rec = max(1, int(cnn_cfg.stride_recording))
    stride_idle_motion = max(1, int(cnn_cfg.stride_idle_motion))
    idle_full_stride_frames = 0
    if float(cnn_cfg.idle_fullframe_stride_sec) > 0:
        idle_full_stride_frames = max(1, int(round(float(cnn_cfg.idle_fullframe_stride_sec) * video_fps)))

    last_p_frame = 0.0
    cnn_runs = 0

    n_range = max(0, end_frame - start_frame + 1)
    pred_all = np.zeros((n_range,), dtype=bool)
    gt_all = np.zeros((n_range,), dtype=bool)
    detection_rows: List[Dict[str, Any]] = []

    processed = 0
    t_total0 = time.perf_counter()

    frame_iter = maybe_tqdm(range(start_frame, end_frame + 1), enabled=use_tqdm, total=n_range, desc=f"Frames: {video_id}")

    for i, frame_idx in enumerate(frame_iter):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        t_frame0 = time.perf_counter()

        frame_rs, gray = preprocess_pair(frame_bgr, resize_width=pre.get("resize_width", None))

        fgmask = mog2.apply(gray)
        fgmask = postprocess_mask(fgmask, post)

        fgmask_filt = filter_components(fgmask, min_area=min_blob_area, max_blobs=max_blobs)
        motion_binary = (fgmask_filt > 0)

        H, W = motion_binary.shape[:2]
        fg_pixels = int(np.count_nonzero(motion_binary))
        fg_ratio = float(fg_pixels) / float(max(1, H * W))

        # Motion "energy" EMA
        motion_ema = motion_alpha * fg_ratio + (1.0 - motion_alpha) * motion_ema

        gt = bool(timeline.is_positive(frame_idx))
        gt_all[i] = gt

        # motion_present (cheap scheduling heuristic)
        motion_present = fg_ratio >= float(cnn_cfg.motion_min_fg_ratio)

        # ---- CNN scheduling ----
        p_frame = float(last_p_frame)
        run_cnn = False
        ran_cnn = False
        full_frame = False

        if bool(cnn_cfg.enabled) and (cnn is not None):
            if gate.recording:
                run_cnn = (i % stride_rec) == 0
            else:
                if motion_present:
                    run_cnn = (i % stride_idle_motion) == 0
                else:
                    if idle_full_stride_frames > 0:
                        run_cnn = (i % idle_full_stride_frames) == 0
                        full_frame = True

            if (not gate.recording) and bool(cnn_cfg.start_requires_motion) and (not motion_present):
                if not full_frame:
                    run_cnn = False

            if run_cnn:
                ran_cnn = True
                if full_frame:
                    boxes = [(0, 0, frame_rs.shape[1], frame_rs.shape[0])]
                    p_frame = cnn.predict_frame_prob(frame_rs, boxes, fallback_full_frame=True)
                else:
                    mask_01 = motion_binary.astype(np.uint8)
                    boxes = boxes_from_mask(
                        mask_01,
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

        # Presence latch + stay score
        presence_score = float(presence.step(float(p_frame), motion_present=bool(motion_present)))
        stay_score = max(float(motion_ema), presence_score)

        scores: Dict[str, float] = {
            "mog2_fg_pixels": float(fg_pixels),
            "mog2_fg_ratio": float(fg_ratio),
            "cnn_p_frame": float(p_frame),
            "cnn_presence_score": float(presence_score),
            "stay_score": float(stay_score),
        }

        recording, backfill_k, reason = gate.step(scores, cnn_p_frame=float(p_frame))
        pred_all[i] = bool(recording)

        # Backfill pre-roll frames (consistent with ViBe evaluator)
        if backfill_k is not None and int(backfill_k) > 1:
            j0 = max(0, i - (int(backfill_k) - 1))
            pred_all[j0 : i + 1] = True
            for jj in range(j0, i):
                if jj < len(detection_rows):
                    detection_rows[jj]["final_decision"] = 1
                    detection_rows[jj]["trigger_source"] = "pre_roll_backfill"

        inference_time_ms = float((time.perf_counter() - t_frame0) * 1000.0)
        trigger_source = _trigger_source_from_reason(
            recording=bool(recording),
            reason=str(reason),
            motion_present=bool(motion_present),
            ran_cnn=bool(ran_cnn),
            cnn_p=float(p_frame),
            p_set=float(p_set),
        )

        # Optional debug UI
        if bool(ui.get("show_debug", False)):
            vis_mask = cv2.cvtColor(fgmask_filt.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            stacked = np.hstack([frame_rs, vis_mask])
            cv2.putText(
                stacked,
                f"rec={int(recording)} reason={reason} p={p_frame:.2f} pres={presence_score:.2f} fg={fg_ratio:.4f} ema={motion_ema:.4f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(f"MOG2+CNN Debug: {video_id}", stacked)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        detection_rows.append(
            {
                "video_id": str(video_id),
                "frame_id": int(frame_idx),
                "ground_truth": int(gt),
                # Schema-aligned names (MOG2 uses fg_ratio as "vibe_mask_ratio")
                "vibe_mask_ratio": float(fg_ratio),
                # For MOG2: "mhi_energy" is a stand-in temporal keep signal
                "mhi_energy": float(stay_score),
                "cnn_confidence": float(p_frame),
                "final_decision": int(bool(recording)),
                "inference_time_ms": float(inference_time_ms),
                "trigger_source": str(trigger_source),
            }
        )

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

    # Event-level metrics
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


def _default_out_paths(cfg_dir: Path, cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Return (summary_csv, detection_csv) with backward compatible behavior."""
    # Preferred new keys
    summary_csv = cfg.get("benchmark_summary_csv", None) or cfg.get("output_csv", None)
    detection_csv = cfg.get("detection_log_csv", None)

    if summary_csv:
        summary_csv = resolve_out_path(cfg_dir, str(summary_csv))
        if detection_csv:
            detection_csv = resolve_out_path(cfg_dir, str(detection_csv))
        else:
            sp = Path(summary_csv)
            detection_csv = str((sp.parent / f"{sp.stem}_detection_log.csv").resolve())
        return summary_csv, detection_csv

    # Old style: output_dir + run_name, with timestamp
    output_dir = str((cfg_dir / cfg.get("output_dir", "experiment_logs")).resolve())
    run_name = str(cfg.get("run_name", "mog2_cnn"))
    os.makedirs(output_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_csv = str((Path(output_dir) / f"benchmark_summary_{run_name}_{ts}.csv").resolve())
    detection_csv = str((Path(output_dir) / f"detection_log_{run_name}_{ts}.csv").resolve())
    return summary_csv, detection_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent
    cfg = load_yaml(str(cfg_path))

    # Resolve paths
    manifest_path = resolve_path(cfg["manifest_path"], bases=[cfg_dir, ROOT])
    manifest_dir = Path(manifest_path).resolve().parent

    only_ids = cfg.get("only_ids", None)
    only_ids = set(only_ids) if only_ids else None

    ui = cfg["ui"]
    use_tqdm_videos = bool(ui.get("use_tqdm", False)) and (not bool(ui.get("show_debug", False)))

    # CNN init (optional)
    cnn_cfg = CNNConfig(**cfg.get("cnn", {}))
    cnn = CNNRunner(cnn_cfg) if bool(cnn_cfg.enabled) else None

    summary_csv, detection_csv = _default_out_paths(cfg_dir, cfg)
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(detection_csv) or ".", exist_ok=True)

    manifest = load_json(manifest_path)
    items = manifest["items"]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f_sum, open(
        detection_csv, "w", newline="", encoding="utf-8"
    ) as f_det:
        w_sum = csv_mod.DictWriter(f_sum, fieldnames=BENCHMARK_SUMMARY_COLUMNS, extrasaction="ignore")
        w_det = csv_mod.DictWriter(f_det, fieldnames=DETECTION_LOG_COLUMNS, extrasaction="ignore")
        w_sum.writeheader()
        w_det.writeheader()

        video_iter = maybe_tqdm(items, enabled=use_tqdm_videos, total=len(items), desc="Videos")

        wrote_any = False
        for it in video_iter:
            vid = it["id"]
            if only_ids and vid not in only_ids:
                continue

            video_path = resolve_path(it["video_path"], bases=[manifest_dir, cfg_dir, ROOT])
            ann_path = resolve_path(it["annotation_path"], bases=[manifest_dir, cfg_dir, ROOT])

            summary_row, det_rows = eval_one_video(
                video_id=str(vid),
                video_path=str(video_path),
                annotation_path=str(ann_path),
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

    print("\nDone.")
    print(f"Wrote benchmark summary: {summary_csv}")
    print(f"Wrote detection log:     {detection_csv}")


if __name__ == "__main__":
    main()
