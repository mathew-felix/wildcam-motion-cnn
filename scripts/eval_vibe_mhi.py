from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Prevent "csv shadowing" bugs
import csv as csv_mod

# --- make "src/..." imports work when running as a script ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.motion.vibe import ViBe, ViBeConfig  # noqa: E402
from src.motion.vibe_mhi import (  # noqa: E402
    MHIConfig,
    MotionHistoryImage,
    GateConfig,
    ClipGate,
    compute_scores,
)


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
    """Try p as-is, else try joining with each base."""
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


def maybe_tqdm(iterable, enabled: bool, total: Optional[int] = None, desc: str = ""):
    if not enabled:
        return iterable
    from tqdm import tqdm
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, file=sys.stdout)


@dataclass
class EventTimeline:
    intervals: List[Tuple[int, int]]  # inclusive [start_frame, end_frame]
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

    def reset(self) -> None:
        self._ptr = 0

    def is_positive(self, frame_idx: int) -> bool:
        while self._ptr < len(self.intervals) and frame_idx > self.intervals[self._ptr][1]:
            self._ptr += 1
        if self._ptr >= len(self.intervals):
            return False
        s, e = self.intervals[self._ptr]
        return s <= frame_idx <= e


def preprocess_frame(frame_bgr: np.ndarray, resize_width: Optional[int], grayscale: bool) -> np.ndarray:
    if resize_width is not None and int(resize_width) > 0:
        h, w = frame_bgr.shape[:2]
        rw = int(resize_width)
        if w != rw:
            scale = rw / float(w)
            nh = max(1, int(round(h * scale)))
            frame_bgr = cv2.resize(frame_bgr, (rw, nh), interpolation=cv2.INTER_AREA)

    # ViBe expects grayscale
    if grayscale:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


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
    """
    Remove tiny speckles and (optionally) keep only the largest K blobs.

    This matters a lot for MHI: speckles from wind/grass can accumulate in MHI and
    cause long false recordings. Filtering BEFORE MHI update is the cheapest fix.

    Args:
      mask: uint8 0/255 mask
      min_area: drop components with area < min_area (0 disables)
      max_blobs: keep only the largest max_blobs components (0 disables)

    Returns:
      uint8 0/255 mask after filtering.
    """
    if (min_area is None or int(min_area) <= 0) and (max_blobs is None or int(max_blobs) <= 0):
        return mask

    mm = (mask > 0).astype(np.uint8)
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mm, connectivity=8)

    # stats: [label, x, y, w, h, area] with 0 as background
    comps = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if min_area and area < int(min_area):
            continue
        comps.append((area, lab))

    if not comps:
        return np.zeros_like(mask)

    # Sort by area descending
    comps.sort(key=lambda t: t[0], reverse=True)

    if max_blobs and int(max_blobs) > 0:
        comps = comps[: int(max_blobs)]

    keep_labels = set(lab for _area, lab in comps)
    out = np.zeros_like(mm)
    for lab in keep_labels:
        out[labels == lab] = 1

    return (out * 255).astype(np.uint8)


def eval_one_video(
    video_id: str,
    video_path: str,
    annotation_path: str,
    manifest_fps: Optional[float],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[{video_id}] video not found: {video_path}")
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"[{video_id}] annotation not found: {annotation_path}")

    ann = load_json(annotation_path)
    events = ann.get("events", [])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[{video_id}] failed to open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 1e-6:
        video_fps = float(manifest_fps or 30.0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Eval window
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

    timeline = EventTimeline.from_events(events, video_fps)
    timeline.reset()

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pre = cfg["preprocess"]
    post = cfg["postprocess"]
    ui = cfg["ui"]

    use_tqdm = bool(ui.get("use_tqdm", False)) and (not bool(ui.get("show_debug", False)))

    # ViBe
    vibe_cfg = ViBeConfig(**cfg["vibe"])
    vibe = ViBe(vibe_cfg, rng_seed=int(cfg.get("rng_seed", 0) or 0))

    # MHI + gate
    mhi_cfg = MHIConfig(**cfg.get("mhi", {}))
    gate_cfg = GateConfig(**cfg.get("gate", {}))

    # We'll allocate MHI after first frame (because resize may change shape)
    mhi: Optional[MotionHistoryImage] = None
    gate = ClipGate(gate_cfg, fps=video_fps)

    # Component filter settings (applied before MHI update)
    comp_cfg = cfg.get("components", {})
    min_blob_area = int(comp_cfg.get("min_blob_area", 0) or 0)
    max_blobs = int(comp_cfg.get("max_blobs", 0) or 0)

    # For debugging / analysis
    sum_mhi_active = 0.0
    sum_mhi_recent = 0.0

    # Store predictions for the entire evaluated range so we can apply pre-roll cleanly
    n_range = max(0, end_frame - start_frame + 1)
    pred_all = np.zeros((n_range,), dtype=bool)
    gt_all = np.zeros((n_range,), dtype=bool)

    processed = 0
    min_fps = float("inf")
    t_total0 = time.perf_counter()

    frame_iter = maybe_tqdm(
        range(start_frame, end_frame + 1),
        enabled=use_tqdm,
        total=n_range,
        desc=f"Frames: {video_id}",
    )

    for i, frame_idx in enumerate(frame_iter):
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()

        # Preprocess -> grayscale
        gray = preprocess_frame(
            frame,
            resize_width=pre.get("resize_width", None),
            grayscale=bool(pre.get("grayscale", True)),
        )

        # Initialize MHI lazily after we know final frame shape
        if mhi is None:
            mhi = MotionHistoryImage(mhi_cfg, frame_shape=gray.shape[:2], fps=video_fps)

        # ViBe FG mask -> postprocess -> component filter
        fgmask = vibe.apply(gray)
        fgmask = postprocess_mask(fgmask, post)
        fgmask = filter_components(fgmask, min_area=min_blob_area, max_blobs=max_blobs)

        # GT membership (for all frames; we'll ignore warmup frames later)
        gt_pos = timeline.is_positive(frame_idx)
        gt_all[i] = bool(gt_pos)

        # Update MHI from the cleaned mask
        motion_binary = (fgmask > 0)
        mhi.update(motion_binary)

        # Compute scores used by gate
        scores = compute_scores(motion_binary, mhi)

        # Gate decision (clip-level recording)
        recording, backfill_k = gate.step(scores)

        # Mark current frame
        pred_all[i] = bool(recording)

        # If we just started recording, apply pre-roll by backfilling previous frames
        if backfill_k is not None and backfill_k > 1:
            j0 = max(0, i - (backfill_k - 1))
            pred_all[j0 : i + 1] = True

        # Track MHI stats
        sum_mhi_active += scores["mhi_active_ratio"]
        sum_mhi_recent += scores["mhi_recent_ratio"]

        # Runtime
        t1 = time.perf_counter()
        inst_fps = 1.0 / max(1e-12, (t1 - t0))
        min_fps = min(min_fps, inst_fps)
        processed += 1

        # Optional debug view
        if bool(ui.get("show_debug", False)):
            show = frame
            if pre.get("resize_width", None):
                show = cv2.resize(show, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
            vis_mask = cv2.cvtColor(fgmask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            stacked = np.hstack([show, vis_mask])
            cv2.imshow(f"ViBe+MHI Debug: {video_id}", stacked)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    if bool(ui.get("show_debug", False)):
        cv2.destroyAllWindows()

    # Only consider frames we actually processed (in case of early stop)
    pred_all = pred_all[:processed]
    gt_all = gt_all[:processed]

    elapsed = max(1e-12, time.perf_counter() - t_total0)
    avg_fps = processed / elapsed if processed > 0 else 0.0
    if min_fps == float("inf"):
        min_fps = 0.0

    # Score only after warmup
    score_i0 = max(0, int(score_start - start_frame))
    pred_sc = pred_all[score_i0:]
    gt_sc = gt_all[score_i0:]

    tp = int(np.sum(pred_sc & gt_sc))
    fp = int(np.sum(pred_sc & (~gt_sc)))
    fn = int(np.sum((~pred_sc) & gt_sc))
    tn = int(np.sum((~pred_sc) & (~gt_sc)))

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Kept% vs ORIGINAL total video frames, Saved% = 100 - kept%
    pred_motion_frames = int(np.sum(pred_all))
    kept_frames_pct = (pred_motion_frames / float(max(1, total_frames))) * 100.0
    saved_frames_pct = 100.0 - kept_frames_pct

    avg_mhi_active = (sum_mhi_active / processed) if processed > 0 else 0.0
    avg_mhi_recent = (sum_mhi_recent / processed) if processed > 0 else 0.0

    # mhi may be None if no frames were read
    tau_frames = int(mhi.tau_frames) if mhi is not None else 0

    return {
        "video_id": video_id,
        "video_path": video_path,

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,

        "avg_fps": avg_fps,
        "min_fps": min_fps,

        "saved_frames_pct": saved_frames_pct,
        "kept_frames_pct": kept_frames_pct,

        # Extra reproducibility + debug
        "total_frames": total_frames,
        "eval_start_frame": start_frame,
        "eval_end_frame": end_frame,
        "warmup_frames": warmup_frames,
        "processed_frames": processed,
        "pred_recorded_frames": pred_motion_frames,
        "video_fps_used": video_fps,

        # MHI/Gate stats
        "mhi_tau_sec": float(mhi_cfg.tau_sec),
        "mhi_tau_frames": tau_frames,
        "avg_mhi_active_ratio": avg_mhi_active,
        "avg_mhi_recent_ratio": avg_mhi_recent,

        "gate_start_metric": gate_cfg.start_metric,
        "gate_start_threshold": float(gate_cfg.start_threshold),
        "gate_start_consecutive": int(gate_cfg.start_consecutive),
        "gate_stay_metric": gate_cfg.stay_metric,
        "gate_stay_threshold": float(gate_cfg.stay_threshold),
        "gate_pre_roll_sec": float(gate_cfg.pre_roll_sec),
        "gate_post_roll_sec": float(gate_cfg.post_roll_sec),

        "comp_min_blob_area": int(min_blob_area),
        "comp_max_blobs": int(max_blobs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_vibe_mhi_high_recall.yaml", help="Path to YAML config.")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent

    cfg = load_yaml(str(cfg_path))

    manifest_path = resolve_path(cfg["manifest_path"], bases=[cfg_dir, ROOT])
    out_csv = str((cfg_dir / cfg["output_csv"]).resolve()) if not Path(cfg["output_csv"]).is_absolute() else cfg["output_csv"]

    manifest = load_json(manifest_path)
    items = manifest["items"]

    only_ids = cfg.get("only_ids", None)
    only_ids = set(only_ids) if only_ids else None

    ui = cfg["ui"]
    use_tqdm_videos = bool(ui.get("use_tqdm", False)) and (not bool(ui.get("show_debug", False)))

    rows: List[Dict[str, Any]] = []
    video_iter = maybe_tqdm(items, enabled=use_tqdm_videos, total=len(items), desc="Videos")

    manifest_dir = Path(manifest_path).resolve().parent

    for it in video_iter:
        vid = it["id"]
        if only_ids and vid not in only_ids:
            continue

        video_path = resolve_path(it["video_path"], bases=[manifest_dir, cfg_dir, ROOT])
        ann_path = resolve_path(it["annotation_path"], bases=[manifest_dir, cfg_dir, ROOT])

        row = eval_one_video(
            video_id=vid,
            video_path=video_path,
            annotation_path=ann_path,
            manifest_fps=it.get("fps", None),
            cfg=cfg,
        )
        rows.append(row)

    if not rows:
        print("No videos evaluated (check only_ids / manifest).")
        return

    required_cols = [
        "video_id",
        "tp", "fp", "fn", "tn",
        "precision", "recall",
        "avg_fps", "min_fps",
        "saved_frames_pct",
    ]
    extra_cols = [k for k in rows[0].keys() if k not in required_cols]
    cols = [*required_cols, *extra_cols]
    cols = [str(c) for c in cols]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
