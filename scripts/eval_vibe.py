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

# IMPORTANT: prevent "csv" shadowing bugs
import csv as csv_mod

# --- make "src/..." imports work when running as a script ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.motion.vibe import ViBe, ViBeConfig  # noqa: E402


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
    return str(pp)  # last resort (let later open fail with clear error)


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
            end_f = int(math.ceil(t * fps)) - 1  # inclusive
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


def motion_flag(mask: np.ndarray, cfg: Dict[str, Any]) -> bool:
    fg = int(np.count_nonzero(mask))
    h, w = mask.shape[:2]
    total = max(1, h * w)

    min_fg_pixels = cfg.get("min_fg_pixels", None)
    min_fg_ratio = cfg.get("min_fg_ratio", None)

    if min_fg_pixels is not None:
        return fg >= int(min_fg_pixels)
    if min_fg_ratio is not None:
        return (fg / float(total)) >= float(min_fg_ratio)

    return fg > 0


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

    # If debug window is enabled, disable tqdm (they don't play nicely)
    use_tqdm = bool(ui.get("use_tqdm", False)) and (not bool(ui.get("show_debug", False)))

    vibe_cfg = ViBeConfig(**cfg["vibe"])
    vibe = ViBe(vibe_cfg, rng_seed=int(cfg.get("rng_seed", 0) or 0))

    tp = fp = fn = tn = 0
    pred_pos_frames = 0
    processed = 0
    min_fps = float("inf")

    t_total0 = time.perf_counter()

    eval_frames = max(0, end_frame - start_frame + 1)
    frame_iter = maybe_tqdm(
        range(start_frame, end_frame + 1),
        enabled=use_tqdm,
        total=eval_frames,
        desc=f"Frames: {video_id}",
    )

    for frame_idx in frame_iter:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()

        gray = preprocess_frame(
            frame,
            resize_width=pre.get("resize_width", None),
            grayscale=bool(pre.get("grayscale", True)),
        )
        fgmask = vibe.apply(gray)
        fgmask = postprocess_mask(fgmask, post)

        pred_pos = motion_flag(fgmask, post)
        if pred_pos:
            pred_pos_frames += 1

        if frame_idx >= score_start:
            gt_pos = timeline.is_positive(frame_idx)
            if pred_pos and gt_pos:
                tp += 1
            elif pred_pos and (not gt_pos):
                fp += 1
            elif (not pred_pos) and gt_pos:
                fn += 1
            else:
                tn += 1

        t1 = time.perf_counter()
        inst_fps = 1.0 / max(1e-12, (t1 - t0))
        min_fps = min(min_fps, inst_fps)
        processed += 1

        if bool(ui.get("show_debug", False)):
            show = frame
            if pre.get("resize_width", None):
                show = cv2.resize(show, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
            vis_mask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
            stacked = np.hstack([show, vis_mask])
            cv2.imshow(f"ViBe Debug: {video_id}", stacked)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    if bool(ui.get("show_debug", False)):
        cv2.destroyAllWindows()

    elapsed = max(1e-12, time.perf_counter() - t_total0)
    avg_fps = processed / elapsed if processed > 0 else 0.0
    if min_fps == float("inf"):
        min_fps = 0.0

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Kept% vs ORIGINAL total video frames, Saved% = 100 - kept%
    kept_frames_pct = (pred_pos_frames / float(max(1, total_frames))) * 100.0
    saved_frames_pct = 100.0 - kept_frames_pct

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

        "saved_frames_pct": saved_frames_pct,  # required
        "kept_frames_pct": kept_frames_pct,    # extra (useful)

        "total_frames": total_frames,
        "eval_start_frame": start_frame,
        "eval_end_frame": end_frame,
        "warmup_frames": warmup_frames,
        "processed_frames": processed,
        "pred_motion_frames": pred_pos_frames,
        "video_fps_used": video_fps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_vibe.yaml", help="Path to YAML config.")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent

    cfg = load_yaml(str(cfg_path))

    # Resolve manifest/output relative to config folder and project root
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
