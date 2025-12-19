from __future__ import annotations

"""
ViBe + MHI + CNN (MobileNetV3-Small) utilities — "FP-safe" high-recall variant.

This version is specifically tuned to reduce the false-positive explosion you saw
(e.g., deer_night) while keeping recall improvements where CNN truly helps.

Notes:
  - In configs, AT typically maps to vibe.adaptive_min_matches
  - In configs, ALR typically maps to vibe.adaptive_update

What changed vs v2:
  1) model_format="auto" still supported (TorchScript -> state_dict fallback)
  2) CNN START can be required to have motion present (prevents idle CNN latch)
  3) Presence latch refresh can require motion (prevents latch from refreshing on noise)
     - BUT a *very high* p_frame (>= p_start_high) can refresh even without motion
  4) Idle full-frame spot-check is OFF by default in config (big FP + speed cost)

Your training setup (from your code):
  - student.classifier[-1] = Linear(in_features, 1)
  - loss: BCEWithLogitsLoss
  - save: torch.save(student.state_dict(), "...best_by_recall.pt")
So the default inference mode is:
  - num_classes: 1
  - output: sigmoid
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


# =============================================================================
# CNN runner (PyTorch)
# =============================================================================

@dataclass
class CNNConfig:
    enabled: bool = False

    # Model loading
    model_path: str = ""
    model_format: str = "auto"                # "auto", "torchscript", "torchvision_mobilenetv3_small"
    device: str = "auto"

    # Input preprocessing
    input_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Output interpretation (your model is 1-logit -> sigmoid)
    output: str = "sigmoid"                   # "auto", "sigmoid", "softmax"
    animal_index: int = 1                     # only for softmax

    # Latch thresholds
    p_set: float = 0.45                       # refresh latch when >= p_set (motion required by default)
    p_start_high: float = 0.80                # "very confident" threshold (can refresh even without motion)

    presence_tau_sec: float = 2.0             # latch duration in seconds

    # Scheduling
    stride_recording: int = 8                 # run CNN every N frames while recording
    stride_idle_motion: int = 15              # run CNN every N frames while idle but motion exists
    idle_fullframe_stride_sec: float = 0.0    # OFF by default (set >0 to enable)

    # Crop proposals
    max_crops: int = 3
    min_crop_area: int = 160                  # slightly higher to drop tiny speckles
    expand_ratio: float = 1.35                # reduce overreach (helps deer_night FPs)
    pad_px: int = 4
    full_frame_if_no_crops: bool = True

    # Motion presence heuristic (used for scheduling + optional requirements)
    motion_min_fg_ratio: float = 0.00025

    # Safety switches (reduce FP explosions)
    start_requires_motion: bool = True        # CNN-start requires motion present
    refresh_requires_motion: bool = True      # latch refresh at p_set requires motion present

    # Torchvision model options (state_dict path)
    num_classes: int = 1
    state_dict_key: Optional[str] = None
    strict_load: bool = False


class CNNRunner:
    """
    model_format="auto":
      1) try torch.jit.load(...) -> TorchScript archive
      2) fallback: load state_dict into torchvision mobilenet_v3_small
    """

    def __init__(self, cfg: CNNConfig):
        self.cfg = cfg
        self._torch = None
        self._torchvision = None

        if not cfg.enabled:
            self.model = None
            self.device = "cpu"
            return

        try:
            import torch  # noqa
            self._torch = torch
        except ImportError as e:
            raise ImportError("PyTorch is required for cnn.enabled=true. Install torch/torchvision.") from e

        self.device = self._select_device(cfg.device)
        self.model = self._load_model(cfg)
        self.model.eval()
        self.model.to(self.device)

        mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array(cfg.std, dtype=np.float32).reshape(1, 3, 1, 1)
        self._mean = self._torch.from_numpy(mean).to(self.device)
        self._std = self._torch.from_numpy(std).to(self.device)

    def _select_device(self, dev: str) -> str:
        dev = str(dev or "auto").lower()
        if dev == "cpu":
            return "cpu"
        if dev == "cuda":
            return "cuda"
        return "cuda" if (self._torch is not None and self._torch.cuda.is_available()) else "cpu"

    def _load_model(self, cfg: CNNConfig):
        torch = self._torch
        mp = str(cfg.model_path or "").strip()
        if not mp:
            raise ValueError("cnn.model_path is required when cnn.enabled=true")

        fmt = str(cfg.model_format or "auto").lower()

        def _load_torchscript():
            return torch.jit.load(mp, map_location=self.device)

        def _load_state_dict():
            try:
                import torchvision
                self._torchvision = torchvision
            except ImportError as e:
                raise ImportError(
                    "torchvision is required to load a state_dict MobileNetV3-Small. "
                    "Install torchvision or export TorchScript."
                ) from e

            m = self._torchvision.models.mobilenet_v3_small(weights=None)
            in_features = m.classifier[-1].in_features
            m.classifier[-1] = torch.nn.Linear(in_features, int(cfg.num_classes))

            ckpt = torch.load(mp, map_location="cpu")
            if isinstance(ckpt, dict) and cfg.state_dict_key and cfg.state_dict_key in ckpt:
                ckpt = ckpt[cfg.state_dict_key]
            if isinstance(ckpt, dict) and "state_dict" in ckpt and cfg.state_dict_key is None:
                ckpt = ckpt["state_dict"]

            if not isinstance(ckpt, dict):
                raise ValueError("Expected a state_dict dict from torch.load(model_path).")

            cleaned = {}
            for k, v in ckpt.items():
                kk = str(k)
                if kk.startswith("module."):
                    kk = kk[len("module."):]
                cleaned[kk] = v

            m.load_state_dict(cleaned, strict=bool(cfg.strict_load))
            return m

        if fmt == "torchscript":
            return _load_torchscript()

        if fmt == "torchvision_mobilenetv3_small":
            return _load_state_dict()

        if fmt == "auto":
            try:
                return _load_torchscript()
            except Exception:
                return _load_state_dict()

        raise ValueError(f"Unknown cnn.model_format: {cfg.model_format}")

    def _preprocess_bgr_batch(self, crops_bgr: List[np.ndarray]):
        torch = self._torch
        size = int(self.cfg.input_size)

        xs = []
        for im in crops_bgr:
            if im is None or im.size == 0:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(rgb).to(torch.float32) / 255.0
            x = x.permute(2, 0, 1).contiguous()
            xs.append(x)

        if not xs:
            return None

        batch = torch.stack(xs, dim=0).to(self.device)
        batch = (batch - self._mean) / self._std
        return batch

    def _to_probs(self, logits):
        torch = self._torch
        out = logits
        if isinstance(out, (list, tuple)):
            out = out[0]
        if not torch.is_tensor(out):
            out = torch.as_tensor(out)

        mode = str(self.cfg.output or "auto").lower()

        if mode == "sigmoid":
            if out.ndim == 2 and out.shape[1] == 1:
                out = out[:, 0]
            probs = torch.sigmoid(out)
        elif mode == "softmax":
            probs_all = torch.softmax(out, dim=1)
            probs = probs_all[:, int(self.cfg.animal_index)]
        else:
            # auto
            if out.ndim == 1:
                probs = torch.sigmoid(out)
            elif out.ndim == 2 and out.shape[1] == 1:
                probs = torch.sigmoid(out[:, 0])
            elif out.ndim == 2 and out.shape[1] >= 2:
                probs_all = torch.softmax(out, dim=1)
                probs = probs_all[:, int(self.cfg.animal_index)]
            else:
                probs = torch.sigmoid(out.reshape(out.shape[0], -1)[:, 0])

        return probs.detach().to("cpu").numpy().astype(np.float32)

    def predict_frame_prob(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: List[Tuple[int, int, int, int]],
        fallback_full_frame: bool,
    ) -> float:
        if not self.cfg.enabled or self.model is None:
            return 0.0

        crops: List[np.ndarray] = []
        H, W = frame_bgr.shape[:2]

        for (x1, y1, x2, y2) in boxes_xyxy:
            x1 = int(max(0, min(W - 1, x1)))
            x2 = int(max(0, min(W, x2)))
            y1 = int(max(0, min(H - 1, y1)))
            y2 = int(max(0, min(H, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(frame_bgr[y1:y2, x1:x2])

        if (not crops) and fallback_full_frame:
            crops = [frame_bgr]

        if not crops:
            return 0.0

        x = self._preprocess_bgr_batch(crops)
        if x is None:
            return 0.0

        torch = self._torch
        with torch.no_grad():
            y = self.model(x)
        probs = self._to_probs(y)

        return float(np.max(probs)) if probs.size else 0.0


# =============================================================================
# Motion -> box proposals
# =============================================================================

def boxes_from_mask(
    mask_01: np.ndarray,
    frame_shape_hw: Tuple[int, int],
    min_area: int,
    max_boxes: int,
    expand_ratio: float,
    pad_px: int,
) -> List[Tuple[int, int, int, int]]:
    H, W = int(frame_shape_hw[0]), int(frame_shape_hw[1])
    mm = (mask_01 > 0).astype(np.uint8)

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mm, connectivity=8)
    comps: List[Tuple[int, int, int, int, int]] = []  # area,x,y,w,h

    for lab in range(1, num):
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if int(min_area) > 0 and area < int(min_area):
            continue
        comps.append((area, x, y, w, h))

    if not comps:
        return []

    comps.sort(key=lambda t: t[0], reverse=True)
    if int(max_boxes) > 0:
        comps = comps[: int(max_boxes)]

    out: List[Tuple[int, int, int, int]] = []
    for _area, x, y, w, h in comps:
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        nw = w * float(expand_ratio)
        nh = h * float(expand_ratio)

        x1 = int(round(cx - 0.5 * nw)) - int(pad_px)
        y1 = int(round(cy - 0.5 * nh)) - int(pad_px)
        x2 = int(round(cx + 0.5 * nw)) + int(pad_px)
        y2 = int(round(cy + 0.5 * nh)) + int(pad_px)

        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W, x2))
        y2 = max(0, min(H, y2))

        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))

    return out


# =============================================================================
# Presence latch + Hybrid gate
# =============================================================================

@dataclass
class PresenceConfig:
    enabled: bool = True
    tau_sec: float = 2.0
    p_set: float = 0.45
    p_start_high: float = 0.80
    refresh_requires_motion: bool = True


class PresenceLatch:
    """
    Refresh policy:
      - Always refresh if p_frame >= p_start_high (very confident)
      - Else refresh if p_frame >= p_set and (not refresh_requires_motion or motion_present)
    """
    def __init__(self, cfg: PresenceConfig, fps: float):
        self.cfg = cfg
        self.fps = float(max(1e-6, fps))
        self.tau_frames = max(1, int(round(float(cfg.tau_sec) * self.fps)))
        self.mem = 0

    def reset(self):
        self.mem = 0

    def step(self, p_frame: float, motion_present: bool) -> float:
        if not self.cfg.enabled:
            return 0.0

        self.mem = max(0, int(self.mem) - 1)
        p = float(p_frame)

        refresh = False
        if p >= float(self.cfg.p_start_high):
            refresh = True
        elif p >= float(self.cfg.p_set):
            refresh = (motion_present or (not bool(self.cfg.refresh_requires_motion)))

        if refresh:
            self.mem = self.tau_frames

        return float(self.mem) / float(self.tau_frames)


@dataclass
class HybridGateConfig:
    enabled: bool = True

    # Motion START
    start_metric: str = "vibe_fg_pixels"
    start_threshold: float = 550.0
    start_consecutive: int = 1

    # CNN START
    cnn_start_enabled: bool = True
    cnn_start_threshold: float = 0.80
    cnn_start_consecutive: int = 2
    cnn_start_requires_motion: bool = True
    cnn_start_motion_metric: str = "vibe_fg_ratio"
    cnn_start_motion_threshold: float = 0.00025

    # STAY (use "stay_score")
    stay_metric: str = "stay_score"
    stay_threshold: float = 0.0010

    # Clip shaping
    pre_roll_sec: float = 0.0
    post_roll_sec: float = 1.5


class HybridClipGate:
    def __init__(self, cfg: HybridGateConfig, fps: float):
        self.cfg = cfg
        self.fps = float(max(1e-6, fps))
        self.pre_roll_frames = max(0, int(round(float(cfg.pre_roll_sec) * self.fps)))
        self.post_roll_frames = max(0, int(round(float(cfg.post_roll_sec) * self.fps)))

        self.recording = False
        self._motion_hits = 0
        self._cnn_hits = 0
        self._post_left = 0

        self.last_reason = "init"
    def reset(self):
        self.recording = False
        self._motion_hits = 0
        self._cnn_hits = 0
        self._post_left = 0

        self.last_reason = "reset"
    @staticmethod
    def _get(scores: Dict[str, float], name: str) -> float:
        if name not in scores:
            raise KeyError(f"Gate metric '{name}' missing. Available: {list(scores.keys())}")
        return float(scores[name])

    def step(self, scores: Dict[str, float], cnn_p_frame: float) -> Tuple[bool, Optional[int]]:
        """
        Step the gate for one frame.

        Returns:
          (recording, backfill_k)

        Side-effect:
          sets self.last_reason to a short string describing *why* the decision happened.
          Possible values: init, reset, disabled, no_start, start_motion, start_cnn,
          start_motion_and_cnn, stay, post_roll, stop.
          This is used by evaluation scripts to build per-frame detection logs.
        """
        if not self.cfg.enabled:
            self.last_reason = "disabled"
            return True, None

        start_val = self._get(scores, self.cfg.start_metric)
        stay_val = self._get(scores, self.cfg.stay_metric)

        if not self.recording:
            # motion start
            if start_val >= float(self.cfg.start_threshold):
                self._motion_hits += 1
            else:
                self._motion_hits = 0

            # cnn start (optionally requires motion)
            cnn_allowed = True
            if bool(self.cfg.cnn_start_requires_motion):
                mv = self._get(scores, self.cfg.cnn_start_motion_metric)
                cnn_allowed = mv >= float(self.cfg.cnn_start_motion_threshold)

            if bool(self.cfg.cnn_start_enabled) and cnn_allowed and (float(cnn_p_frame) >= float(self.cfg.cnn_start_threshold)):
                self._cnn_hits += 1
            else:
                self._cnn_hits = 0

            motion_ok = self._motion_hits >= max(1, int(self.cfg.start_consecutive))
            cnn_ok = self._cnn_hits >= max(1, int(self.cfg.cnn_start_consecutive))

            if motion_ok or cnn_ok:
                self.recording = True
                self._post_left = self.post_roll_frames

                # Reason + cleanup
                if motion_ok and cnn_ok:
                    self.last_reason = "start_motion_and_cnn"
                elif cnn_ok:
                    self.last_reason = "start_cnn"
                else:
                    self.last_reason = "start_motion"

                self._motion_hits = 0
                self._cnn_hits = 0

                backfill_k = self.pre_roll_frames + 1
                return True, backfill_k

            self.last_reason = "no_start"
            return False, None

        # recording
        if stay_val >= float(self.cfg.stay_threshold):
            self._post_left = self.post_roll_frames
            self.last_reason = "stay"
            return True, None

        if self._post_left > 0:
            self._post_left -= 1
            self.last_reason = "post_roll"
            return True, None

        self.recording = False
        self.last_reason = "stop"
        return False, None
