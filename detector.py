"""Single-stage YOLOv8 detector for Xoc Dia UI (15-class schema).

This module is intentionally lean: it only does detection + light post-
processing (grouping, annotation). OCR and game-state reasoning live in
``pipeline.py`` so the detector stays reusable in batch / streaming jobs.

Example::

    from detector import XocDiaDetector
    det = XocDiaDetector(weights="runs/detect/runs/detect/xocdia/weights/best.pt")
    detections = det.detect(frame)
    groups = det.group_by_category(detections)
    annotated = det.annotate(frame, detections)
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from classes import CLASSES, COLORS, category_of


DEFAULT_WEIGHTS = "runs/detect/runs/detect/xocdia/weights/best.pt"


@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    category: str = field(default="")

    def __post_init__(self):
        if not self.category:
            self.category = category_of(self.class_id)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area_px(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def contains_point(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def iou(self, other: "Detection") -> float:
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = self.area_px + other.area_px - inter
        return inter / union if union > 0 else 0.0


class XocDiaDetector:
    """Thin wrapper around ``ultralytics.YOLO`` with typed outputs.

    Two layered fallbacks rescue scenes the primary pass misses:

    1. **imgsz fallback** (``imgsz_fallback``, runs ALWAYS by default):
       always re-run YOLO at a larger ``imgsz_fallback`` and merge.
       The previous version gated this behind ``fallback_threshold``,
       which fired only when primary returned <3 dets - but in
       practice the small-text classes (``percent_cell``, ``timer``,
       ``total_count_cell``) are exactly the ones that vanish at
       ``imgsz=800`` while the larger panel boxes stay visible, so
       primary returns ``len ~= 11`` with the small classes missing
       and the gate never fires. Always-run is a multi-scale ensemble:
       cheap (one extra forward pass) and recovers the small classes
       reliably. Set ``imgsz_fallback_always=False`` to revert to the
       threshold-gated behaviour, or ``imgsz_fallback=0`` to disable.
    2. **crop fallback** (``crop_fallback_top_pct``, gated by
       ``fallback_threshold``): if the merged result still has fewer
       than ``fallback_threshold`` detections, drop the top
       ``crop_fallback_top_pct`` of the frame (where the history-%
       CHĂN/LẺ overlay tends to sit) and re-run at ``imgsz_fallback``.
       Bbox coordinates from the crop are shifted back into the
       original frame's coordinate space. This is the more expensive
       path so we keep it gated.

    All result sets are merged through per-class NMS
    (``crop_fallback_iou``) so callers see one consistent list. Set
    ``imgsz_fallback=0`` and ``fallback_threshold=0`` to disable both.
    """

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        conf: float = 0.4,
        iou: float = 0.45,
        device: Optional[str] = None,
        imgsz: int = 800,
        imgsz_fallback: int = 1280,
        imgsz_fallback_always: bool = True,
        fallback_threshold: int = 3,
        crop_fallback_top_pct: float = 0.30,
        crop_fallback_iou: float = 0.5,
        # Back-compat alias - older callers passed ``crop_fallback_threshold``.
        crop_fallback_threshold: Optional[int] = None,
    ):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.imgsz_fallback = imgsz_fallback
        self.imgsz_fallback_always = imgsz_fallback_always
        if crop_fallback_threshold is not None:
            fallback_threshold = crop_fallback_threshold
        self.fallback_threshold = fallback_threshold
        self.crop_fallback_top_pct = crop_fallback_top_pct
        self.crop_fallback_iou = crop_fallback_iou

    # Back-compat read-only alias so external callers / configs that
    # still reference the old attribute name keep working.
    @property
    def crop_fallback_threshold(self) -> int:
        return self.fallback_threshold

    # ---------- inference ----------

    def _infer(
        self,
        frame: np.ndarray,
        imgsz: Optional[int] = None,
    ) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=imgsz if imgsz is not None else self.imgsz,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        if results.boxes is None:
            return detections

        xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), cid, cf in zip(xyxy, cls_ids, confs):
            detections.append(
                Detection(
                    class_id=int(cid),
                    class_name=CLASSES.get(int(cid), f"cls{cid}"),
                    conf=float(cf),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        return detections

    @staticmethod
    def _shift_y(dets: List[Detection], dy: int) -> List[Detection]:
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            d.bbox = (x1, y1 + dy, x2, y2 + dy)
        return dets

    def _merge_with_nms(
        self,
        primary: List[Detection],
        secondary: List[Detection],
    ) -> List[Detection]:
        """Add detections from ``secondary`` that don't overlap with any
        same-class detection already in ``primary``. When two same-class
        boxes overlap above ``crop_fallback_iou`` we keep the
        higher-confidence one."""
        merged = list(primary)
        for cand in secondary:
            best_existing_idx: Optional[int] = None
            best_iou = 0.0
            for i, kept in enumerate(merged):
                if kept.class_id != cand.class_id:
                    continue
                iou = kept.iou(cand)
                if iou > best_iou:
                    best_iou = iou
                    best_existing_idx = i
            if best_existing_idx is None or best_iou < self.crop_fallback_iou:
                merged.append(cand)
            elif cand.conf > merged[best_existing_idx].conf:
                merged[best_existing_idx] = cand
        return merged

    def detect(self, frame: np.ndarray) -> List[Detection]:
        primary = self._infer(frame)
        merged = primary

        # Stage 1: same frame at a larger imgsz so small-text classes
        # (percent_cell / timer / total_count_cell) that get downsampled
        # past the model's receptive field at imgsz=800 surface again
        # at imgsz=1280. Always-run by default because primary alone is
        # unreliable: it can return ~11 dets while still missing every
        # percent_cell, which would never satisfy a ``len < N`` gate.
        run_imgsz_fallback = (
            self.imgsz_fallback > 0
            and self.imgsz_fallback != self.imgsz
            and (
                self.imgsz_fallback_always
                or (
                    self.fallback_threshold > 0
                    and len(primary) < self.fallback_threshold
                )
            )
        )
        if run_imgsz_fallback:
            bigger = self._infer(frame, imgsz=self.imgsz_fallback)
            merged = self._merge_with_nms(merged, bigger)

        # Stage 2 (gated): if even the multi-scale merge still under-
        # detects, crop the top % (where the history-% overlay sits)
        # and retry at the bumped imgsz so the betting panel dominates
        # the input.
        if (
            self.fallback_threshold > 0
            and len(merged) < self.fallback_threshold
            and 0 < self.crop_fallback_top_pct < 1
        ):
            h = frame.shape[0]
            crop_top = int(h * self.crop_fallback_top_pct)
            # Defensive guard: too little frame left to be useful.
            if h - crop_top >= 200:
                cropped = frame[crop_top:]
                crop_imgsz = (
                    self.imgsz_fallback
                    if self.imgsz_fallback > 0
                    else self.imgsz
                )
                crop_dets = self._shift_y(
                    self._infer(cropped, imgsz=crop_imgsz),
                    crop_top,
                )
                merged = self._merge_with_nms(merged, crop_dets)

        return merged

    def detect_batch(self, frames: Iterable[np.ndarray]) -> List[List[Detection]]:
        return [self.detect(f) for f in frames]

    # ---------- helpers ----------

    @staticmethod
    def group_by_category(
        detections: Iterable[Detection],
    ) -> Dict[str, List[Detection]]:
        groups: Dict[str, List[Detection]] = {
            "state": [], "area": [], "dice": [], "cell": [], "unknown": [],
        }
        for d in detections:
            groups.setdefault(d.category, []).append(d)
        return groups

    @staticmethod
    def group_by_class(
        detections: Iterable[Detection],
    ) -> Dict[str, List[Detection]]:
        groups: Dict[str, List[Detection]] = {}
        for d in detections:
            groups.setdefault(d.class_name, []).append(d)
        return groups

    @staticmethod
    def crop(frame: np.ndarray, det: Detection) -> np.ndarray:
        x1, y1, x2, y2 = det.bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        return frame[y1:y2, x1:x2]

    # ---------- visualization ----------

    def annotate(
        self,
        frame: np.ndarray,
        detections: Optional[List[Detection]] = None,
    ) -> np.ndarray:
        if detections is None:
            detections = self.detect(frame)

        out = frame.copy()
        for d in detections:
            color = COLORS.get(d.class_id, (200, 200, 200))
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_text = max(th + 4, y1 - 4)
            cv2.rectangle(
                out,
                (x1, y_text - th - 4),
                (x1 + tw + 4, y_text + 2),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (x1 + 2, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
        return out


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run XocDia detector on an image.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", required=True, help="Path to an image file.")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save annotated image. If omitted, write next to source.",
    )
    return parser.parse_args()


def main():
    from pathlib import Path

    args = parse_args()
    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"Cannot read image: {args.source}")

    det = XocDiaDetector(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
    )
    detections = det.detect(frame)
    print(f"{len(detections)} detections:")
    for d in sorted(detections, key=lambda x: (-x.conf,)):
        print(f"  [{d.category:>5}] {d.class_name:<20} conf={d.conf:.3f} bbox={d.bbox}")

    out_path = Path(args.output) if args.output else Path(args.source).with_suffix(".annotated.png")
    cv2.imwrite(str(out_path), det.annotate(frame, detections))
    print(f"Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()
