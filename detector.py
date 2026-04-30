"""Single-stage YOLOv8 detector for Xoc Dia UI (15-class schema).

This module is intentionally lean: it only does detection + light post-
processing (grouping, annotation). OCR and game-state reasoning live in
``pipeline.py`` so the detector stays reusable in batch / streaming jobs.

Example::

    from detector import XocDiaDetector
    det = XocDiaDetector(weights="runs/detect/xocdia/weights/best.pt")
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

    Crop-fallback recall (``crop_fallback_threshold`` / ``crop_fallback_top_pct``)
    is a defensive second pass: when the first full-frame inference returns
    fewer detections than ``crop_fallback_threshold`` we re-run YOLO on the
    bottom ``1 - crop_fallback_top_pct`` of the frame. Empirically the
    history-percent overlay that the game UI sometimes paints across the
    top of the screen confuses the network and zeroes out detections; the
    crop trims that overlay off and lets the betting panel features win
    again. Bbox coordinates from the crop are shifted back into the
    original frame's coordinate space and merged via per-class NMS so
    callers see one consistent list. Set ``crop_fallback_threshold=0`` to
    disable.
    """

    def __init__(
        self,
        weights: str = "runs/detect/xocdia/weights/best.pt",
        conf: float = 0.4,
        iou: float = 0.45,
        device: Optional[str] = None,
        imgsz: int = 800,
        crop_fallback_threshold: int = 3,
        crop_fallback_top_pct: float = 0.30,
        crop_fallback_iou: float = 0.5,
    ):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.crop_fallback_threshold = crop_fallback_threshold
        self.crop_fallback_top_pct = crop_fallback_top_pct
        self.crop_fallback_iou = crop_fallback_iou

    # ---------- inference ----------

    def _infer(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
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

        if (
            self.crop_fallback_threshold <= 0
            or len(primary) >= self.crop_fallback_threshold
            or self.crop_fallback_top_pct <= 0
            or self.crop_fallback_top_pct >= 1
        ):
            return primary

        h = frame.shape[0]
        crop_top = int(h * self.crop_fallback_top_pct)
        # Defensive guard: if cropping leaves too little of the frame,
        # YOLO can't make use of it. Skip the fallback in that case.
        if h - crop_top < 200:
            return primary

        cropped = frame[crop_top:]
        crop_dets = self._shift_y(self._infer(cropped), crop_top)
        return self._merge_with_nms(primary, crop_dets)

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
    parser.add_argument("--weights", default="runs/detect/xocdia/weights/best.pt")
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
