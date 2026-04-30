"""Per-class YOLO detection diagnostic.

Sweeps confidence + image-size combinations on a single frame and
reports how many detections each class produces. Use it to figure
out *why* live capture sometimes returns 0 detections — usually
either:

* The model genuinely can't find the cells at the deployed
  ``conf=0.4`` threshold (lower-conf would surface them).
* The bounding boxes are tiny relative to ``imgsz=800`` and need a
  larger inference size.
* Specific classes are missing entirely (training-data gap).

Usage::

    python tools/diag_detection.py path/to/capture_*.png

Saves an annotated image alongside the input named
``<input>.diag.<conf>_<imgsz>.png`` so you can eyeball where the
boxes landed at the most-permissive setting.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from classes import CLASSES  # noqa: E402
from detector import XocDiaDetector  # noqa: E402


# (conf, imgsz) sweep. Ordered from "production today" -> "throw the
# kitchen sink at it". Stops early once we see >= 10 detections per
# pass to keep output small.
SWEEPS: List[Tuple[float, int]] = [
    (0.40, 800),   # production default
    (0.25, 800),   # lower conf only
    (0.40, 1280),  # higher imgsz only
    (0.25, 1280),  # both relaxed
    (0.10, 1280),  # very permissive
    (0.05, 1600),  # last-resort recall
]


def annotate(frame: np.ndarray, detections, out_path: Path) -> None:
    canvas = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.class_name} {d.conf:.2f}"
        cv2.putText(canvas, label, (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imwrite(str(out_path), canvas)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("frame", help="Path to a saved capture frame (PNG).")
    p.add_argument("--weights",
                   default="runs/detect/runs/detect/xocdia-2/weights/best.pt")
    p.add_argument("--device", default=None,
                   help="Pass 'cpu' or 'mps' to force a device.")
    args = p.parse_args()

    frame_path = Path(args.frame).expanduser().resolve()
    if not frame_path.is_file():
        print(f"ERROR: frame not found: {frame_path}", file=sys.stderr)
        return 2
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"ERROR: failed to decode {frame_path}", file=sys.stderr)
        return 2
    print(f"Frame: {frame_path}  shape={frame.shape}")

    print("\nclass id -> name:")
    for cid, name in sorted(CLASSES.items()):
        print(f"  {cid:>2} {name}")

    print("\nSweep (conf, imgsz) -> total dets / per-class breakdown:")
    print("-" * 72)
    for conf, imgsz in SWEEPS:
        det = XocDiaDetector(weights=args.weights, conf=conf,
                             imgsz=imgsz, device=args.device)
        dets = det.detect(frame)
        per_cls = Counter(d.class_name for d in dets)
        per_cls_str = ", ".join(f"{k}={v}" for k, v in
                                sorted(per_cls.items())) or "-"
        print(f"conf={conf:.2f} imgsz={imgsz:>4}  total={len(dets):>3}  "
              f"per_cls=[{per_cls_str}]")

        out_path = frame_path.with_suffix(
            f".diag.c{int(conf*100):02d}_i{imgsz}.png")
        annotate(frame, dets, out_path)

    print("\nAnnotated images written next to the input frame "
          f"(suffix .diag.cXX_iYYYY.png).")
    print("\nInterpretation hints:")
    print("  * If conf=0.40 imgsz=800 returns 0 but conf=0.25 imgsz=1280")
    print("    finds them: lower the live-capture defaults.")
    print("  * If even conf=0.05 imgsz=1600 finds nothing: the model")
    print("    has a training-data gap for this scene; capture more")
    print("    frames + relabel.")
    print("  * If only some classes show up at low conf: that class is")
    print("    under-represented in training - target labelling there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
