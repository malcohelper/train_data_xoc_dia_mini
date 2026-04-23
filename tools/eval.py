"""Evaluate a trained YOLO model against the val split and print per-class metrics.

Usage::

    python tools/eval.py --weights runs/detect/xocdia/weights/best.pt
    python tools/eval.py --weights best.pt --data xocdia.yaml --split val --imgsz 800
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from classes import CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to .pt weights.")
    parser.add_argument("--data", default="xocdia.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--conf", type=float, default=0.001, help="Conf threshold for val.")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for val.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="eval")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=False,
    )

    print("\n=== Overall ===")
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")

    # Per-class metrics. Ultralytics exposes arrays indexed by class ID.
    # Some classes may have no validation instances -> guard with len check.
    p = getattr(metrics.box, "p", []) or []
    r = getattr(metrics.box, "r", []) or []
    maps = getattr(metrics.box, "maps", []) or []
    ap50s = getattr(metrics.box, "ap50", None)

    print("\n=== Per class ===")
    print(f"{'id':>3}  {'name':<20} {'P':>7} {'R':>7} {'mAP50':>7} {'mAP50-95':>9}")
    for cid, name in CLASSES.items():
        p_v = float(p[cid]) if cid < len(p) else float("nan")
        r_v = float(r[cid]) if cid < len(r) else float("nan")
        m50 = float(ap50s[cid]) if ap50s is not None and cid < len(ap50s) else float("nan")
        m = float(maps[cid]) if cid < len(maps) else float("nan")
        print(f"{cid:>3}  {name:<20} {p_v:>7.3f} {r_v:>7.3f} {m50:>7.3f} {m:>9.3f}")

    if hasattr(metrics, "save_dir"):
        print(f"\nArtifacts: {metrics.save_dir}")
        print("  confusion_matrix.png / confusion_matrix_normalized.png / PR curves in save_dir.")


if __name__ == "__main__":
    main()
