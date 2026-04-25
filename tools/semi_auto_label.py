"""Pre-label new images using a trained detector.

After your first round of training the model is good enough to bootstrap
the next batch of labels: instead of drawing every box from scratch, you
run this script to fill ``dataset/labels/<split>`` with YOLO-format
predictions, then open the same images in ``label_tool.py`` and just fix
what the model got wrong.

Default policy (safe):
- Only writes a .txt file for an image that doesn't already have one.
  (Use ``--overwrite`` to replace an existing label - DANGEROUS if the
  file was hand-edited.) Note: if the model produces zero detections
  for an image, the resulting empty ``.txt`` file is still treated as
  "already labeled" on subsequent runs - pass ``--overwrite`` to
  re-process those images after retraining a better model.
- Per-class confidence thresholds: ``area_*`` (always 6 boxes per frame)
  use a higher threshold than ``dice_*`` (rare) by default.
- Drops any per-class detections that exceed
  ``EXPECTED_INSTANCES_PER_FRAME`` - keeps the top-N highest-conf boxes.
- Annotated PNG previews can optionally be exported alongside, so you can
  spot-check before importing.

Usage::

    # Pre-label every image in dataset/images/train that has no label yet.
    python tools/semi_auto_label.py --weights runs/detect/xocdia/weights/best.pt

    # Same, but also export annotated previews for QA:
    python tools/semi_auto_label.py --weights .../best.pt --preview-dir qa_preview/auto

    # Custom split / folder.
    python tools/semi_auto_label.py --weights .../best.pt --split val
    python tools/semi_auto_label.py --weights .../best.pt \\
        --images-folder rounds/png --labels-folder rounds/auto_labels

After running, open the labels in the standard GUI to fix mistakes::

    python label_tool.py --images-folder dataset/images/train \\
                        --labels-folder dataset/labels/train

The label tool will treat the auto-generated boxes as a normal starting
point - you can move/resize/delete them and add the ones the model
missed (commonly fine OCR cells like ``total_count_cell``).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classes import (
    CLASS_GROUPS,
    CLASSES,
    COLORS,
    EXPECTED_INSTANCES_PER_FRAME,
    category_of,
)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# Per-category default confidence thresholds. Tuned conservatively: areas
# have high recall on a healthy model, dice are rare so we lean lower so
# the 1 box that exists isn't dropped. Override with --conf or
# --per-category-conf "<cat>=<value>" pairs.
DEFAULT_CONF_BY_CATEGORY = {
    "state": 0.45,    # timer
    "area":  0.50,    # 6 stable boxes per frame
    "dice":  0.30,    # rare, prefer recall
    "cell":  0.35,    # percent / total_bet / total_count
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the trained YOLO weights (e.g. runs/detect/xocdia/weights/best.pt).",
    )
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--split", choices=["train", "val", "raw"], default="train")
    parser.add_argument("--images-folder", default=None)
    parser.add_argument("--labels-folder", default=None)
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Single confidence threshold applied to every class. Overrides "
             "per-category defaults if given.",
    )
    parser.add_argument(
        "--per-category-conf",
        nargs="*",
        default=[],
        metavar="CAT=VAL",
        help="Per-category overrides, e.g. 'dice=0.25 area=0.55'. "
             "Categories: state, area, dice, cell.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (passed to YOLO).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device, e.g. 'cuda:0' or 'cpu'. Default: auto.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files. By default the script skips "
             "images that already have a label - this is the safe default. "
             "NOTE: a 0-byte .txt (no model detections) also counts as "
             "'already labeled'; use --overwrite to retry those.",
    )
    parser.add_argument(
        "--preview-dir",
        default=None,
        help="If set, also save annotated PNG copies here so you can QA "
             "before opening the label tool.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many images (handy for a smoke test).",
    )
    return parser.parse_args()


def resolve_conf_thresholds(args) -> Dict[str, float]:
    if args.conf is not None:
        if args.per_category_conf:
            print(
                "[WARN] --conf is set; --per-category-conf values are "
                "ignored. Use only --per-category-conf if you want "
                "per-category overrides."
            )
        return {cat: args.conf for cat, _ in CLASS_GROUPS}
    thresholds = dict(DEFAULT_CONF_BY_CATEGORY)
    for tok in args.per_category_conf:
        if "=" not in tok:
            raise SystemExit(f"Bad --per-category-conf token: {tok!r} (need CAT=VAL)")
        cat, val = tok.split("=", 1)
        if cat not in thresholds:
            raise SystemExit(f"Unknown category: {cat}. Pick from {list(thresholds)}.")
        thresholds[cat] = float(val)
    return thresholds


def collect_images(folder: Path) -> List[Path]:
    images: List[Path] = []
    for ext in IMAGE_EXTS:
        images.extend(folder.glob(f"*{ext}"))
    return sorted(images)


def cap_per_class(
    detections: List[Tuple[int, float, Tuple[int, int, int, int]]],
) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
    """Keep only the top-N detections per class according to
    EXPECTED_INSTANCES_PER_FRAME's max value. Sorted by descending conf.
    """
    by_cls: Dict[int, List[Tuple[int, float, Tuple[int, int, int, int]]]] = {}
    for cid, cf, box in detections:
        by_cls.setdefault(cid, []).append((cid, cf, box))

    capped: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
    for cid, items in by_cls.items():
        items.sort(key=lambda x: x[1], reverse=True)
        max_n = EXPECTED_INSTANCES_PER_FRAME.get(cid, (0, 1))[1]
        capped.extend(items[:max_n])
    return capped


def to_yolo_line(
    cid: int, box: Tuple[int, int, int, int], img_w: int, img_h: int,
) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def annotate(img, dets):
    out = img.copy()
    for cid, cf, (x1, y1, x2, y2) in dets:
        color = COLORS.get(cid, (200, 200, 200))
        name = CLASSES.get(cid, f"cls{cid}")
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{name} {cf:.2f}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights file not found: {weights}")

    images_folder = Path(
        args.images_folder or f"{args.dataset_root}/images/{args.split}"
    )
    labels_folder = Path(
        args.labels_folder or f"{args.dataset_root}/labels/{args.split}"
    )
    if not images_folder.exists():
        raise SystemExit(f"Images folder not found: {images_folder}")
    labels_folder.mkdir(parents=True, exist_ok=True)

    images = collect_images(images_folder)
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        print(f"No images in {images_folder}")
        return

    conf_by_cat = resolve_conf_thresholds(args)
    # Pull the lowest threshold to use as the model-level conf, then we
    # filter per-class ourselves (so YOLO doesn't reject a 0.3 dice box).
    model_conf = min(conf_by_cat.values())

    print(f"Loading weights: {weights}")
    # Imported lazily so importing this file (e.g. for tests) doesn't
    # require the heavy ultralytics stack.
    from ultralytics import YOLO  # noqa: WPS433
    model = YOLO(str(weights))

    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    total_boxes = 0
    for img_path in images:
        label_path = labels_folder / f"{img_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip unreadable] {img_path.name}")
            continue
        h, w = img.shape[:2]

        results = model(
            img,
            conf=model_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        raw = []
        if results.boxes is not None:
            xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cid, cf in zip(xyxy, cls_ids, confs):
                cid = int(cid)
                cat = category_of(cid)
                if float(cf) < conf_by_cat.get(cat, 0.5):
                    continue
                raw.append((cid, float(cf), (int(x1), int(y1), int(x2), int(y2))))

        kept = cap_per_class(raw)

        # Write YOLO label file (or empty file if nothing passed).
        lines = [to_yolo_line(cid, box, w, h) for cid, _cf, box in kept]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        if preview_dir:
            cv2.imwrite(str(preview_dir / img_path.name), annotate(img, kept))

        written += 1
        total_boxes += len(kept)
        print(f"  {img_path.name}: {len(kept)} boxes "
              f"(raw={len(raw)}, conf cutoffs={conf_by_cat})")

    print()
    print("=== Semi-auto label summary ===")
    print(f"Images processed   : {written}")
    print(f"Images skipped     : {skipped}  (already labeled; use --overwrite to redo)")
    print(f"Total boxes written: {total_boxes}")
    if preview_dir:
        print(f"Annotated previews : {preview_dir}")
    print()
    print("Next: open label_tool.py and refine the auto-generated boxes:")
    print(f"  python label_tool.py --images-folder {images_folder} "
          f"--labels-folder {labels_folder}")


if __name__ == "__main__":
    main()
