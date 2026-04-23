"""Visualize YOLO labels on top of their images.

Reads labels (YOLO TXT format) from a labels folder and draws them onto the
matching image. Useful to QA a freshly labeled dataset before training.

Usage:
    python tools/visualize.py                              # dataset/images/train + dataset/labels/train
    python tools/visualize.py --split val
    python tools/visualize.py --images-folder custom/imgs --labels-folder custom/lbls
    python tools/visualize.py --save-dir qa_preview        # batch-export annotated PNGs

Interactive keys (when not using --save-dir):
    space / n   next image
    p           previous image
    q / ESC     quit
"""

import argparse
from pathlib import Path

import cv2

from label_tool import CLASSES, COLORS


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--images-folder", default=None)
    parser.add_argument("--labels-folder", default=None)
    parser.add_argument(
        "--save-dir",
        default=None,
        help="If set, write annotated copies here and exit (no GUI).",
    )
    parser.add_argument(
        "--only-labeled",
        action="store_true",
        help="Skip images that have no matching label file.",
    )
    return parser.parse_args()


def collect_images(images_folder: Path):
    images = []
    for ext in IMAGE_EXTS:
        images.extend(images_folder.glob(f"*{ext}"))
    return sorted(images)


def draw_labels(img, label_path: Path) -> int:
    if not label_path.exists():
        return 0
    h, w = img.shape[:2]
    count = 0
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, bw, bh = (float(p) for p in parts[1:])
        except ValueError:
            continue

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = COLORS.get(cls_id, (200, 200, 200))
        name = CLASSES.get(cls_id, f"cls{cls_id}")

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{cls_id}:{name}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        count += 1
    return count


def main():
    args = parse_args()
    images_folder = Path(
        args.images_folder or f"{args.dataset_root}/images/{args.split}"
    )
    labels_folder = Path(
        args.labels_folder or f"{args.dataset_root}/labels/{args.split}"
    )

    images = collect_images(images_folder)
    if args.only_labeled:
        images = [p for p in images if (labels_folder / f"{p.stem}.txt").exists()]

    if not images:
        print(f"No images found in {images_folder}")
        return

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            n = draw_labels(img, labels_folder / f"{img_path.stem}.txt")
            out = save_dir / img_path.name
            cv2.imwrite(str(out), img)
            print(f"Wrote {out} ({n} boxes)")
        print(f"Done. Exported {len(images)} image(s) to {save_dir}.")
        return

    window = "Label Preview"
    cv2.namedWindow(window)
    idx = 0
    while 0 <= idx < len(images):
        img_path = images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue
        n = draw_labels(img, labels_folder / f"{img_path.stem}.txt")
        title = f"[{idx+1}/{len(images)}] {img_path.name} | {n} boxes"
        cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window, img)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            idx = max(0, idx - 1)
        else:
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
