"""Highlight a chosen subset of YOLO label classes on the image.

Complements ``tools/visualize.py``: visualize.py renders ALL boxes for QA;
cell_preview narrows down to one or a few classes (e.g. only
``total_bet_cell``) so the operator can quickly answer questions like
"are my total_bet boxes tight enough?" without visual noise from area
and percent boxes covering the same region.

Usage::

    # Preview only total_bet_cell across the train split, GUI mode.
    python tools/cell_preview.py --classes total_bet_cell

    # Preview a few class ids, batch-export annotated PNGs.
    python tools/cell_preview.py --classes 13 14 --save-dir qa_preview/cells

    # Preview a whole semantic group (state / area / dice / cell).
    python tools/cell_preview.py --group cell

    # Cell preview against the val split.
    python tools/cell_preview.py --classes percent_cell --split val

Interactive keys (when not using --save-dir):
    space / n   next image
    p           previous image
    f           toggle dimming of non-target classes (full-frame vs solo)
    q / ESC     quit

Selection precedence: ``--classes`` overrides ``--group``. Class tokens
can be class_ids (``13``) or class_names (``total_bet_cell``).
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Set

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classes import CLASSES, CLASS_GROUPS, CLASS_NAME_TO_ID, COLORS


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--images-folder", default=None)
    parser.add_argument("--labels-folder", default=None)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="One or more class ids or class names to highlight (e.g. "
             "'13 14' or 'total_bet_cell total_count_cell').",
    )
    parser.add_argument(
        "--group",
        choices=[g[0] for g in CLASS_GROUPS],
        default=None,
        help="Highlight a whole semantic group (state/area/dice/cell). "
             "Ignored if --classes is given.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="If set, write annotated copies here and exit (no GUI).",
    )
    parser.add_argument(
        "--only-with-target",
        action="store_true",
        help="Skip images whose label file has no boxes of the selected "
             "classes (best for quickly auditing rare classes like dice_4w).",
    )
    parser.add_argument(
        "--no-dim",
        action="store_true",
        help="Don't dim the non-target classes - just draw target boxes "
             "on the original image.",
    )
    return parser.parse_args()


def resolve_class_ids(args) -> Set[int]:
    if args.classes:
        ids: Set[int] = set()
        for tok in args.classes:
            if tok.isdigit():
                cid = int(tok)
                if cid not in CLASSES:
                    raise SystemExit(f"Unknown class id: {cid}")
                ids.add(cid)
            elif tok in CLASS_NAME_TO_ID:
                ids.add(CLASS_NAME_TO_ID[tok])
            else:
                raise SystemExit(
                    f"Unknown class token: {tok!r}. Use one of: "
                    f"{sorted(CLASSES.values())}"
                )
        return ids

    if args.group:
        for name, ids in CLASS_GROUPS:
            if name == args.group:
                return set(ids)

    raise SystemExit("Specify --classes or --group.")


def collect_images(folder: Path) -> List[Path]:
    images: List[Path] = []
    for ext in IMAGE_EXTS:
        images.extend(folder.glob(f"*{ext}"))
    return sorted(images)


def parse_label_file(label_path: Path) -> List[tuple]:
    """Return list of (class_id, cx, cy, bw, bh) from a YOLO label file."""
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, bw, bh = (float(p) for p in parts[1:])
        except ValueError:
            continue
        rows.append((cid, cx, cy, bw, bh))
    return rows


def render(
    img: np.ndarray,
    rows: Iterable[tuple],
    target_ids: Set[int],
    dim_others: bool,
) -> tuple:
    """Render highlighted boxes; return (annotated_img, n_target, n_total)."""
    h, w = img.shape[:2]

    # Optionally dim the whole frame, then draw target boxes on top in
    # full color so they really pop.
    if dim_others:
        out = (img * 0.35).astype(np.uint8)
    else:
        out = img.copy()

    n_target = 0
    n_total = 0
    for cid, cx, cy, bw, bh in rows:
        n_total += 1
        if cid not in target_ids:
            # Optionally draw a faint outline so the operator still has
            # spatial context.
            if dim_others:
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(out, (x1, y1), (x2, y2), (80, 80, 80), 1)
            continue

        n_target += 1
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = COLORS.get(cid, (0, 255, 255))
        name = CLASSES.get(cid, f"cls{cid}")
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out, f"{cid}:{name}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
    return out, n_target, n_total


def main():
    args = parse_args()
    target_ids = resolve_class_ids(args)
    target_names = sorted(CLASSES[i] for i in target_ids)

    images_folder = Path(
        args.images_folder or f"{args.dataset_root}/images/{args.split}"
    )
    labels_folder = Path(
        args.labels_folder or f"{args.dataset_root}/labels/{args.split}"
    )
    if not images_folder.exists():
        raise SystemExit(f"Images folder not found: {images_folder}")

    images = collect_images(images_folder)
    if args.only_with_target:
        kept = []
        for p in images:
            rows = parse_label_file(labels_folder / f"{p.stem}.txt")
            if any(cid in target_ids for cid, *_ in rows):
                kept.append(p)
        images = kept

    if not images:
        print(
            f"No images to preview. (folder={images_folder}, "
            f"only_with_target={args.only_with_target}, target={target_names})"
        )
        return

    print(
        f"Preview target classes: {target_names} "
        f"({len(images)} image(s), labels={labels_folder})"
    )

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in images:
            img = cv2.imread(str(p))
            if img is None:
                continue
            rows = parse_label_file(labels_folder / f"{p.stem}.txt")
            annotated, n_t, n_all = render(
                img, rows, target_ids, dim_others=not args.no_dim,
            )
            out = out_dir / p.name
            cv2.imwrite(str(out), annotated)
            print(f"  {out.name}: {n_t} target / {n_all} total")
        print(f"Done. Wrote {len(images)} image(s) to {out_dir}.")
        return

    window = "Cell Preview"
    cv2.namedWindow(window)
    idx = 0
    dim_others = not args.no_dim
    while 0 <= idx < len(images):
        p = images[idx]
        img = cv2.imread(str(p))
        if img is None:
            idx += 1
            continue
        rows = parse_label_file(labels_folder / f"{p.stem}.txt")
        annotated, n_t, n_all = render(img, rows, target_ids, dim_others)
        title = (
            f"[{idx+1}/{len(images)}] {p.name} | "
            f"target={n_t}/{n_all}  classes={target_names}"
            f"  ({'dimmed' if dim_others else 'plain'})"
        )
        cv2.putText(annotated, title, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window, annotated)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            idx = max(0, idx - 1)
        elif key == ord("f"):
            dim_others = not dim_others
        else:
            idx += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
