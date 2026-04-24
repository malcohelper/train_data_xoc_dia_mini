"""QA a YOLO-format labeled dataset.

Stats printed:
- Box count per class + number of images containing each class.
- Images with zero / very few boxes.
- Labels with invalid class IDs (outside the 15-class schema).
- Labels with out-of-range coords (not in [0, 1]).
- Classes with obvious imbalance (< 10% of the busiest class).
- Frames that deviate from EXPECTED_INSTANCES_PER_FRAME.

Usage::

    python tools/check_labels.py                     # dataset/labels/train
    python tools/check_labels.py --split val
    python tools/check_labels.py --labels-folder custom/lbls
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classes import CLASSES, EXPECTED_INSTANCES_PER_FRAME


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--labels-folder", default=None)
    parser.add_argument(
        "--min-boxes-per-image",
        type=int,
        default=6,
        help="Warn if an image has fewer than this many boxes.",
    )
    parser.add_argument(
        "--no-expected-check",
        action="store_true",
        help="Skip the per-class-per-frame expected-range check.",
    )
    return parser.parse_args()


def check_labels(labels_folder: Path, min_boxes: int, check_expected: bool):
    files = sorted(labels_folder.glob("*.txt"))
    if not files:
        print(f"No label files found in {labels_folder}")
        return 1

    box_per_class = defaultdict(int)
    images_per_class = defaultdict(int)
    invalid_class_files = []
    bad_coord_files = []
    zero_box_files = []
    few_box_files = []
    expected_violations = []

    for lf in files:
        lines = [ln for ln in lf.read_text().splitlines() if ln.strip()]

        if not lines:
            zero_box_files.append(lf)
            continue

        per_class_in_frame = defaultdict(int)
        has_bad = False
        has_bad_coord = False

        for raw in lines:
            parts = raw.strip().split()
            if len(parts) != 5:
                has_bad = True
                continue
            try:
                cid = int(parts[0])
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                has_bad = True
                continue
            if cid not in CLASSES:
                has_bad = True
                continue
            if any((c < 0.0 or c > 1.0) for c in coords):
                has_bad_coord = True
                continue

            box_per_class[cid] += 1
            per_class_in_frame[cid] += 1

        for cid in per_class_in_frame:
            images_per_class[cid] += 1

        if has_bad:
            invalid_class_files.append(lf)
        if has_bad_coord:
            bad_coord_files.append(lf)

        total_valid = sum(per_class_in_frame.values())
        if total_valid == 0:
            zero_box_files.append(lf)
        elif total_valid < min_boxes:
            few_box_files.append((lf, total_valid))

        if check_expected:
            for cid, (lo, hi) in EXPECTED_INSTANCES_PER_FRAME.items():
                count = per_class_in_frame.get(cid, 0)
                if count < lo or count > hi:
                    expected_violations.append((lf, cid, count, lo, hi))

    total_files = len(files)
    total_boxes = sum(box_per_class.values())

    print(f"Labels folder : {labels_folder}")
    print(f"Label files   : {total_files}")
    print(f"Total boxes   : {total_boxes}")

    print("\n=== Boxes per class ===")
    print(f"{'id':>3}  {'name':<20} {'boxes':>6} {'images':>7}")
    busiest = max(box_per_class.values()) if box_per_class else 0
    warned_imbalance = []
    for cid, name in CLASSES.items():
        b = box_per_class.get(cid, 0)
        i = images_per_class.get(cid, 0)
        print(f"{cid:>3}  {name:<20} {b:>6} {i:>7}")
        if busiest and b > 0 and b < 0.1 * busiest:
            warned_imbalance.append((cid, name, b, busiest))
        if b == 0:
            warned_imbalance.append((cid, name, 0, busiest))

    if warned_imbalance:
        print("\n[WARN] Class imbalance (either 0 or <10% of busiest class):")
        for cid, name, b, bu in warned_imbalance:
            print(f"  - {cid}:{name}  boxes={b}  busiest={bu}")

    if zero_box_files:
        print(f"\n[WARN] {len(zero_box_files)} file(s) with 0 boxes (showing up to 10):")
        for f in zero_box_files[:10]:
            print(f"  - {f.name}")

    if few_box_files:
        print(f"\n[WARN] {len(few_box_files)} file(s) with < {min_boxes} boxes (showing up to 10):")
        for f, n in few_box_files[:10]:
            print(f"  - {f.name} ({n} boxes)")

    if invalid_class_files:
        print(f"\n[ERROR] {len(invalid_class_files)} file(s) with invalid/malformed lines (showing up to 10):")
        for f in invalid_class_files[:10]:
            print(f"  - {f.name}")

    if bad_coord_files:
        print(f"\n[ERROR] {len(bad_coord_files)} file(s) with out-of-range coords (showing up to 10):")
        for f in bad_coord_files[:10]:
            print(f"  - {f.name}")

    if expected_violations:
        print(
            f"\n[INFO] {len(expected_violations)} frame/class pair(s) outside expected instance range"
            " (showing up to 20)."
            " Range is from classes.EXPECTED_INSTANCES_PER_FRAME and is a soft heuristic."
        )
        for f, cid, n, lo, hi in expected_violations[:20]:
            name = CLASSES.get(cid, f"cls{cid}")
            print(f"  - {f.name}  {cid}:{name} count={n} expected=[{lo},{hi}]")

    has_errors = bool(invalid_class_files) or bool(bad_coord_files)
    return 2 if has_errors else 0


def main():
    args = parse_args()
    labels_folder = Path(
        args.labels_folder or f"{args.dataset_root}/labels/{args.split}"
    )
    rc = check_labels(
        labels_folder,
        min_boxes=args.min_boxes_per_image,
        check_expected=not args.no_expected_check,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
