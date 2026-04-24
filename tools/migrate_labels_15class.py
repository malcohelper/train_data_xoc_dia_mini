"""Migrate YOLO label files from the 17-class schema to the 15-class one.

Old schema (17):
    0 round_id, 1 timer, 2 new_round,
    3 area_chan, 4 area_le, 5 area_4_red,
    6 area_3w_1r, 7 area_3r_1w, 8 area_4_white,
    9 dice_4r, 10 dice_4w, 11 dice_3w1r, 12 dice_3r1w, 13 dice_2w2r,
    14 percent_cell, 15 total_bet_cell, 16 total_count_cell

New schema (15):
    0 timer,
    1 area_chan, 2 area_le, 3 area_4_red,
    4 area_3w_1r, 5 area_3r_1w, 6 area_4_white,
    7 dice_4r, 8 dice_4w, 9 dice_3w1r, 10 dice_3r1w, 11 dice_2w2r,
    12 percent_cell, 13 total_bet_cell, 14 total_count_cell

Classes 0 (round_id) and 2 (new_round) from the old schema are DROPPED.
All other classes are shifted down to fill the gaps.

Usage::

    python tools/migrate_labels_15class.py                # dry-run
    python tools/migrate_labels_15class.py --apply        # write changes
    python tools/migrate_labels_15class.py --apply --labels-folder dataset/labels
"""

import argparse
from collections import defaultdict
from pathlib import Path

# Dropped class IDs in the OLD schema.
DROP = {0, 2}

# Remap: OLD class id -> NEW class id (after dropping).
REMAP = {
    1: 0,    # timer
    3: 1,    # area_chan
    4: 2,    # area_le
    5: 3,    # area_4_red
    6: 4,    # area_3w_1r
    7: 5,    # area_3r_1w
    8: 6,    # area_4_white
    9: 7,    # dice_4r
    10: 8,   # dice_4w
    11: 9,   # dice_3w1r
    12: 10,  # dice_3r1w
    13: 11,  # dice_2w2r
    14: 12,  # percent_cell
    15: 13,  # total_bet_cell
    16: 14,  # total_count_cell
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-folder",
        default="dataset/labels",
        help="Root folder containing train/ and val/ subdirs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in place. Without this flag the script only reports.",
    )
    return parser.parse_args()


def migrate_file(path: Path, apply: bool) -> dict:
    stats = {"dropped": 0, "remapped": 0, "kept_lines": 0, "unknown": 0}
    lines = path.read_text().splitlines()
    out_lines = []
    for raw in lines:
        parts = raw.strip().split()
        if len(parts) != 5:
            # keep malformed lines untouched so the user can inspect them
            out_lines.append(raw)
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            out_lines.append(raw)
            stats["unknown"] += 1
            continue
        if cid in DROP:
            stats["dropped"] += 1
            continue
        if cid in REMAP:
            parts[0] = str(REMAP[cid])
            out_lines.append(" ".join(parts))
            stats["remapped"] += 1
            stats["kept_lines"] += 1
            continue
        # Unknown class id (out of old schema) -> drop with warning.
        stats["unknown"] += 1

    if apply:
        # Trailing newline kept for consistency with label_tool.save_labels.
        path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return stats


def main() -> int:
    args = parse_args()
    root = Path(args.labels_folder)
    if not root.exists():
        print(f"Labels folder not found: {root}")
        return 1

    files = sorted(root.rglob("*.txt"))
    if not files:
        print(f"No .txt label files under {root}")
        return 1

    total = defaultdict(int)
    changed_files = 0
    for f in files:
        stats = migrate_file(f, apply=args.apply)
        for k, v in stats.items():
            total[k] += v
        if stats["dropped"] or stats["remapped"] or stats["unknown"]:
            changed_files += 1

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"=== Label migration ({mode}) ===")
    print(f"Root              : {root}")
    print(f"Files scanned     : {len(files)}")
    print(f"Files affected    : {changed_files}")
    print(f"Boxes remapped    : {total['remapped']}")
    print(f"Boxes dropped     : {total['dropped']}  (class 0 round_id + class 2 new_round)")
    print(f"Unknown/malformed : {total['unknown']}")
    if not args.apply:
        print("\nRun again with --apply to write changes in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
