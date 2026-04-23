"""Copy raw game captures from rounds/ into dataset/images/raw/.

The rounds/ folder stores PNG + JSON pairs produced by realtime_capture.py.
Only PNGs are used as YOLO training inputs; JSON metadata is preserved in
rounds/ for reference (round ID, timestamp, etc.).

Filenames starting with a stray single-quote (legacy files) are normalized
so they match the JSON stem. If a destination file already exists it is
skipped unless --overwrite is passed.

Usage:
    python tools/rounds_to_dataset.py                      # copy all PNGs
    python tools/rounds_to_dataset.py --limit 50           # first 50 files
    python tools/rounds_to_dataset.py --move               # move instead of copy
    python tools/rounds_to_dataset.py --dst dataset/images/raw
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="rounds", help="Source folder (default: rounds).")
    parser.add_argument(
        "--dst",
        default="dataset/images/raw",
        help="Destination folder (default: dataset/images/raw).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max files to transfer (0 = all).")
    parser.add_argument("--move", action="store_true", help="Move instead of copy.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    return parser.parse_args()


def normalize_stem(stem: str) -> str:
    # Some legacy captures have a stray leading apostrophe in the filename.
    return stem.lstrip("'")


def transfer(src: Path, dst: Path, move: bool):
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f"Source folder not found: {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(src_dir.glob("*.png"))
    if args.limit > 0:
        pngs = pngs[: args.limit]

    transferred = 0
    skipped = 0
    for src in pngs:
        dst = dst_dir / f"{normalize_stem(src.stem)}.png"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        transfer(src, dst, args.move)
        transferred += 1

    action = "Moved" if args.move else "Copied"
    print(f"{action} {transferred} file(s) into {dst_dir} (skipped {skipped}).")


if __name__ == "__main__":
    main()
