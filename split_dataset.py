import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset into train/val."
    )
    parser.add_argument(
        "--src-images",
        required=True,
        help="Source folder containing original images.",
    )
    parser.add_argument(
        "--src-labels",
        required=True,
        help="Source folder containing original YOLO labels (.txt).",
    )
    parser.add_argument(
        "--dst-root",
        default="dataset",
        help="Destination root folder (e.g. dataset or dataset_sub).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio, from 0 to 1. Default: 0.8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Copy or move files into train/val folders.",
    )
    return parser.parse_args()


def ensure_dirs(dst_root: Path):
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        (dst_root / rel).mkdir(parents=True, exist_ok=True)


def collect_pairs(src_images: Path, src_labels: Path):
    pairs = []
    missing_labels = []

    for img_path in sorted(src_images.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = src_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    return pairs, missing_labels


def transfer_file(src: Path, dst: Path, mode: str):
    if src.resolve() == dst.resolve():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def split_dataset(
    pairs,
    dst_root: Path,
    train_ratio: float,
    seed: int,
    mode: str,
):
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")

    random.seed(seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    for img_path, label_path in train_pairs:
        transfer_file(img_path, dst_root / "images" / "train" / img_path.name, mode)
        transfer_file(
            label_path,
            dst_root / "labels" / "train" / label_path.name,
            mode,
        )

    for img_path, label_path in val_pairs:
        transfer_file(img_path, dst_root / "images" / "val" / img_path.name, mode)
        transfer_file(
            label_path,
            dst_root / "labels" / "val" / label_path.name,
            mode,
        )

    return len(train_pairs), len(val_pairs)


def main():
    args = parse_args()
    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    dst_root = Path(args.dst_root)

    if not src_images.exists() or not src_images.is_dir():
        raise FileNotFoundError(f"Invalid --src-images folder: {src_images}")
    if not src_labels.exists() or not src_labels.is_dir():
        raise FileNotFoundError(f"Invalid --src-labels folder: {src_labels}")

    ensure_dirs(dst_root)
    pairs, missing_labels = collect_pairs(src_images, src_labels)

    if not pairs:
        raise RuntimeError("No valid image/label pairs found.")

    train_count, val_count = split_dataset(
        pairs=pairs,
        dst_root=dst_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        mode=args.mode,
    )

    print("Split completed.")
    print(f"- Total pairs: {len(pairs)}")
    print(f"- Train: {train_count}")
    print(f"- Val: {val_count}")
    print(f"- Missing labels: {len(missing_labels)}")
    if missing_labels:
        print("  Missing label files for:")
        for name in missing_labels[:20]:
            print(f"  - {name}")
        if len(missing_labels) > 20:
            print(f"  ... and {len(missing_labels) - 20} more")


if __name__ == "__main__":
    main()
