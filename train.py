"""Train the single-stage Xoc Dia YOLO detector (17 classes).

Defaults are tuned for UI detection:
- fliplr/flipud/degrees = 0 (UI is always upright, never mirrored)
- mosaic = 0.5 (keep UI layout intact in most tiles)
- hsv_h small, hsv_s/hsv_v moderate (UI colors are fixed, handle dimming)
- imgsz = 800 (enough headroom for small text cells)
- optimizer = "auto" (let Ultralytics pick; good for typical dataset sizes)

Use ``--resume path/to/last.pt`` to resume an interrupted run.
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="xocdia.yaml")
    parser.add_argument(
        "--weights",
        default="yolov8s.pt",
        help="Start weights. Ultralytics will auto-download if not present.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a last.pt checkpoint to resume (overrides --weights).",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="xocdia")
    parser.add_argument("--device", default=None, help="'0', '0,1', 'cpu', or None.")
    parser.add_argument(
        "--optimizer",
        default="auto",
        choices=["auto", "SGD", "Adam", "AdamW"],
        help="Optimizer (default: auto lets Ultralytics choose).",
    )
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
        print(results)
        return

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,

        # Optimizer
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Game-UI specific augmentation
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=0.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,

        # Checkpointing / early stop
        patience=args.patience,
        save=True,
        save_period=args.save_period,
        cache=False,
    )
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print(results)


if __name__ == "__main__":
    main()
