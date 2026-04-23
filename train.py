import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train/fine-tune stage-1 YOLO model.")
    parser.add_argument("--data", default="xocdia.yaml")
    parser.add_argument("--weights", default="yolov8n.pt", help="Start/fine-tune weights.")
    parser.add_argument("--resume", default=None, help="Resume from run checkpoint (usually last.pt).")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="train")
    parser.add_argument("--project", default="runs/detect")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model = YOLO(args.weights)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            close_mosaic=10,
            copy_paste=0.1,
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            mosaic=1.0,
            fliplr=0.5,
        )
    print(results)


if __name__ == "__main__":
    main()
