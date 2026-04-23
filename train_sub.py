import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train/fine-tune stage-2 YOLO model.")
    parser.add_argument("--data", default="xocdia_sub.yaml")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--resume", default=None, help="Resume from sub last.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--name", default="sub_train")
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
            augment=True,
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            translate=0.05,
            scale=0.3,
            mosaic=0.5,
        )
    print(results)


if __name__ == "__main__":
    main()
