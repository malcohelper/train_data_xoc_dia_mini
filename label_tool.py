import argparse
from pathlib import Path

import cv2


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class LabelTool:
    def __init__(
        self,
        images_folder="dataset/images/train",
        labels_folder="dataset/labels/train",
        only_unlabeled=False,
        autosave=True,
        auto_next_class=True,
    ):
        self.images_folder = Path(images_folder)
        self.labels_folder = Path(labels_folder)
        self.labels_folder.mkdir(parents=True, exist_ok=True)

        self.classes = {
            0: "round_id",
            1: "timer",
            2: "area_chan",
            3: "area_le",
            4: "area_4_red",
            5: "area_3w_1r",
            6: "area_3r_1w",
            7: "area_4_white",
            8: "new_round",
            9: "4r",
            10: "4w",
            11: "3w1r",
            12: "3r1w",
            13: "2w2r",
        }
        self.colors = {
            0: (0, 255, 255),
            1: (0, 220, 255),
            2: (0, 165, 255),
            3: (0, 255, 0),
            4: (255, 255, 0),
            5: (255, 0, 0),
            6: (255, 0, 255),
            7: (128, 0, 128),
            8: (60, 180, 255),
            9: (0, 0, 255),
            10: (220, 220, 220),
            11: (255, 80, 80),
            12: (255, 120, 200),
            13: (180, 180, 0),
        }

        self.only_unlabeled = only_unlabeled
        self.autosave = autosave
        self.auto_next_class = auto_next_class
        self.current_class = 0
        self.boxes = []
        self.drawing = False
        self.start_point = None
        self.temp_point = None
        self.window_name = "Label Tool"
        self.current_img = None
        self.current_img_path = None
        self.current_index = 0

        self.images = self.collect_images()
        if self.only_unlabeled:
            self.images = [p for p in self.images if not self.get_label_path(p).exists()]

    def collect_images(self):
        images = []
        for ext in IMAGE_EXTS:
            images.extend(self.images_folder.glob(f"*{ext}"))
        return sorted(images)

    def get_label_path(self, img_path: Path):
        return self.labels_folder / f"{img_path.stem}.txt"

    def load_labels(self, label_path: Path):
        loaded = []
        if not label_path.exists():
            return loaded
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in self.classes:
                continue
            loaded.append({"class": cls_id, "bbox": [float(x) for x in parts[1:]]})
        return loaded

    def save_labels(self, label_path: Path):
        lines = []
        for box in self.boxes:
            line = f"{box['class']} " + " ".join(f"{x:.6f}" for x in box["bbox"])
            lines.append(line)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"Saved: {label_path.name} ({len(self.boxes)} boxes)")

    def copy_from_previous(self):
        if self.current_index <= 0:
            print("No previous image to copy from.")
            return
        prev_label = self.get_label_path(self.images[self.current_index - 1])
        if not prev_label.exists():
            print("Previous image has no label.")
            return
        self.boxes = self.load_labels(prev_label)
        print(f"Copied {len(self.boxes)} boxes from {prev_label.name}")

    def mouse_callback(self, event, x, y, _flags, _param):
        if self.current_img is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.temp_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.temp_point = (x, y)
            self.add_box_from_points(self.start_point, self.temp_point)
            self.start_point = None
            self.temp_point = None

    def add_box_from_points(self, p1, p2):
        if self.current_img is None:
            return
        x1, y1 = p1
        x2, y2 = p2
        if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
            return

        h, w = self.current_img.shape[:2]
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = abs(x2 - x1) / w
        height = abs(y2 - y1) / h

        self.boxes.append(
            {
                "class": self.current_class,
                "bbox": [x_center, y_center, width, height],
            }
        )
        print(f"Added: {self.classes[self.current_class]} | total={len(self.boxes)}")
        if self.auto_next_class:
            self.current_class = (self.current_class + 1) % len(self.classes)
            print(f"Auto-next class -> {self.current_class}:{self.classes[self.current_class]}")

    def draw_boxes(self, frame):
        h, w = frame.shape[:2]
        for box in self.boxes:
            cx, cy, bw, bh = box["bbox"]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cls_id = box["class"]
            color = self.colors.get(cls_id, (200, 200, 200))
            thickness = 3 if cls_id == self.current_class else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                self.classes[cls_id],
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        if self.drawing and self.start_point and self.temp_point:
            cv2.rectangle(frame, self.start_point, self.temp_point, (255, 255, 255), 1)

    def draw_status(self, frame):
        idx = self.current_index + 1
        total = len(self.images)
        title = f"[{idx}/{total}] {self.current_img_path.name} | class={self.current_class}:{self.classes[self.current_class]}"
        cv2.putText(frame, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        controls = "0-9 class | j/k prev/next class | t auto-next | u undo | x clear class | c copy prev | p prev | space next | s save | a autosave | q quit"
        cv2.putText(frame, controls, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        status = (
            f"autosave={'ON' if self.autosave else 'OFF'} | "
            f"auto-next-class={'ON' if self.auto_next_class else 'OFF'} | "
            f"boxes={len(self.boxes)}"
        )
        cv2.putText(frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1)

        y = 92
        class_counts = {k: 0 for k in self.classes}
        for box in self.boxes:
            class_counts[box["class"]] += 1
        summary = " ".join(f"{k}:{class_counts[k]}" for k in self.classes)
        cv2.putText(frame, summary, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (220, 220, 220), 1)

    def handle_key(self, key):
        label_path = self.get_label_path(self.current_img_path)

        if ord("0") <= key <= ord("9"):
            self.current_class = key - ord("0")
            print(f"Class -> {self.current_class}:{self.classes[self.current_class]}")
            return "stay"

        if key == ord("j"):
            self.current_class = (self.current_class - 1) % len(self.classes)
            print(f"Class -> {self.current_class}:{self.classes[self.current_class]}")
            return "stay"

        if key == ord("k"):
            self.current_class = (self.current_class + 1) % len(self.classes)
            print(f"Class -> {self.current_class}:{self.classes[self.current_class]}")
            return "stay"

        if key == ord("u") and self.boxes:
            self.boxes.pop()
            print("Undo last box")
            return "stay"

        if key == ord("x"):
            before = len(self.boxes)
            self.boxes = [b for b in self.boxes if b["class"] != self.current_class]
            print(f"Cleared class {self.current_class}: removed {before - len(self.boxes)} boxes")
            return "stay"

        if key == ord("c"):
            self.copy_from_previous()
            return "stay"

        if key == ord("a"):
            self.autosave = not self.autosave
            print(f"Autosave -> {'ON' if self.autosave else 'OFF'}")
            return "stay"

        if key == ord("t"):
            self.auto_next_class = not self.auto_next_class
            print(f"Auto-next class -> {'ON' if self.auto_next_class else 'OFF'}")
            return "stay"

        if key == ord("s"):
            self.save_labels(label_path)
            return "stay"

        if key == ord("p"):
            if self.autosave:
                self.save_labels(label_path)
            return "prev"

        if key == ord(" ") or key == ord("n"):
            if self.autosave:
                self.save_labels(label_path)
            return "next"

        if key == ord("q"):
            if self.autosave:
                self.save_labels(label_path)
            return "quit"

        return "stay"

    def run_single_image(self, img_path: Path):
        self.current_img_path = img_path
        self.current_img = cv2.imread(str(img_path))
        if self.current_img is None:
            print(f"Skip unreadable image: {img_path}")
            return "next"

        self.boxes = self.load_labels(self.get_label_path(img_path))
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            display = self.current_img.copy()
            self.draw_boxes(display)
            self.draw_status(display)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(16) & 0xFF
            action = self.handle_key(key)
            if action in {"next", "prev", "quit"}:
                return action

    def label_images(self):
        if not self.images:
            print(f"No images found in {self.images_folder}")
            return

        print(f"Images folder: {self.images_folder}")
        print(f"Labels folder: {self.labels_folder}")
        print(f"Total images: {len(self.images)}")

        while 0 <= self.current_index < len(self.images):
            action = self.run_single_image(self.images[self.current_index])
            if action == "next":
                self.current_index += 1
            elif action == "prev":
                self.current_index = max(0, self.current_index - 1)
            elif action == "quit":
                break

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Fast YOLO label tool for train/val.")
    parser.add_argument("--dataset-root", default="dataset", help="Dataset root folder.")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Label split.")
    parser.add_argument("--images-folder", default=None, help="Custom images folder.")
    parser.add_argument("--labels-folder", default=None, help="Custom labels folder.")
    parser.add_argument(
        "--only-unlabeled",
        action="store_true",
        help="Open only images that do not have label file yet.",
    )
    parser.add_argument(
        "--no-autosave",
        action="store_true",
        help="Disable autosave when changing image.",
    )
    parser.add_argument(
        "--no-auto-next-class",
        action="store_true",
        help="Disable auto jump to next class after each drawn box.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images_folder = args.images_folder or f"{args.dataset_root}/images/{args.split}"
    labels_folder = args.labels_folder or f"{args.dataset_root}/labels/{args.split}"

    tool = LabelTool(
        images_folder=images_folder,
        labels_folder=labels_folder,
        only_unlabeled=args.only_unlabeled,
        autosave=not args.no_autosave,
        auto_next_class=not args.no_auto_next_class,
    )
    tool.label_images()