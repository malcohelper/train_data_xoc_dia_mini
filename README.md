# Xoc Dia Detector (17-class, single-stage)

Single-stage YOLOv8 + PaddleOCR pipeline for the Xoc Dia game UI. One YOLO
model detects all 17 UI elements (state / areas / dice / text cells), then a
small post-processing pass uses PaddleOCR to read the text inside each
detected cell and a geometric mapping step figures out which bet type each
cell belongs to.

## Pipeline

```
Frame ─► YOLOv8 (17 classes) ─► Group by category
                                ├─ state:   round_id, timer, new_round
                                ├─ area:    6 anchors (chan/le/4_red/...)
                                ├─ dice:    5 outcomes (4r/4w/3w1r/3r1w/2w2r)
                                └─ cell:    percent / total_bet / total_count
                                               │
                                               ▼
                                     PaddleOCR (read text)
                                               │
                                               ▼
                                 Geometric map cell → bet_type
                                               │
                                               ▼
                                        GameState dict
```

## Class schema (17 classes)

| Group | IDs | Classes |
|---|---|---|
| State | 0–2 | `round_id`, `timer`, `new_round` |
| Areas | 3–8 | `area_chan`, `area_le`, `area_4_red`, `area_3w_1r`, `area_3r_1w`, `area_4_white` |
| Dice  | 9–13 | `dice_4r`, `dice_4w`, `dice_3w1r`, `dice_3r1w`, `dice_2w2r` |
| Cells | 14–16 | `percent_cell`, `total_bet_cell`, `total_count_cell` |

Text cells use generic classes (6 instances per frame) and are assigned to
their owning bet type downstream via geometric containment against the
`area_*` anchors. Defined once in [`classes.py`](classes.py); everything
imports from there.

## Install

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install ultralytics opencv-python paddleocr paddlepaddle numpy mss pillow
# GPU-only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Repo layout

```
.
├── classes.py                # single source of truth for classes/colors/groups
├── xocdia.yaml               # YOLO data config (must match classes.py)
├── label_tool.py             # fast label GUI (17-class + text-input picker)
├── train.py                  # train YOLOv8 (game-UI tuned defaults)
├── detector.py               # lightweight XocDiaDetector API + CLI
├── ocr_engine.py             # PaddleOCR wrapper
├── pipeline.py               # GameAnalysisPipeline (detect + OCR + map)
├── realtime_capture.py       # screen-grab + live analysis + rounds/ dump
├── split_dataset.py          # 80/20 train/val split
├── dataset/
│   ├── images/{raw,train,val}
│   └── labels/{raw,train,val}
├── rounds/                   # per-round PNG + JSON snapshots
└── tools/
    ├── rounds_to_dataset.py  # copy rounds/*.png into dataset/images/raw/
    ├── visualize.py          # preview YOLO labels over images
    ├── check_labels.py       # label QA / imbalance warnings
    └── eval.py               # per-class mAP / P / R on the val split
```

## End-to-end workflow

### 1. Collect frames

```bash
# Live capture (writes rounds/<round_id>_<ts>.{png,json} on each new round).
python realtime_capture.py
```

Or copy already-captured PNGs from `rounds/` into the dataset raw folder:

```bash
python tools/rounds_to_dataset.py                     # copy all
python tools/rounds_to_dataset.py --limit 100         # first 100
python tools/rounds_to_dataset.py --move              # move instead of copy
```

### 2. Label

```bash
python label_tool.py                                  # dataset/images/train
python label_tool.py --images-folder dataset/images/raw \
                     --labels-folder dataset/labels/raw
```

Hotkeys:

| Key | Action |
|---|---|
| `0–9` | jump to class 0–9 |
| `/` then `NN` then `Enter` | jump to class N (0–16) via text input |
| `j` / `k` | prev / next class |
| `u` | undo last box |
| `x` | clear all boxes of current class |
| `c` | copy boxes from previous image |
| `s` | save | 
| `p` / `space` / `n` | prev / next image |
| `a` | toggle autosave |
| `t` | toggle auto-next-class after draw |
| `q` | quit |

QA the labels before spending GPU time on training:

```bash
python tools/check_labels.py                          # dataset/labels/train
python tools/check_labels.py --split val
python tools/visualize.py --split train               # interactive
python tools/visualize.py --split train --save-dir qa # batch export
```

### 3. Split into train / val

```bash
python split_dataset.py
```

### 4. Train

```bash
python train.py --epochs 150 --batch 16 --imgsz 800
python train.py --resume runs/detect/xocdia/weights/last.pt
```

Defaults are tuned for game UI (`fliplr=0`, `flipud=0`, `degrees=0`,
`mosaic=0.5`, `hsv_h=0.01`, `optimizer=auto`, `yolov8s.pt`, `imgsz=800`,
`patience=20`, `save_period=10`). All knobs exposed as CLI flags.

### 5. Evaluate

```bash
python tools/eval.py --weights runs/detect/xocdia/weights/best.pt
```

Prints overall mAP50 / mAP50-95 / P / R and a per-class table. Confusion
matrix + PR curves are written to `runs/detect/eval/`.

### 6. Run

```bash
# Offline: single image -> annotated image + JSON GameState
python pipeline.py --weights runs/detect/xocdia/weights/best.pt \
                   --source frame.png --save-annotated annotated.png

# Detector only (no OCR / no state inference)
python detector.py --weights runs/detect/xocdia/weights/best.pt --source frame.png

# Real-time screen capture + analyze
python realtime_capture.py --weights runs/detect/xocdia/weights/best.pt
```

## Dataset sizing

- Minimum workable: ~80–100 labeled frames covering all phases (betting,
  bowl-opens, new-round banner).
- Comfortable: 150–250 frames. Text cells benefit from variety of values
  (different `xxM` amounts, different % splits).
- Keep ~20% for val. Use `split_dataset.py`.

## Pipeline internals

`pipeline.GameAnalysisPipeline.analyze(frame)` returns a `GameState` with:

- `phase`: `new_round` / `betting` / `result` / `transition`
- `round_id`: OCR'd round id (e.g. `"2777221"`)
- `timer`: OCR'd timer string
- `dice_result`: one of `4_red` / `4_white` / `3w_1r` / `3r_1w` / `2w_2r` (or `None`)
- `bets[bet_type]`: `BetState(percent, total_bet, total_count, area_bbox)`

Cell assignment rules:

- `total_bet_cell` / `total_count_cell` — assigned by **containment**: the
  cell center must fall inside one of the `area_*` bboxes; ties are broken
  by smallest area. Nearest-area is used as a fallback if nothing contains
  the cell.
- `percent_cell` — 6 instances live in the scoreboard row above the
  play area, so they are assigned by **y-coordinate order** using the
  `PERCENT_ROW_ORDER` constant in `pipeline.py`. Adjust that list if the
  game layout swaps rows.

## Troubleshooting

- `ModuleNotFoundError: paddle` → `pip install paddlepaddle` (CPU) or the
  GPU variant matching your CUDA.
- Mean metrics OK but one class collapses → run `tools/check_labels.py`,
  check imbalance warnings and labeled bbox quality with `tools/visualize.py`.
- Small text cells missed at `imgsz=640` → re-run training with
  `--imgsz 800` (default) or higher.
- Dice class confused at low data → ensure each dice outcome has at least
  ~20 labeled frames; dice variants are visually distinct so data volume
  dominates.
