# Xoc Dia Detector (15-class, single-stage)

Single-stage YOLOv8 + PaddleOCR pipeline for the Xoc Dia game UI. One YOLO
model detects all 15 UI elements (state / areas / dice / text cells), then a
small post-processing pass uses PaddleOCR to read the text inside each
detected cell and a geometric mapping step figures out which bet type each
cell belongs to.

Round boundaries are detected on the fly by a lightweight state machine:
`timer` ≥ 46 signals a new round (scoreboard percent is captured then);
`dice_*` signals the end of a round (bets / counts / result are logged and
the round JSON is persisted). No `round_id` or `new_round` class is needed
— round IDs are timestamp-based.

## Pipeline

```
Frame ─► YOLOv8 (15 classes) ─► Group by category
                                ├─ state:   timer
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
                                    GameState (per-frame)
                                               │
                                               ▼
                              RoundTracker (multi-frame state machine)
                                               │
                                               ▼
                              1 log line + rounds/<ts>.json per round
```

## Class schema (15 classes)

| Group | IDs  | Classes |
|-------|------|---------|
| State | 0    | `timer` |
| Areas | 1–6  | `area_chan`, `area_le`, `area_4_red`, `area_3w_1r`, `area_3r_1w`, `area_4_white` |
| Dice  | 7–11 | `dice_4r`, `dice_4w`, `dice_3w1r`, `dice_3r1w`, `dice_2w2r` |
| Cells | 12–14| `percent_cell`, `total_bet_cell`, `total_count_cell` |

Text cells use generic classes (up to 6 instances per frame) and are
assigned to their owning bet type downstream via geometric containment
against the `area_*` anchors. Defined once in [`classes.py`](classes.py);
everything imports from there.

## Round state machine

Round tracking lives in `realtime_capture.RoundTracker`:

```
IDLE
  │  timer transitions from <46 (or None) to >=46 on a fresh frame
  ▼
ACTIVE  ─► capture scoreboard percent (done once, scoreboard stable)
  │       keep refreshing total_bet / total_count every frame
  │
  │  dice_* detected
  ▼
LOG 1x  ─► format + print log, save rounds/<ts>.json
  │
  ▼
IDLE (wait for next timer>=46)
```

Log format:

```
============================================================
ROUND 20260423_220512 | Dice: 3w_1r
  chan     total_bet=272K     count=35
  4_red    total_bet=730K     count=21
  4_white  total_bet=795K     count=35
  le       total_bet=602K     count=24
  3r_1w    total_bet=4062     count=27
  3w_1r    total_bet=701K     count=42
PERCENT: chan 58% | 4_red 12% | 4_white 12% | le 42% | 3r_1w 33% | 3w_1r 42%
============================================================
```

The row order is `PERCENT_ROW_ORDER`, defined once in `pipeline.py` and
imported from there by `realtime_capture.py`. Edit that single list if the
scoreboard layout changes.

## Install

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install ultralytics opencv-python paddleocr paddlepaddle numpy mss pillow
# GPU-only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# macOS-only (window picker, see "Capture source" below):
# pip install pyobjc-framework-Quartz pyobjc-framework-ScreenCaptureKit
# brew install python-tk@3.11   # Tk for the picker dialog (Homebrew
#                                # Python doesn't bundle it by default)
# After install, grant Screen Recording permission to your terminal
# in System Settings -> Privacy & Security -> Screen Recording so
# ScreenCaptureKit can capture live frames across macOS Spaces.
```

## Repo layout

```
.
├── classes.py                # single source of truth for classes/colors/groups
├── xocdia.yaml               # YOLO data config (must match classes.py)
├── label_tool.py             # fast label GUI (15-class + text-input picker)
├── train.py                  # train YOLOv8 (game-UI tuned defaults)
├── detector.py               # lightweight XocDiaDetector API + CLI
├── ocr_engine.py             # PaddleOCR wrapper
├── ocr_postprocess.py        # per-class OCR sanitisers (regex + confusables)
├── cell_preprocessor.py      # per-class image preprocessing before OCR
                              #   (HSV mask, upscale, binarise) - big win on
                              #   stylised game-UI text vs raw crops
├── pipeline.py               # per-frame GameAnalysisPipeline
├── realtime_capture.py       # screen-grab + state machine + rounds/ dump
├── split_dataset.py          # 80/20 train/val split
├── dataset/
│   ├── images/{raw,train,val}
│   └── labels/{raw,train,val}
├── rounds/                   # per-round PNG + JSON snapshots
├── analytics/
│   ├── serve.py              # `python -m analytics.serve` - static HTML +
│   │                         #   /api/rounds.json endpoint
│   ├── index.html            # dashboard (Tailwind CDN)
│   └── app.js                # Chẵn/Lẻ + vị stats + Big Road renderer
└── tools/
    ├── rounds_to_dataset.py          # copy rounds/*.png into dataset/images/raw/
    ├── migrate_labels_15class.py     # one-shot: 17-class -> 15-class label remap
    ├── visualize.py                  # preview YOLO labels over images
    ├── cell_preview.py               # highlight a single class / group (label QA)
    ├── semi_auto_label.py            # pre-label new images with a trained model
    ├── check_labels.py               # label QA / imbalance warnings
    └── eval.py                       # per-class mAP / P / R on the val split
```

## End-to-end workflow

### 1. Collect frames

```bash
# Live capture (writes rounds/<ts>.json once per finished round).
# PNG is NOT saved from the realtime loop anymore (kept it lightweight).
# Use tools/rounds_to_dataset.py on previously-captured PNGs if you need them.
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
| `/` then `NN` then `Enter` | jump to class N (0–14) via text input |
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
python tools/visualize.py --split train               # interactive (all classes)
python tools/visualize.py --split train --save-dir qa # batch export
python tools/cell_preview.py --classes total_bet_cell # highlight ONE class
python tools/cell_preview.py --group cell --save-dir qa/cell
python tools/cell_preview.py --classes dice_4w dice_2w2r --only-with-target
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

### 5b. Semi-auto labeling (optional, after first training)

Once you have a working `best.pt` you can pre-label the next batch of
images with the model and only manually fix what it got wrong, instead
of drawing every box from scratch:

```bash
# Fill empty .txt files in dataset/labels/train with model predictions.
python tools/semi_auto_label.py --weights runs/detect/xocdia/weights/best.pt

# Same, but also write annotated PNG previews to qa_preview/auto for
# spot-checking before opening the label tool.
python tools/semi_auto_label.py --weights runs/detect/xocdia/weights/best.pt \
                                --preview-dir qa_preview/auto

# Per-category confidence overrides (categories: state, area, dice, cell).
python tools/semi_auto_label.py --weights .../best.pt \
                                --per-category-conf dice=0.25 area=0.55
```

By default the script **never overwrites** an existing `.txt` (so your
hand-edited labels are safe). Use `--overwrite` only when you really
want to redo a folder. **Caveat:** an empty `.txt` (model returned 0
detections) also counts as "already labeled", so re-running won't
re-attempt those images - pass `--overwrite` after retraining a better
model if you want to retry them. After running, open the same images in
`label_tool.py` and refine the auto-generated boxes - the workflow is
identical to manual labeling, except you start with boxes drawn instead
of an empty canvas.

### 6. Run

```bash
# Offline: single image -> annotated image + JSON GameState
python pipeline.py --weights runs/detect/xocdia/weights/best.pt \
                   --source frame.png --save-annotated annotated.png

# Detector only (no OCR / no state inference)
python detector.py --weights runs/detect/xocdia/weights/best.pt --source frame.png

# Real-time screen capture + analyze
python realtime_capture.py --weights runs/detect/xocdia/weights/best.pt
# A small dialog pops immediately with two buttons:
#   * "Pick Window" - lists the macOS app windows currently visible
#     (Quartz/`pyobjc-framework-Quartz`); pick e.g. "Safari - XocDia"
#     and the script captures exactly that window's bbox and follows
#     it when you move/resize the window.
#   * "Drag ROI"    - the original drag-rectangle flow on full screen.
# Right after either choice we run a single high-resolution YOLO pass
# and tighten the capture region to the detected UI bbox, so the loop
# runs at the default imgsz without losing detail. Press `r` later in
# the preview to re-pick (mirrors the startup flow), `c` to re-tighten
# without re-picking, `q` to quit. Pass `--capture-mode window|roi` to
# skip the dialog, `--no-auto-roi` to skip the picker entirely, or
# `--no-auto-clamp` to skip the auto-tighten pass.
```

Tune CPU usage on slow machines via `--preview-fps` (default 10;
detection still runs on its own `--interval`, default 1s). Pass
`--diag` to print one diagnostic line per detection tick (phase,
timer, dice, det count, monitor bounds) when troubleshooting.

#### Picking a window vs dragging a ROI

The original "drag a rectangle" flow makes it very easy to drag past
the game window into your terminal or another app, which then becomes
part of the YOLO input frame and tanks detection rates because the
game ends up squeezed in the smaller fraction of the captured pixels.
Picking a window via Quartz sidesteps that entirely and also lets the
script auto-follow the window when you reposition it (re-checked
every 5 s by default, see `RealtimeCapture.window_refresh_interval`).

In window-picker mode the capture pipeline reads pixels straight from
the window's backing store, **not** from screen coordinates. Three
backends are tried in order, the first one that returns a frame wins:

1. **`ScreenCaptureKit`** (macOS 12.3+, requires
   `pyobjc-framework-ScreenCaptureKit`) — the modern API. Captures
   live frames even when the target window is in another macOS Space:
   you can run Safari fullscreen and continue using your terminal on
   another Space, the script keeps detecting against the *live*
   Safari content. Requires a one-time **Screen Recording**
   permission grant in *System Settings → Privacy & Security → Screen
   Recording* for whichever terminal/IDE launches Python.
2. **`CGWindowListCreateImage`** (Quartz, deprecated on macOS 14+,
   fallback for older systems). Captures the window's backing store
   so other windows on top of the game don't contaminate the frame.
   Limitation: returns a *stale snapshot* for windows in another
   Space, which is why the SCKit path above is preferred.
3. **`mss`** screen-region capture — last resort, also used for
   drag-ROI mode. Reads whatever is rendered at the configured screen
   coordinates, so other windows on top *will* replace the game in
   the captured frame.

Trade-offs visible to the user:

- The OpenCV preview window we render ourselves can sit anywhere on
  screen (even on top of the game) without contaminating the captured
  frame in modes 1 and 2.
- Notifications, dialogs, and other apps in front of the game don't
  cause `dets=0` ticks in modes 1 and 2.
- Switching to a different Space (e.g. fullscreen Safari + a separate
  terminal Space) only works correctly with mode 1 (SCKit). Modes 2
  and 3 will produce stale snapshots / wrong content respectively.

Auto-clamp (`--no-auto-clamp` to disable) only runs in drag-ROI mode.
In window mode the captured region is already exactly the window
content and the 1-shot YOLO clamp pass tends to trim into a partial
UI fragment (e.g. just the betting cells, missing the scoreboard the
model needs as context); the `c` hotkey is also a no-op in window
mode for the same reason.

The window picker requires `pyobjc-framework-Quartz` *and* `tkinter`
on macOS. The cross-Space SCKit path additionally requires
`pyobjc-framework-ScreenCaptureKit`. Homebrew Python 3.11 doesn't
bundle Tk - install it with `brew install python-tk@3.11` (or the
matching version for your Python). On other platforms or when those
packages aren't installed the dialog falls back to drag-ROI with a
clear log message.

### Analytics dashboard

`analytics/` is a small static page that reads the `rounds/*.json`
dumps written by `realtime_capture.py` and renders a Chẵn/Lẻ
progress card, per-dice-combo (4-trắng / 3T1Đ / 2T2Đ / 3Đ1T / 4-đỏ)
stats, and a Baccarat-style 6-row Big Road. It polls `/api/rounds.json`
every 3 seconds so a live capture session updates the dashboard in
the browser without a reload.

```bash
# From the repo root, with realtime_capture.py writing to ./rounds
python -m analytics.serve                          # http://127.0.0.1:8000
python -m analytics.serve --rounds-dir ~/captures/rounds --port 8080
```

The time-range filter defaults to `[first round, latest round]` and
follows new rounds automatically. Click `AUTO` (or edit either date)
to pin a historical window; click again to re-enable follow.

## Packaging as `XocDia.app` (macOS bundle)

To ship a double-clickable `.app` to friends who don't want to manage
a Python venv, run on macOS:

```bash
./build_app.sh                                      # produces dist/XocDia.app + dist/XocDia.zip
```

See [`BUILD.md`](BUILD.md) for prerequisites, troubleshooting, and how
to point at a non-default `best.pt`. End-user docs (Gatekeeper bypass,
Screen Recording permission, daily usage) live in
[`USER_GUIDE.md`](USER_GUIDE.md) and ship inside the bundle.

The bundle is unsigned; recipients right-click → Open the first time,
grant Screen Recording permission, then double-click as normal. Apple
Silicon only (no Intel Mac support).

## Dataset sizing

- Minimum workable: ~80–100 labeled frames covering all phases (betting,
  bowl-opens, transitions).
- Comfortable: 150–250 frames. Text cells benefit from variety of values
  (different `xxM` amounts, different % splits).
- Keep ~20% for val. Use `split_dataset.py`.

## Pipeline internals

`pipeline.GameAnalysisPipeline.analyze(frame)` returns a `GameState` with:

- `phase`: `betting` / `result` / `transition`
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

## Migrating from the old 17-class schema

If you already labeled data against the old 17-class schema (with
`round_id` and `new_round`), run the one-shot remapper once:

```bash
python tools/migrate_labels_15class.py            # dry-run preview
python tools/migrate_labels_15class.py --apply    # rewrite in place
```

It drops all `round_id` / `new_round` boxes and shifts the remaining class
IDs down to fill the gaps.

**Idempotency via sentinel file.** On success the tool writes a sentinel
file `dataset/labels/.migrated_to_15class` and refuses to run again
unless you pass `--force`. This is the only reliable guard: the two
schemas share class IDs `0..14`, so you cannot distinguish an
already-migrated file from an old-schema file that merely happens not to
contain `total_bet_cell`/`total_count_cell` boxes. Running on already-
migrated labels corrupts class IDs, hence the hard stop. The dataset
shipped in this repo already has the sentinel, so pulling main and
running `--apply` is a no-op.

## Troubleshooting

- `ModuleNotFoundError: paddle` → `pip install paddlepaddle` (CPU) or the
  GPU variant matching your CUDA.
- Mean metrics OK but one class collapses → run `tools/check_labels.py`,
  check imbalance warnings and labeled bbox quality with `tools/visualize.py`.
- Small text cells missed at `imgsz=640` → re-run training with
  `--imgsz 800` (default) or higher.
- OCR returns garbage like `WBEL` / `Ell` → bboxes are too loose. Re-label
  the offending frames so the box hugs **only** the digits (no surrounding
  frame / icon / whitespace). See the labeling guide in the PR description.
- Realtime log still shows letters in numeric fields (e.g. `count=LE`,
  `total_bet=W92'2`) → run
  `python realtime_capture.py --weights ... --log-ocr-rejects` to print
  one `[OCR-REJECT] cls=... raw=...` line per rejected cell. The realtime
  pipeline runs two complementary OCR-quality layers:
    1. **`cell_preprocessor.py`** — image-side: HSV mask the foreground
       (yellow money / white counts), Otsu binarise, bicubic upscale to
       ~64-80px, white-pad. Turns stylised game-UI text into clean
       black-on-white that PaddleOCR's CRNN handles well.
    2. **`ocr_postprocess.py`** — text-side: per-class regex + confusable
       normalisation (`B↔8`, `O↔0`, `I↔1`, `S↔5`, `Z↔2`, ...). Rejects
       values that don't match the expected pattern.
  When a cell is rejected the log shows `-` instead of garbage, and the
  reject log helps identify which cells need tighter labels.
- Percent in the `[46,48]` window flickers between values (e.g. `47%` on
  one frame, `7%` on the next due to OCR noise) → no action needed. The
  realtime tracker accumulates *all* sanitised percent reads inside the
  window and resolves them via majority vote at round finalisation
  (`Round.finalise_percent`). A single bad frame in a 2-3 frame window is
  voted out automatically.
- Need to tune `cell_preprocessor.py` for a specific cell class? Pass
  `--debug-save-cells [DIR]` (default `debug_cells/`) to
  `realtime_capture.py`. Every cell crop fed to the OCR is then dumped
  to disk both before and after preprocessing, alongside a `.txt` with
  the OCR + sanitised result:
  ```
  debug_cells/
    f00001_chan_total_bet_cell_raw.png      # what YOLO cropped
    f00001_chan_total_bet_cell_prep.png     # what PaddleOCR sees
    f00001_chan_total_bet_cell.txt          # raw_ocr=... / sanitised=...
    ...
  ```
  Disable in production - the I/O is non-trivial.
- Dice class confused at low data → ensure each dice outcome has at least
  ~20 labeled frames; dice variants are visually distinct so data volume
  dominates.
