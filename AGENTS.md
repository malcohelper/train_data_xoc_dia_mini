# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11 Xoc Dia detection pipeline. Core runtime modules live at the repository root: `detector.py` loads YOLO detections, `pipeline.py` converts a frame into structured game state, `ocr_engine.py` and `ocr_postprocess.py` handle OCR, and `realtime_capture.py` runs live capture plus round logging. Training and labeling entry points are `train.py`, `label_tool.py`, and `split_dataset.py`.

Support scripts are under `tools/` for dataset conversion, label checks, visualization, evaluation, and semi-automatic labeling. The browser analytics UI is in `analytics/`, served by `analytics/serve.py`. Unit tests live in `tests/`. Generated or local-heavy data such as `dataset/`, `rounds/`, `debug_cells/`, `runs/`, `build/`, and `dist/` should stay uncommitted unless the change explicitly requires a curated fixture.

## Build, Test, and Development Commands

- `./run.sh`: create or reuse `venv`, install `requirements.txt`, choose a `best.pt`, and launch live capture.
- `python realtime_capture.py --weights runs/detect/runs/detect/xocdia/weights/best.pt`: run capture directly from an activated environment.
- `python -m analytics.serve --rounds-dir rounds --port 8000`: serve the analytics UI and `/api/rounds.json`.
- `python tools/check_labels.py --split val`: validate YOLO labels against images.
- `python train.py --epochs 150 --batch 16 --imgsz 800`: train the YOLOv8 model with repository defaults.
- `./build_app.sh`: build `dist/XocDia.app` and `dist/XocDia.zip` for macOS distribution.

## Coding Style & Naming Conventions

Use 4-space indentation, type hints where they clarify public data structures, and small functions with explicit inputs. Keep class names in `PascalCase`, functions and variables in `snake_case`, and constants in `UPPER_SNAKE_CASE`. Preserve the existing dataclass-based style for game state objects. Treat `classes.py` and `xocdia.yaml` as a coupled schema: update both when class IDs or names change.

## Testing Guidelines

Run `python -m pytest tests/` before submitting changes. Add regression tests near the code they protect; current tests use `test_<behavior>` names and focus on OCR recovery cases. For pipeline, detector, or labeling changes, also run the relevant manual QA command from `tools/`, especially `check_labels.py`, `visualize.py`, or `eval.py`.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style messages such as `feat(analytics): ...`, `fix(analytics): ...`, and `feat: ...`. Keep messages imperative and scoped when useful. Pull requests should describe the user-visible behavior, list commands run, note any model/data artifacts required, and include screenshots or sample logs for UI, capture, or analytics changes.
