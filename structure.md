# Cấu trúc dự án Xoc Dia (train_data_xoc_dia_mini)

Tài liệu tổng hợp thư mục, file chính và chức năng. Pipeline cốt lõi: **YOLOv8 15 lớp** phát hiện UI → **PaddleOCR** đọc ô chữ → **state machine** theo dõi vòng chơi và ghi `rounds/*.json`. Phần **analytics** là một trang HTML tĩnh (`frame-predict.html`) + server nhỏ (`serve.py`) đọc `rounds/*.json` qua `GET /api/rounds.json`. **Python 3.11** là phiên bản chuẩn cho dev (`venv`, `run.sh`, build `.app` — xem `BUILD.md`).

---

## Cây thư mục (tóm tắt)

```
train_data_xoc_dia_mini/
├── README.md, requirements.txt, xocdia.yaml
├── annotated.png, preview_capture.png          # ảnh minh họa / preview capture
├── prediction_history.json                     # (tuỳ chọn, legacy) không còn được serve.py đọc/ghi
├── predictor.html                              # UI React độc lập (Column Logic Discovery)
│
├── classes.py                                  # Schema 15 class (single source of truth)
├── detector.py                                 # YOLOv8 detector + nhóm theo category
├── pipeline.py                                 # GameAnalysisPipeline: 1 frame → GameState
├── ocr_engine.py, ocr_postprocess.py           # PaddleOCR + hậu xử lý chuỗi
├── cell_preprocessor.py                        # Tiền xử lý crop OCR (binary, upscale)
├── realtime_capture.py                         # Capture màn hình + RoundTracker + ghi JSON
├── window_picker.py                            # Chọn vùng ROI / cửa sổ
├── train.py                                    # Huấn luyện YOLO (Ultralytics)
├── split_dataset.py                            # Chia train/val dataset
├── label_tool.py                               # Công cụ gán nhãn bbox (CV)
├── app_main.py                                 # Entry cho bundle macOS .app (log, đường dẫn weights)
├── run.sh                                      # Launcher dev: venv + realtime_capture (xem README)
├── build_app.sh, BUILD.md                      # Đóng gói XocDia.app + tài liệu build
├── USER_GUIDE.md                               # Hướng dẫn người dùng app (.zip / .app)
│
├── dataset/
│   ├── images/train, images/val
│   └── labels/train, labels/val                # YOLO labels (.txt)
│
├── rounds/                                     # Round JSON (phiên bản/giai đoạn capture)
├── rounds-2/                                   # Round JSON (bộ dữ liệu lớn, timestamp *.json)
├── rounds-22/                                  # Thư mục round bổ sung (nếu có)
│
├── runs/                                       # Ultralytics: weights, logs (vd. detect/xocdia/)
├── debug_cells/                                # Debug crop ô (nếu dùng khi dev)
├── prediction_compare/                         # Giữ chỗ / so sánh (có thể trống)
├── tools/                                      # Script phụ trợ dataset & đánh giá
├── tests/                                      # Pytest (OCR postprocess, …)
├── analytics/                                  # frame-predict.html + serve.py (API rounds)
├── venv/                                       # Virtualenv Python (môi trường cục bộ)
└── .cursor/                                    # Plan/rules Cursor (IDE)
```

**Lưu ý:** `rounds-2/` chứa rất nhiều file `YYYYMMDD_HHMMSS.json` — mỗi file một vòng đã kết thúc (`round_id`, `dice_result`, `bets`, `percent`, …). Không liệt kê từng file tại đây.

---

## Thư mục gốc — file và vai trò

| File | Chức năng |
|------|-----------|
| `README.md` | Mô tả pipeline, schema 15 class, state machine round, định dạng log. |
| `requirements.txt` | Phụ thuộc Python (YOLO, PaddleOCR, OpenCV, …); đã kiểm tra trên **Python 3.11**. |
| `xocdia.yaml` | Cấu hình Ultralytics: `path`, train/val, map ID→tên class (timer, area_*, dice_*, *_cell). |
| `classes.py` | Định nghĩa `CLASSES`, màu, nhóm category; mọi module import từ đây. |
| `detector.py` | `XocDiaDetector`: load `.pt`, `detect()`, `group_by_category()`, vẽ annotation. |
| `pipeline.py` | `GameAnalysisPipeline`: detect → gán cell vào bet type (geometry + `PERCENT_ROW_ORDER`) → OCR → `GameState` + phase (`betting`/`result`/…). |
| `ocr_engine.py` | Lớp bọc PaddleOCR cho crop ô. |
| `ocr_postprocess.py` | Chuẩn hoá/sửa chuỗi số ký tự OCR (vd. B↔8). |
| `cell_preprocessor.py` | Mask màu, Otsu, upscale, padding — ảnh đen-trắng cho CRNN. |
| `realtime_capture.py` | Vòng lặp capture (`mss`), máy trạng thái round (`timer ≥ 46` → vòng mới; `dice_*` → kết thúc), ghi log + `rounds/<ts>.json`, hotkey `r/c/s/d/q`. |
| `window_picker.py` | Chọn vùng/quản lý cửa sổ game. |
| `train.py` | CLI huấn luyện YOLOv8 (epochs, imgsz 800, augment UI-friendly, resume). |
| `split_dataset.py` | Chia ảnh/nhãn train vs val. |
| `label_tool.py` | Tool labeling bbox theo class (filter class/category, lưu YOLO). |
| `app_main.py` | Entry py2app/PyInstaller: redirect log, `best.pt` trong bundle, rounds user-writable, messagebox lỗi. |
| `run.sh` | Launcher dev: tìm **Python 3.11** (fallback 3.12), kiểm tra tkinter, tạo `venv`, cài `requirements.txt`, chọn `best.pt`, gọi `realtime_capture.py`. |
| `build_app.sh` | Đóng gói macOS → `dist/XocDia.app` (kèm `BUILD.md`). |
| `BUILD.md` | Prerequisites và troubleshooting khi build bundle. |
| `USER_GUIDE.md` | Hướng dẫn cài đặt / dùng `XocDia.app`; mục analytics cho người clone repo. |
| `predictor.html` | Trang độc lập (React + Babel + Tailwind): khám phá “column logic”, highlight pattern, so khớp AI. |

---

## `dataset/`

- **`images/train`, `images/val`**: Khung hình gốc cho huấn luyện/validation.
- **`labels/train`, `labels/val`**: Nhãn YOLO (bbox normalized) tương ứng 15 class trong `xocdia.yaml`.

---

## `rounds/`, `rounds-2/`, `rounds-22/`

- **Mục đích:** Lưu snapshot mỗi vòng chơi đã hoàn tất.
- **Schema điển hình:** `round_id`, `started_at`, `finalised_at`, `dice_result` (vd. `3r_1w`), `percent` (scoreboard %), `bets` (mỗi cửa: `total_bet`, `total_count`).
- **`analytics/serve.py`** mặc định đọc `rounds/*.json`; có thể đổi `--rounds-dir` trỏ tới `rounds-2` khi phân tích bộ dữ liệu lớn.

---

## `runs/`

- Artefact **Ultralytics** sau `train.py`: weights (`best.pt`), metrics, plots.
- `detector`/`pipeline` thường trỏ `runs/detect/.../weights/best.pt` hoặc symlink.

---

## `tools/`

| Script | Chức năng |
|--------|-----------|
| `check_labels.py` | Kiểm tra nhãn vs ảnh, consistency. |
| `eval.py` | Đánh giá model trên val/test, per-class metrics. |
| `visualize.py` | Trực quan hoá detection/labels. |
| `semi_auto_label.py` | Gợi ý nhãn bán tự động. |
| `rounds_to_dataset.py` | Chuyển dump round → ảnh/cặp huấn luyện (nếu dùng). |
| `cell_preview.py` | Preview crop ô OCR. |
| `diag_detection.py` | Chẩn đoán detection. |
| `migrate_labels_15class.py` | Migration schema nhãn 15 class. |

---

## `tests/`

- **`test_ocr_postprocess.py`**: Kiểm tra logic hậu xử lý OCR (regression cho ký tự nhầm).

---

## `analytics/` — frame-predict

| File | Chức năng |
|------|-----------|
| `serve.py` | HTTP static root = `analytics/`; **`GET /api/rounds.json`** — gộp mọi `*.json` trong thư mục rounds (tuỳ chọn `?tail=N`); `--tunnel` Cloudflare quick tunnel. |
| `frame-predict.html` | Trang đơn (CSS + JS inline): poll `/api/rounds.json`, khung cầu + dự đoán; trạng thái trong **localStorage**. |
| `__init__.py` | Cho phép `python -m analytics.serve`. |

---

## File & thư mục phụ trợ

| Đường dẫn | Ghi chú |
|-----------|---------|
| `prediction_history.json` | File legacy ở root (nếu còn); không còn được `serve.py` phục vụ. |
| `debug_cells/` | Ảnh/debug crop khi phát triển OCR/pipeline. |
| `annotated.png`, `preview_capture.png` | Minh họa hoặc frame lưu nhanh (`s` trong capture). |
| `venv/` | Môi trường Python cục bộ — không commit thường quy. |
| `.cursor/` | Kế hoạch/chỉnh IDE Cursor. |

---

## Luồng dữ liệu tổng quát

1. **Capture:** `realtime_capture.py` → YOLO + pipeline + OCR → JSON mỗi vòng.
2. **Phân tích web:** `python -m analytics.serve` → mở `frame-predict.html` → đọc `/api/rounds.json` (có thể `?tail=N`).

---

*Tài liệu được tạo để định hướng điều hướng codebase; cập nhật khi thêm module hoặc đổi đường dẫn mặc định.*
