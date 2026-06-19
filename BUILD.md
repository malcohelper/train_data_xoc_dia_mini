# Building `XocDia.app`

Audience: maintainer (you) who already has the repo running locally and
wants to ship a `.app` bundle to a small group of friends. End-user
install instructions live in [`USER_GUIDE.md`](USER_GUIDE.md).

## What gets built

`build_app.sh` produces two artefacts in `dist/`:

| File | Purpose |
|------|---------|
| `XocDia.app` | The bundle itself. Double-clickable, drag into `/Applications`. |
| `XocDia.zip` | `ditto -c -k --keepParent` compressed bundle. This is what you share. |

The bundle ships with:

* The Python entry point (`app_main.py` → `realtime_capture.main()`).
* All Python deps (ultralytics, paddleocr, opencv, pyobjc-frameworks, …)
  copied into `Contents/Resources/lib/`.
* `best.pt` (your trained YOLO weights) at `Contents/Resources/best.pt`.

The bundle does NOT ship:

* PaddleOCR detector / recogniser checkpoints — they download lazily
  on first run to `~/.paddlex/`. The user needs internet access for
  the first launch only (~200 MB download).
* `rounds/` output directory — created at `~/Documents/XocDia/rounds`
  on first run.
* Code signature / notarisation — see "Distribution caveats" below.

Bundle size: 600 MB – 1.2 GB depending on torch / paddle version.
Compressed zip: ~300 – 500 MB.

## Prerequisites

1. **macOS 12.3+** (ScreenCaptureKit lives there).
2. **Python 3.11** — đây là phiên bản dùng hàng ngày trong repo; PyInstaller
   và Paddle được kiểm tra chủ yếu trên 3.11. (3.12 có thể dùng thử; khi lệch
   hãy quay về 3.11.) Chạy trong virtualenv của project:

   ```bash
   source venv/bin/activate
   ```

3. **All runtime deps already installed in that venv.** The build
   script verifies this by attempting `import cv2, ultralytics, paddle,
   paddleocr, ScreenCaptureKit, …` before invoking PyInstaller — if any
   import fails it aborts with a clear message.

4. **A trained `best.pt`** at the default location:
   `runs/detect/runs/detect/xocdia/weights/best.pt`

   To use a different weights file:

   ```bash
   XOCDIA_WEIGHTS=/path/to/your/best.pt ./build_app.sh
   ```

## Building

From the repo root:

```bash
./build_app.sh
```

That's it. The script:

1. Validates platform / venv / required imports.
2. Installs `pyinstaller` + `pyinstaller-hooks-contrib` if missing.
3. Removes any previous `build/` and `dist/` (PyInstaller caches
   sometimes pick up stale `.dylib`s after a `pip install --upgrade`).
4. Runs `python -m PyInstaller --clean --noconfirm xocdia.spec`.
5. Compresses the bundle to `dist/XocDia.zip` via `ditto`.

Build time: 5 – 15 minutes on Apple Silicon depending on which deps
need their `.dylib`s copied.

## Smoke-testing the bundle locally

```bash
open dist/XocDia.app
```

Expected behaviour:

1. macOS Gatekeeper *will* warn the first time (unsigned). Right-click
   the bundle → **Open** → confirm "Open Anyway".
2. The Tkinter window picker dialog should appear within ~3 seconds.
3. After picking Safari, the OpenCV preview window should show the
   game frame and the diag log file at
   `~/Library/Logs/XocDia/<timestamp>.log` should fill with
   `[diag] phase=…` lines.
4. Round JSONs land in `~/Documents/XocDia/rounds/`.

If the app silently quits, check `~/Library/Logs/XocDia/` for the
latest log — fatal errors (missing weights, denied permission, import
error) are written there *and* surfaced via a Tkinter alert.

## Distribution caveats

* **Unsigned bundle.** Recipients see "App can't be opened because
  Apple cannot check it for malicious software" the first time they
  launch. Documented work-around in `USER_GUIDE.md`: right-click →
  Open → "Open Anyway". One time only.

  To remove the warning permanently you'd need an Apple Developer
  account ($99/year) and a `codesign` + `xcrun notarytool submit`
  step. Not in scope for this build.

* **Architecture.** PyInstaller builds for the architecture you build
  on. Building on Apple Silicon → `arm64`-only bundle that won't run
  on Intel Macs. To support both you'd need a universal2 Python install
  + universal wheels for every dep, which paddleocr in particular
  doesn't ship.

* **macOS version floor.** The `Info.plist` declares
  `LSMinimumSystemVersion = 12.3` because ScreenCaptureKit needs that.

## Common build failures

| Symptom | Fix |
|---------|-----|
| `RuntimeError: operator torchvision::nms does not exist` at launch | This was the symptom that killed our py2app attempt. PyInstaller's `collect_all('torch'/'torchvision')` ships the compiled ops correctly. If it returns: ensure `pyinstaller-hooks-contrib` is installed and `torchvision` is in `xocdia.spec`'s `_collect()` list. |
| `ModuleNotFoundError: paddle._C` at app launch | Re-build with `rm -rf build dist` first; PyInstaller's cache occasionally drops paddle's compiled extensions when paddle is upgraded. |
| Bundle launches, dialog appears, but `dets=0` forever | Screen Recording permission missing for the *bundle* — even if you'd granted it to Terminal previously. System Settings → Privacy & Security → Screen Recording → enable `XocDia.app`. |
| `tkinter` dialog never shows | The bundle is using a Python without Tk linked. Use `python.org`'s installer or `brew install python-tk@3.11`, then rebuild the venv from that interpreter. |
| `ImportError: No module named 'X'` shortly after launch | One of the heavy deps got dropped from the bundle. The `EXCLUDES` list in `xocdia.spec` is intentionally tiny (only `modelscope` + Windows-only bits) because torch / ultralytics / paddleocr have lazy imports that crash at runtime when their deps are missing — `sympy` was the original culprit. Don't add anything to `EXCLUDES` without testing. If the missing module isn't being collected at all, add it to `_collect()` instead. |
| `OSError: dlopen … libpaddle.dylib` | The bundled paddle shipped without its sibling dylibs. Confirm `_collect("paddle")` is in `xocdia.spec`'s collect list and re-build with `rm -rf build dist` first. |

## Updating the bundle

Whenever `realtime_capture.py`, `pipeline.py`, or `best.pt` changes:

1. `git pull && pip install -r requirements.txt` (or whatever pulls
   updated deps).
2. `./build_app.sh`
3. Re-share `dist/XocDia.zip`.

Bundle version is set in `xocdia.spec` (`CFBundleShortVersionString`);
bump it before each new release so users can tell which they have via
About panel.
