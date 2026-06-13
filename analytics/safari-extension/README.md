# Xoc Dia Safari Extension

Development-only Safari Web Extension for testing DOM-based clicks inside the game tab.

## Install

1. Start analytics locally:
   ```bash
   python3.11 -m analytics.serve --host 127.0.0.1 --port 8000
   ```
2. Open Safari Develop menu and enable unsigned extensions.
3. Use **Add Temporary Extension...** and select this folder:
   ```text
   analytics/safari-extension
   ```
4. Open the game tab. A small `XD Extension` panel should appear.

## Setup

Use the panel inside the game tab:

- `Set Chan`, `Set Le`, `Set Chip trai`, `Set Chip phai`: click the button, then click the matching point in the game.
- `Test`: dispatches a synthetic click at the saved page coordinate.

Coordinates are viewport-relative. Re-set them after resizing or moving the game layout.

## Notes

The extension dispatches DOM `PointerEvent`/`MouseEvent` events inside the game tab. If the game requires trusted user input (`event.isTrusted === true`), Safari may reject synthetic clicks.
