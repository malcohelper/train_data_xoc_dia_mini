"""Tiny static server for the xoc dia analytics page.

Serves the files in ``analytics/`` as the static root and adds a single
JSON endpoint ``/api/rounds.json`` that reads every ``rounds/*.json``
dump written by ``realtime_capture.py`` and returns them as an array.

The frontend polls this endpoint every few seconds so a live capture
session and a browser tab can stay in sync without touching the
realtime loop or adding a database.

Usage:

    python -m analytics.serve                   # default rounds dir = ./rounds
    python -m analytics.serve --rounds-dir /path/to/rounds --port 8000

Then open http://127.0.0.1:8000 in a browser.
"""

from __future__ import annotations

import argparse
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import List

STATIC_DIR = Path(__file__).resolve().parent


def _load_rounds(rounds_dir: Path) -> List[dict]:
    out: List[dict] = []
    if not rounds_dir.exists():
        return out
    for p in sorted(rounds_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            # Ignore half-written / corrupt files; the next poll cycle
            # will pick them up once they're complete.
            continue
    return out


def make_handler(rounds_dir: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

        def log_message(self, fmt, *args):  # quieter default logging
            return

        def do_GET(self) -> None:  # noqa: N802 (SimpleHTTPRequestHandler API)
            # Strip query string for endpoint dispatch.
            path = self.path.split("?", 1)[0]
            if path == "/api/rounds.json":
                data = _load_rounds(rounds_dir)
                body = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
                return
            return super().do_GET()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds-dir",
        default="rounds",
        help="Directory containing per-round JSON dumps (default: ./rounds).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    rounds_dir = Path(args.rounds_dir).resolve()
    print(f"[analytics] serving static from: {STATIC_DIR}")
    print(f"[analytics] rounds source:       {rounds_dir}")
    print(f"[analytics] listening on:        http://{args.host}:{args.port}")

    handler_cls = make_handler(rounds_dir)
    HTTPServer((args.host, args.port), handler_cls).serve_forever()


if __name__ == "__main__":
    main()
