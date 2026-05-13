"""Tiny static server for the xoc dia analytics page.

Serves the files in ``analytics/`` as the static root and adds:

* ``GET /api/rounds.json`` — reads every ``rounds/*.json`` dump written
  by ``realtime_capture.py`` and returns them as an array. Optional
  query ``?tail=N`` returns only the last ``N`` rounds (used by
  ``frame-predict.html``).

Usage:

    python -m analytics.serve                   # LAN only (default)
    python -m analytics.serve --tunnel          # public qua cloudflare (vào thẳng)
    python -m analytics.serve --rounds-dir /path/to/rounds --port 8000
    python -m analytics.serve --host 127.0.0.1  # localhost only (không public)

Default bind 0.0.0.0 — mọi máy trong cùng mạng LAN đều truy cập được.
Thêm ``--tunnel`` để tạo URL public qua cloudflared (không cần tài khoản,
không interstitial). Cài: ``brew install cloudflared`` hoặc tải từ
https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
"""

from __future__ import annotations

import argparse
import atexit
import json
import re
import shutil
import socket
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import List

STATIC_DIR = Path(__file__).resolve().parent


def _get_lan_ip() -> str:
    """Best-effort LAN IP detection (no external traffic sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def _start_cloudflare_tunnel(port: int) -> None:
    """Open a free Cloudflare Quick Tunnel — no account, no interstitial."""
    cf = shutil.which("cloudflared")
    if not cf:
        print(
            "[tunnel] cloudflared chưa cài.\n"
            "[tunnel]   macOS:   brew install cloudflared\n"
            "[tunnel]   Linux:   https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/\n"
            "[tunnel]   Windows: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        )
        return

    proc = subprocess.Popen(
        [cf, "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    atexit.register(proc.terminate)

    url_pattern = re.compile(r"(https://[a-z0-9-]+\.trycloudflare\.com)")

    def _watch() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            m = url_pattern.search(line)
            if m:
                print(f"[tunnel] public URL:  {m.group(1)}")
                print("[tunnel] Gửi link trên cho người khác — vào thẳng, không cần click gì.")
                break

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    t.join(timeout=15)
    if t.is_alive():
        print("[tunnel] Đang chờ cloudflared khởi tạo tunnel...")


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

        def log_message(self, fmt, *args):
            if self.path and self.path.startswith("/api/"):
                return
            print(f"[req] {self.client_address[0]} {self.requestline}")

        def _send_json(self, status: int, payload, *, extra_headers=None) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (SimpleHTTPRequestHandler API)
            # Strip query string for endpoint dispatch.
            raw_path = self.path
            path = raw_path.split("?", 1)[0]
            if path == "/api/rounds.json":
                rounds = _load_rounds(rounds_dir)
                # ?tail=N  →  return only the last N rounds (most recent).
                # Useful for frame-predict.html which only needs the last
                # 15 cols × 6 rows = 90 rounds to build the display frame.
                qs = raw_path.split("?", 1)[1] if "?" in raw_path else ""
                for part in qs.split("&"):
                    if part.startswith("tail="):
                        try:
                            n = int(part[5:])
                            if n > 0:
                                rounds = rounds[-n:]
                        except ValueError:
                            pass
                        break
                self._send_json(200, rounds)
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tunnel",
        action="store_true",
        help="Mở tunnel cloudflare để public ra internet (cần cloudflared).",
    )
    args = parser.parse_args()

    rounds_dir = Path(args.rounds_dir).resolve()
    print(f"[analytics] serving static from: {STATIC_DIR}")
    print(f"[analytics] rounds source:       {rounds_dir}")
    print(f"[analytics] listening on:        http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        lan_ip = _get_lan_ip()
        print(f"[analytics] LAN URL:             http://{lan_ip}:{args.port}")
        print(f"[analytics] localhost URL:        http://127.0.0.1:{args.port}")

    if args.tunnel:
        _start_cloudflare_tunnel(args.port)

    handler_cls = make_handler(rounds_dir)
    HTTPServer((args.host, args.port), handler_cls).serve_forever()


if __name__ == "__main__":
    main()
