"""Tiny static server for the xoc dia analytics page.

Serves the files in ``analytics/`` as the static root and adds JSON
endpoints:

* ``GET /api/rounds.json`` — reads every ``rounds/*.json`` dump written
  by ``realtime_capture.py`` and returns them as an array.
* ``GET /api/prediction-history.json`` — returns the shared local
  prediction history (whatever ``app-with-prediction.js`` last POSTed).
  Returns ``[]`` if the file doesn't exist yet.
* ``POST /api/prediction-history.json`` — accepts a JSON array and
  atomically writes it to disk. Rejected (403) when the request looks
  like it came in via a Cloudflare tunnel — viewers connecting through
  ``--tunnel`` can READ the host's history but not OVERWRITE it.

The frontend polls these endpoints every few seconds so a live capture
session and any number of browser tabs (including read-only viewers
through a tunnel) stay in sync without a database.

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
import os
import re
import shutil
import socket
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import List, Optional

STATIC_DIR = Path(__file__).resolve().parent

# Max bytes accepted on POST /api/prediction-history.json. 8 MB ≈ ~16k
# rows of ~500 bytes each, far above any realistic local history. We
# reject larger bodies up-front to avoid OOM on a malicious POST.
MAX_HISTORY_POST_BYTES = 8 * 1024 * 1024

# Headers that Cloudflare's edge always adds when a request comes through
# a quick tunnel. Presence of any of them => the request is a remote
# viewer and must NOT be allowed to mutate the host's history file.
CLOUDFLARE_TUNNEL_HEADERS = ("cf-connecting-ip", "cf-ray", "cf-ipcountry")


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


# Single mutex protecting all reads/writes to the shared history file.
# The HTTPServer is single-threaded today, but we lock anyway so that
# switching to ThreadingHTTPServer later doesn't introduce torn writes.
_HISTORY_LOCK = threading.Lock()


def _load_prediction_history(history_file: Path) -> List[dict]:
    if not history_file.exists():
        return []
    try:
        raw = history_file.read_text(encoding="utf-8")
    except OSError:
        return []
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _save_prediction_history(history_file: Path, data: list) -> None:
    """Atomic write: serialise to a sibling tmp file then os.replace().

    os.replace is atomic on POSIX and Windows for files on the same
    filesystem, so concurrent readers always see either the previous
    snapshot or the new one — never a half-written file.
    """
    history_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = history_file.with_name(f".{history_file.name}.tmp")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, history_file)


def _is_via_tunnel(headers) -> bool:
    """Return True iff the request looks like it arrived via Cloudflare tunnel.

    `headers` is the http.server `BaseHTTPRequestHandler.headers` object
    (an `email.message.Message`). Any of the cf-* headers being present
    is taken as proof of a tunnel hop — cloudflared always sets at
    least cf-ray and cf-connecting-ip on forwarded requests.
    """
    for h in CLOUDFLARE_TUNNEL_HEADERS:
        if headers.get(h) is not None:
            return True
    return False


def make_handler(rounds_dir: Path, history_file: Path):
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
            path = self.path.split("?", 1)[0]
            if path == "/api/rounds.json":
                self._send_json(200, _load_rounds(rounds_dir))
                return
            if path == "/api/prediction-history.json":
                with _HISTORY_LOCK:
                    data = _load_prediction_history(history_file)
                # Tell the client whether they would be allowed to write.
                # A viewer browser can use this header to short-circuit
                # POST attempts without paying a 403 round-trip.
                writable = "0" if _is_via_tunnel(self.headers) else "1"
                self._send_json(
                    200,
                    data,
                    extra_headers={"X-History-Writable": writable},
                )
                return
            return super().do_GET()

        def do_POST(self) -> None:  # noqa: N802 (SimpleHTTPRequestHandler API)
            path = self.path.split("?", 1)[0]
            if path != "/api/prediction-history.json":
                self.send_error(404, "Not Found")
                return

            if _is_via_tunnel(self.headers):
                # Read-only mode for tunnel viewers. We deliberately do
                # NOT read the body — saves bandwidth and makes the
                # rejection cheap.
                self._send_json(
                    403,
                    {
                        "error": "read_only_via_tunnel",
                        "message": (
                            "Prediction history is read-only when accessed "
                            "through the cloudflare tunnel. The host machine "
                            "running analytics.serve is the single writer."
                        ),
                    },
                )
                return

            length_raw = self.headers.get("Content-Length")
            try:
                length = int(length_raw) if length_raw is not None else -1
            except ValueError:
                length = -1
            if length < 0:
                self._send_json(411, {"error": "length_required"})
                return
            if length > MAX_HISTORY_POST_BYTES:
                self._send_json(
                    413,
                    {
                        "error": "payload_too_large",
                        "limit_bytes": MAX_HISTORY_POST_BYTES,
                    },
                )
                return

            try:
                raw = self.rfile.read(length) if length > 0 else b""
            except OSError:
                self._send_json(400, {"error": "read_failed"})
                return

            try:
                payload = json.loads(raw.decode("utf-8")) if raw else []
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "invalid_json"})
                return

            if not isinstance(payload, list):
                self._send_json(
                    400,
                    {"error": "expected_array", "got": type(payload).__name__},
                )
                return

            with _HISTORY_LOCK:
                try:
                    _save_prediction_history(history_file, payload)
                except OSError as exc:
                    self._send_json(500, {"error": "write_failed", "detail": str(exc)})
                    return

            self._send_json(200, {"ok": True, "count": len(payload)})

    return Handler


def _resolve_history_file(arg: Optional[str], rounds_dir: Path) -> Path:
    """Resolve --history-file argument into an absolute Path.

    If the user passes ``--history-file``, we honour it. Otherwise we
    keep the file alongside the rounds directory (sibling, not inside)
    so it doesn't get confused with per-round dumps.
    """
    if arg:
        return Path(arg).expanduser().resolve()
    return (rounds_dir.parent / "prediction_history.json").resolve()


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
    parser.add_argument(
        "--history-file",
        default=None,
        help=(
            "Đường dẫn file lưu lịch sử prediction được chia sẻ "
            "(default: <rounds-dir>/../prediction_history.json). "
            "Host browser ghi vào đây qua POST; viewer qua --tunnel chỉ đọc được."
        ),
    )
    args = parser.parse_args()

    rounds_dir = Path(args.rounds_dir).resolve()
    history_file = _resolve_history_file(args.history_file, rounds_dir)
    print(f"[analytics] serving static from: {STATIC_DIR}")
    print(f"[analytics] rounds source:       {rounds_dir}")
    print(f"[analytics] history file:        {history_file}")
    print(f"[analytics] listening on:        http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        lan_ip = _get_lan_ip()
        print(f"[analytics] LAN URL:             http://{lan_ip}:{args.port}")
        print(f"[analytics] localhost URL:        http://127.0.0.1:{args.port}")

    if args.tunnel:
        _start_cloudflare_tunnel(args.port)

    handler_cls = make_handler(rounds_dir, history_file)
    HTTPServer((args.host, args.port), handler_cls).serve_forever()


if __name__ == "__main__":
    main()
