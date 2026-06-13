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
import ipaddress
import json
import re
import shutil
import socket
import subprocess
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, List
from urllib.parse import parse_qs, urlsplit

STATIC_DIR = Path(__file__).resolve().parent


class ApiError(RuntimeError):
    """HTTP-friendly error raised by the analytics API."""

    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


class ExtensionSignalBridge:
    """In-memory queue used by a local Safari Web Extension content script."""

    AMOUNTS = ("1k", "5k", "10k", "20k", "50k", "200k", "500k", "2m", "5m", "20m", "50m")
    AMOUNT_INDEX = {amount: idx for idx, amount in enumerate(AMOUNTS)}

    def __init__(self):
        self._lock = threading.Lock()
        self._seq = 0
        self._intent: dict[str, Any] | None = None
        self._last_result: dict[str, Any] | None = None

    @staticmethod
    def _amount(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text not in ExtensionSignalBridge.AMOUNT_INDEX:
            raise ApiError(400, "Mức cược extension không hợp lệ.")
        return text

    @staticmethod
    def _bet_clicks(value: Any) -> int:
        if value is None:
            return 1
        try:
            clicks = int(value)
        except (TypeError, ValueError):
            raise ApiError(400, "betClicks không hợp lệ.") from None
        if clicks < 1 or clicks > 256:
            raise ApiError(400, "betClicks phải nằm trong khoảng 1..256.")
        return clicks

    def publish(self, data: dict[str, Any]) -> dict[str, Any]:
        side = str(data.get("side", "")).strip().lower()
        if side not in {"chan", "le"}:
            raise ApiError(400, "side phải là chan hoặc le.")
        current = self._amount(data.get("currentAmount"))
        target = self._amount(data.get("targetAmount"))
        bet_clicks = self._bet_clicks(data.get("betClicks"))
        try:
            interval_ms = int(data.get("intervalMs", 650))
        except (TypeError, ValueError):
            raise ApiError(400, "intervalMs không hợp lệ.") from None
        steps = self.AMOUNT_INDEX[target] - self.AMOUNT_INDEX[current]
        interval_ms = max(50, min(600, interval_ms))
        now = time.time()
        with self._lock:
            self._seq += 1
            self._intent = {
                "seq": self._seq,
                "side": side,
                "currentAmount": current,
                "targetAmount": target,
                "steps": steps,
                "betClicks": bet_clicks,
                "intervalMs": interval_ms,
                "createdAt": now,
            }
            return {"ok": True, "seq": self._seq, "intent": self._intent}

    def next(self, since: int) -> dict[str, Any]:
        with self._lock:
            if since > self._seq:
                since = 0
            if self._intent and int(self._intent["seq"]) > since:
                return {"ok": True, "hasIntent": True, "seq": self._seq, "intent": self._intent}
            return {
                "ok": True,
                "hasIntent": False,
                "seq": self._seq,
                "lastResult": self._last_result,
            }

    def record_result(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            seq = int(data.get("seq", 0))
        except (TypeError, ValueError):
            seq = 0
        try:
            bet_clicks = int(data.get("betClicks") or 1)
        except (TypeError, ValueError):
            bet_clicks = 1
        result = {
            "seq": seq,
            "success": bool(data.get("success")),
            "message": str(data.get("message") or ""),
            "side": str(data.get("side") or ""),
            "targetAmount": str(data.get("targetAmount") or ""),
            "betClicks": max(1, min(256, bet_clicks)),
            "ts": time.time(),
        }
        with self._lock:
            self._last_result = result
            return {"ok": True, "lastResult": result, "seq": self._seq}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "seq": self._seq,
                "hasIntent": self._intent is not None,
                "lastIntent": self._intent,
                "lastResult": self._last_result,
            }


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


def _host_is_loopback(value: str | None) -> bool:
    if not value:
        return True
    host = value.strip()
    if not host:
        return True
    if "://" in host:
        host = urlsplit(host).hostname or ""
    elif host.startswith("["):
        host = host[1:].split("]", 1)[0]
    host = host.strip("[]").lower()
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if ip.is_loopback:
            return True
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped and mapped.is_loopback:
            return True
    if host.count(":") == 1:
        return _host_is_loopback(host.rsplit(":", 1)[0])
    return False


def make_handler(
    rounds_dir: Path,
    extension_bridge: ExtensionSignalBridge | None = None,
):
    extension_bridge = extension_bridge or ExtensionSignalBridge()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

        def log_message(self, fmt, *args):
            if self.path and self.path.startswith("/api/"):
                return
            print(f"[req] {self.client_address[0]} {self.requestline}")

        def _cors_origin(self) -> str:
            origin = self.headers.get("Origin")
            if origin and (origin == "null" or _host_is_loopback(origin)):
                return origin
            return "*"

        def _send_json(self, status: int, payload, *, extra_headers=None) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", self._cors_origin())
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _request_host_is_local(self) -> bool:
            client_host = self.client_address[0] if self.client_address else ""
            return _host_is_loopback(client_host) and _host_is_loopback(self.headers.get("Host"))

        def _read_json_body(self) -> dict:
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ApiError(400, "Content-Length không hợp lệ.") from None
            if n <= 0:
                return {}
            if n > 65536:
                raise ApiError(413, "Request quá lớn.")
            try:
                raw = self.rfile.read(n).decode("utf-8")
                data = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ApiError(400, "JSON không hợp lệ.") from None
            if not isinstance(data, dict):
                raise ApiError(400, "Body phải là JSON object.")
            return data

        def _send_api_error(self, exc: ApiError) -> None:
            self._send_json(exc.status, {
                "ok": False,
                "reason": exc.reason,
            })

        def _handle_extension_get(self, raw_path: str, path: str) -> bool:
            if not path.startswith("/api/extension/"):
                return False
            if not self._request_host_is_local():
                print(
                    "[extension] reject GET "
                    f"{path} client={self.client_address[0] if self.client_address else ''} "
                    f"host={self.headers.get('Host', '')}"
                )
                self._send_json(403, {"ok": False, "reason": "Extension API chỉ chạy qua localhost."})
                return True
            if path == "/api/extension/status":
                self._send_json(200, extension_bridge.status())
                return True
            if path == "/api/extension/next":
                qs = parse_qs(urlsplit(raw_path).query)
                try:
                    since = int(qs.get("since", ["0"])[0])
                except (TypeError, ValueError):
                    since = 0
                result = extension_bridge.next(since)
                if result.get("hasIntent"):
                    print(
                        "[extension] next "
                        f"since={since} -> #{result['intent']['seq']} "
                        f"{result['intent']['side']} "
                        f"{result['intent']['currentAmount']}->{result['intent']['targetAmount']}"
                    )
                self._send_json(200, result)
                return True
            self._send_json(404, {"ok": False, "reason": "Endpoint extension không tồn tại."})
            return True

        def _handle_extension_post(self, path: str) -> bool:
            if not path.startswith("/api/extension/"):
                return False
            print(
                "[extension] POST "
                f"{path} client={self.client_address[0] if self.client_address else ''} "
                f"host={self.headers.get('Host', '')} "
                f"origin={self.headers.get('Origin', '')}"
            )
            if not self._request_host_is_local():
                print(f"[extension] reject POST {path}: non-local host/client")
                self._send_json(403, {"ok": False, "reason": "Extension API chỉ chạy qua localhost."})
                return True
            try:
                data = self._read_json_body()
                if path == "/api/extension/intent":
                    result = extension_bridge.publish(data)
                    print(
                        "[extension] intent "
                        f"#{result['seq']} {result['intent']['side']} "
                        f"{result['intent']['currentAmount']}->{result['intent']['targetAmount']} "
                        f"steps={result['intent']['steps']} "
                        f"betClicks={result['intent']['betClicks']}"
                    )
                    self._send_json(200, result)
                    return True
                if path == "/api/extension/result":
                    result = extension_bridge.record_result(data)
                    print(
                        "[extension] result "
                        f"#{result['lastResult']['seq']} "
                        f"success={result['lastResult']['success']} "
                        f"betClicks={result['lastResult']['betClicks']} "
                        f"{result['lastResult']['message']}"
                    )
                    self._send_json(200, result)
                    return True
                raise ApiError(404, "Endpoint extension không tồn tại.")
            except ApiError as exc:
                self._send_api_error(exc)
                return True

        def do_GET(self) -> None:  # noqa: N802 (SimpleHTTPRequestHandler API)
            # Strip query string for endpoint dispatch.
            raw_path = self.path
            path = raw_path.split("?", 1)[0]
            if self._handle_extension_get(raw_path, path):
                return
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

        def do_OPTIONS(self) -> None:  # noqa: N802 (SimpleHTTPRequestHandler API)
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", self._cors_origin())
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 (SimpleHTTPRequestHandler API)
            path = self.path.split("?", 1)[0]
            if self._handle_extension_post(path):
                return
            self._send_json(404, {"ok": False, "reason": "Not found"})

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
    extension_bridge = ExtensionSignalBridge()
    print(f"[analytics] serving static from: {STATIC_DIR}")
    print(f"[analytics] rounds source:       {rounds_dir}")
    print(f"[analytics] listening on:        http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        lan_ip = _get_lan_ip()
        print(f"[analytics] LAN URL:             http://{lan_ip}:{args.port}")
        print(f"[analytics] localhost URL:        http://127.0.0.1:{args.port}")

    if args.tunnel:
        _start_cloudflare_tunnel(args.port)

    handler_cls = make_handler(rounds_dir, extension_bridge)
    HTTPServer((args.host, args.port), handler_cls).serve_forever()


if __name__ == "__main__":
    main()
