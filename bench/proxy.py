"""Loopback recorder between lm-eval and the model server.

lm-eval's chat backend keeps only the reply text. This forwards each
request body unchanged to the server and writes one JSON line per chat
exchange holding the request and the response bodies, so the record shows
every sampling parameter actually sent and everything the server returned
(finish reason, token usage, the full text including thinking).

Both ends are 127.0.0.1. Request headers are not forwarded (apart from the
content type) and nothing about them is recorded. If VLLM_API_KEY is set,
the key is sent to the server as a bearer token, as vLLM expects; it is
never written anywhere.
"""
from __future__ import annotations

import http.client
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHAT_PATH = "/v1/chat/completions"
FORWARDED = ("/v1/", "/version", "/health")


def call(port: int, method: str, path: str, body: bytes | None = None,
         timeout: float | None = None) -> tuple[int, str, bytes]:
    """One HTTP request to 127.0.0.1:port; returns (status, content type, body)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if key := os.environ.get("VLLM_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.getheader("Content-Type", "application/json"), resp.read()
    finally:
        conn.close()


def get_json(port: int, path: str) -> object | None:
    try:
        status, _, body = call(port, "GET", path, timeout=10)
        return json.loads(body) if status == 200 else None
    except (OSError, ValueError):
        return None


def _parsed(raw: bytes) -> object:
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")


class Recorder:
    """Serves on 127.0.0.1:<ephemeral> until close(); appends to `log`."""

    def __init__(self, upstream_port: int, log: Path):
        self.upstream = upstream_port
        self.log = log.open("x", encoding="utf-8")
        self.lock = threading.Lock()
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _forward(self, method: str):
                if not self.path.startswith(FORWARDED):
                    return self._reply(404, "text/plain", b"not forwarded")
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if method == "POST" else None
                started = time.time()
                try:
                    status, ctype, out = call(recorder.upstream, method, self.path, body)
                    error = None
                except OSError as e:
                    status, ctype, out, error = 502, "text/plain", str(e).encode(), str(e)
                if method == "POST" and self.path == CHAT_PATH:
                    line = {"path": self.path, "started": round(started, 3),
                            "seconds": round(time.time() - started, 3),
                            "request": _parsed(body or b""), "status": status,
                            "response": None if error else _parsed(out)}
                    if error:
                        line["error"] = error
                    recorder.write(line)
                self._reply(status, ctype, out)

            def _reply(self, status: int, ctype: str, body: bytes):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_GET(self):
                self._forward("GET")

            def do_POST(self):
                self._forward("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def write(self, line: dict) -> None:
        text = json.dumps(line, ensure_ascii=False, sort_keys=True)
        with self.lock:
            self.log.write(text + "\n")
            self.log.flush()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.log.close()
