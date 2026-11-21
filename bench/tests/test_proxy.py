"""Regression tests for the loopback recorder."""

import http.client
import threading
from contextlib import ExitStack
from http.server import ThreadingHTTPServer

from bench import proxy


def test_recorder_accepts_a_concurrent_connection_burst(tmp_path, monkeypatch):
    # Pause accepts so the test exercises the listener backlog even on a fast host.
    accept = threading.Event()
    serve_forever = ThreadingHTTPServer.serve_forever

    def paused_server(self):
        accept.wait()
        serve_forever(self)

    monkeypatch.setattr(ThreadingHTTPServer, "serve_forever", paused_server)
    monkeypatch.setattr(proxy, "call", lambda *args: (200, "text/plain", b"ok"))
    recorder = proxy.Recorder(0, tmp_path / "requests.jsonl")
    try:
        with ExitStack() as clients:
            burst = []
            for _ in range(16):
                conn = http.client.HTTPConnection("127.0.0.1", recorder.port, timeout=2)
                clients.callback(conn.close)
                conn.request("GET", "/health")
                burst.append(conn)
            accept.set()
            for conn in burst:
                reply = conn.getresponse()
                assert (reply.status, reply.read()) == (200, b"ok")
    finally:
        accept.set()
        recorder.close()
