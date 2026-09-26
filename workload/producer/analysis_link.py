"""The producer's end of the analysis socket (analysis/protocol.md).

One connection per session. `send` is the only way anything reaches the
analysis process and `on_message` the only way anything comes back; both
carry JSON objects, one per line, and this module neither inspects nor
builds their contents beyond the `kind` field. What is allowed to cross is
the protocol document's business, and the producer's (`producer.Session`),
which decides what to put in and what to do with what comes out.

`NullLink` stands in when no socket is configured: the producer then runs
the pixel path alone, which is how the public tree is exercised without the
analysis bundle.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Callable

PROTOCOL = 1

# How long a connect may wait for the analysis process to be listening (it
# starts alongside the producer at boot and is up long before a session
# arrives), and how long `stop` waits for the summary.
CONNECT_TIMEOUT_S = 10.0
READY_TIMEOUT_S = 30.0
SUMMARY_TIMEOUT_S = 10.0


class LinkError(RuntimeError):
    pass


class NullLink:
    """No analysis process: everything sent is dropped, nothing comes back."""

    connected = False
    ready: dict | None = None

    def send(self, message: dict) -> None:
        pass

    def stop(self, at_s: float | None = None) -> dict:
        return {}

    def close(self) -> None:
        pass


class AnalysisLink:
    """A connected session with the analysis process."""

    connected = True

    def __init__(self, path: str, on_message: Callable[[dict], None],
                 on_error: Callable[[str], None] | None = None,
                 connect_timeout_s: float = CONNECT_TIMEOUT_S):
        self.path = path
        self.on_message = on_message
        self.on_error = on_error or (lambda text: None)
        self.ready: dict | None = None
        self._summary: dict | None = None
        self._summary_event = threading.Event()
        self._ready_event = threading.Event()
        self._send_lock = threading.Lock()
        self._closed = False
        self.sock = self._connect(connect_timeout_s)
        self._file = self.sock.makefile("rb")
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _connect(self, timeout_s: float) -> socket.socket:
        deadline = time.monotonic() + timeout_s
        last: Exception | None = None
        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.path)
                return sock
            except OSError as error:
                sock.close()
                last = error
                if time.monotonic() >= deadline:
                    raise LinkError(
                        f"analysis socket {self.path}: {error}") from last
                time.sleep(0.25)

    def hello(self, **fields) -> dict:
        """Send `hello`, wait for `ready`."""
        self.send({"kind": "hello", "protocol": PROTOCOL, **fields})
        if not self._ready_event.wait(READY_TIMEOUT_S):
            raise LinkError("analysis did not answer hello")
        assert self.ready is not None
        if int(self.ready.get("protocol", 0)) != PROTOCOL:
            raise LinkError(
                f"analysis speaks protocol {self.ready.get('protocol')}, "
                f"this producer speaks {PROTOCOL}")
        return self.ready

    def send(self, message: dict) -> None:
        if self._closed:
            return
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        with self._send_lock:
            try:
                self.sock.sendall(data)
            except OSError as error:
                self._closed = True
                self.on_error(f"analysis send failed: {error!r}")

    def stop(self, at_s: float | None = None,
             timeout_s: float = SUMMARY_TIMEOUT_S) -> dict:
        """Send `stop`, return the analysis's summary (empty if none came)."""
        self.send({"kind": "stop", "atS": at_s})
        self._summary_event.wait(timeout_s)
        self.close()
        return dict(self._summary or {})

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def _read_loop(self) -> None:
        try:
            for line in self._file:
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    self.on_error("analysis sent a line that is not JSON")
                    continue
                if not isinstance(message, dict):
                    continue
                kind = message.get("kind")
                if kind == "ready":
                    self.ready = message
                    self._ready_event.set()
                    continue
                if kind == "summary":
                    summary = message.get("summary")
                    self._summary = summary if isinstance(summary, dict) else {}
                    self._summary_event.set()
                    continue
                try:
                    self.on_message(message)
                except Exception as error:  # noqa: BLE001 - said aloud, survived
                    self.on_error(f"analysis message {kind!r} failed: {error!r}")
        except (OSError, ValueError):
            pass
        finally:
            # A reader that ends without a summary must not hang `stop`.
            if self._summary is None:
                self._summary = {}
            self._summary_event.set()
            self._ready_event.set()
            if self.ready is None:
                self.ready = {}


def open_link(path: str, on_message: Callable[[dict], None],
              on_error: Callable[[str], None] | None = None,
              **hello_fields) -> AnalysisLink | NullLink:
    """Connect and greet, or the null link when no path is configured."""
    if not path:
        return NullLink()
    link = AnalysisLink(path, on_message, on_error)
    try:
        link.hello(**hello_fields)
    except LinkError:
        link.close()
        raise
    return link
