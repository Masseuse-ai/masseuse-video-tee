"""The camlink gateway's loopback control API, as the producer drives it.

A camera on the user's home network cannot be dialled from the enclave:
its address is private. The connector (`masseuse-camlink`, a program on a
computer in that network) relays the camera's TLS ciphertext into the
enclave, where a fourth process, `masseuse-camlink-gateway`, accepts
exactly one connector per session and exposes two loopback ends:

  control     http://127.0.0.1:8091 (CAMLINK_GATEWAY_CONTROL)
              POST /expect {connectorKey, ticketHash, expiresAt}: which
                           connector to accept, and with which ticket
              POST /target {host, port}: the camera the connector dials for
                           each relayed connection; 409 with no connector
              POST /clear  forget both, drop the connector
              GET  /status {expecting, connected, sinceMs,
                            connectorKeyPrefix, target}
  relay       127.0.0.1:7441: every TCP connection accepted here is
              carried through the connector to the target. MediaMTX is
              pointed at rtsps://127.0.0.1:7441/<path> and completes the
              camera's own TLS through it (external_source.py).

The trainer reaches /expect and /clear through the producer's
authenticated control routes (/tunnel/expect and /tunnel/clear in
producer.py); /target and /status are the producer's own, driven by the
external camera. Neither the connector nor the gateway decrypts anything;
the target host goes to the gateway and appears in no log or status.
(masseuse-camlink/docs/PROTOCOL.md, section 5.)
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

from tee_mode import SESSION_ID

CONTROL_ENV = "CAMLINK_GATEWAY_CONTROL"
DEFAULT_CONTROL = "http://127.0.0.1:8091"
RELAY_HOST = "127.0.0.1"
RELAY_PORT = 7441
TIMEOUT_S = 3.0
STATUS_CACHE_S = 2.0
# What the trainer's /tunnel/expect carries: the connector's Ed25519 public
# key (32 bytes, base64url unpadded), the SHA-256 of the one-time ticket
# (hex) and when it lapses (unix seconds, bounded like a lease).
CONNECTOR_KEY = re.compile(r"^[A-Za-z0-9_-]{43}$")
TICKET_HASH = re.compile(r"^[0-9a-f]{64}$")
EXPECT_MAX_S = 6 * 3600.0
# What the status carries when the gateway cannot say otherwise.
OFFLINE = {"connected": False, "sinceMs": None}


class GatewayError(Exception):
    """The gateway did not answer (`status` None) or answered with an error
    (`status` the HTTP code, 409 being "no connector attached")."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def unreachable(self) -> bool:
        return self.status is None


def parse_expectation(body: dict, now: float) -> tuple[str, dict]:
    """(session id, the gateway's /expect body) from the trainer's
    /tunnel/expect body, or a ValueError naming the first field that is
    wrong. The session id is the caller's to check against the lease."""
    connector_key = str(body.get("connectorKey") or "")
    if not CONNECTOR_KEY.match(connector_key):
        raise ValueError("connectorKey must be the base64url of 32 bytes, unpadded")
    ticket_hash = str(body.get("ticketHash") or "").lower()
    if not TICKET_HASH.match(ticket_hash):
        raise ValueError("ticketHash must be 64 hex characters (sha256)")
    expires_at = body.get("expiresAt")
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        raise ValueError("expiresAt must be a unix time in seconds")
    expires_at = int(expires_at)
    if not now < expires_at <= now + EXPECT_MAX_S:
        raise ValueError("expiresAt must be in the future and within six hours, "
                         "in seconds")
    session_id = body.get("sessionId")
    if not isinstance(session_id, str) or not SESSION_ID.match(session_id):
        raise ValueError("sessionId must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    return session_id, {"connectorKey": connector_key, "ticketHash": ticket_hash,
                        "expiresAt": expires_at}


def tunnel_view(info: dict | None) -> dict:
    """The `tunnel` object of a source status - {connected, sinceMs} - from
    the gateway's /status, or OFFLINE when there was no answer."""
    if not isinstance(info, dict):
        return dict(OFFLINE)
    since = info.get("sinceMs")
    connected = bool(info.get("connected"))
    if isinstance(since, bool) or not isinstance(since, (int, float)) or not connected:
        since = None
    return {"connected": connected,
            "sinceMs": int(since) if since is not None else None}


class CamlinkGateway:
    """The control API client. `control` is the base URL (the environment's
    CAMLINK_GATEWAY_CONTROL through `from_env`, the loopback default
    otherwise); `opener` and `clock` are replaceable by a test.

    `session_id` remembers which session the current expectation was
    posted for, so a lease for another session drops it; `clear_quietly`
    is the best-effort form the teardown hooks use, and it posts only when
    something may be there to clear (after boot, an expect or a target)
    rather than on every idle teardown.
    """

    def __init__(self, control: str = DEFAULT_CONTROL, *, timeout_s: float = TIMEOUT_S,
                 status_cache_s: float = STATUS_CACHE_S,
                 opener=urllib.request.urlopen, clock=time.monotonic, log=print):
        self.control = (control or DEFAULT_CONTROL).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.status_cache_s = float(status_cache_s)
        self.opener = opener
        self.clock = clock
        self.log = log
        self.session_id = ""
        self._lock = threading.Lock()
        self._status: dict | None = None
        self._status_at: float | None = None
        self._engaged = True

    @classmethod
    def from_env(cls, env=os.environ, **options) -> "CamlinkGateway":
        return cls(env.get(CONTROL_ENV) or DEFAULT_CONTROL, **options)

    # -- what the trainer asks for -------------------------------------------------

    def expect(self, expectation: dict) -> None:
        """POST /expect with a body `parse_expectation` produced."""
        self._post("/expect", expectation)
        self._engaged = True
        self._forget()

    def clear(self) -> None:
        """POST /clear: the gateway forgets the expectation and the target
        and drops the connector."""
        self._post("/clear")
        self.session_id = ""
        self._engaged = False
        self._forget()

    def clear_quietly(self, reason: str) -> bool:
        """`clear` for a hook (teardown, a lease for another session, the
        external camera going): never raises, logs what happened. True
        when the gateway was told."""
        if not self._engaged:
            return False
        try:
            self.clear()
        except GatewayError as error:
            self.log(f"tunnel: clear ({reason}) did not reach the gateway: {error}",
                     flush=True)
            return False
        self.log(f"tunnel: cleared ({reason})", flush=True)
        return True

    # -- what the external camera asks for -----------------------------------------

    def target(self, host: str, port: int) -> None:
        """POST /target: where the connector dials for the next relayed
        connections. A GatewayError with status 409 is "no connector"."""
        self._post("/target", {"host": host, "port": int(port)})
        self._engaged = True

    def status(self) -> dict | None:
        """The gateway's /status, fresh: None when it is not answering.
        Either way it is what `tunnel()` reads for the next while."""
        try:
            with self.opener(f"{self.control}/status",
                             timeout=self.timeout_s) as response:
                info = json.loads(response.read() or b"{}")
            if not isinstance(info, dict):
                info = None
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            info = None
        with self._lock:
            self._status, self._status_at = info, self.clock()
        return info

    def tunnel(self) -> dict:
        """{connected, sinceMs} for /ingest/status and the phone's
        /ingest/source, from a /status at most `status_cache_s` old: the
        trainer polls, the gateway need not answer every poll."""
        with self._lock:
            info, at = self._status, self._status_at
        if at is None or self.clock() - at >= self.status_cache_s:
            info = self.status()
        return tunnel_view(info)

    # -- the wire ------------------------------------------------------------------

    def _forget(self) -> None:
        with self._lock:
            self._status, self._status_at = None, None

    def _post(self, route: str, body: dict | None = None) -> None:
        payload = json.dumps(body).encode() if body is not None else b""
        request = urllib.request.Request(
            f"{self.control}{route}", data=payload, method="POST",
            headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            with self.opener(request, timeout=self.timeout_s) as response:
                code = int(response.status)
        except urllib.error.HTTPError as error:
            code = int(error.code)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            raise GatewayError("the tunnel gateway is not answering") from None
        if not 200 <= code < 300:
            raise GatewayError(f"the tunnel gateway answered {code} to {route}", code)
