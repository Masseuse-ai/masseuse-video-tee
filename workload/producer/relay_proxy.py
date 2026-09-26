"""WebRTC signalling through the producer's one port: WHEP out for the live
overlay, WHIP in for a phone's camera.

Cloud Run exposes a single HTTP port per service and no UDP ingress, so
the relay sidecar (MediaMTX) can be reached only through this process:
its WebRTC endpoint and its control API listen on loopback, and the
producer forwards the handful of signalling requests each leg needs.

    OPTIONS/POST   /overlay/whep           -> relay  /overlay/whep
    PATCH/DELETE   /overlay/whep/<secret>  -> relay  /overlay/whep/<secret>
    GET            /overlay/status         -> relay API paths/get/overlay
    OPTIONS/POST   /ingest/whip            -> relay  /cam/whip
    PATCH/DELETE   /ingest/whip/<secret>   -> relay  /cam/whip/<secret>
    GET            /ingest/status          -> relay API paths/get/<active source>
    PUT/GET/DELETE /ingest/source          the external camera (external_source.py;
                                           TEE only, handled in producer.py)

The relay answers a POST with a `Location` for the session it created; it
comes back rewritten into this proxy's own namespace (`/overlay/whep/<s>`,
`/ingest/whip/<s>`) whatever shape the relay used, so the client follows
it through the same port. Signalling is the only thing that crosses here;
the media path (ICE, DTLS, SRTP) is between the browser and the relay's
ICE candidates, which on Cloud Run means a TURN relay both ends reach.

The overlay leg exists only while a session is publishing (a subscriber
before the session gets a clear 503). The ingest leg is the opposite: the
phone publishes its camera first, and the session that reads it
(`--stream rtsp://127.0.0.1:8554/cam`) is started once /ingest/status
reports the track - so the ingest routes never look at the session.

A client's backend authenticates to this service with an IAM OIDC token;
a browser cannot (the preflight carries no bearer), so that backend
proxies these same routes onward. Nothing here reads Authorization: IAM
already did.

On the Confidential Space slot (producer.py --tee) the phone reaches these
routes directly over the enclave's own TLS, so `public_origin` makes the
rewritten Location absolute on that origin, and `occupied()` lets the
handler hold the relay to one publisher and one subscriber per lease. The
bearer check and the signed evidence live in tee_mode.py; this module
stays the plain forwarder.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit

WHEP_SECRET = re.compile(r"^[A-Za-z0-9-]{1,64}$")
WHEP_METHODS = frozenset({"OPTIONS", "POST", "PATCH", "DELETE"})
# The two legs this proxy carries: (route prefix, relay verb).
LEGS = {"whep": "whep", "whip": "whip"}
_LOCATION_SECRET = re.compile(r"(?:^|/)(?:whep|whip)/([A-Za-z0-9-]{1,64})/?$")
# What a WHEP client sends that the relay reads. Authorization is not on
# the list: the relay does not authenticate, IAM already has.
REQUEST_HEADERS = ("Content-Type", "Accept", "If-Match", "Origin",
                   "Access-Control-Request-Method",
                   "Access-Control-Request-Headers", "User-Agent")
# What the relay answers that a WHEP client reads (draft-ietf-wish-whep):
# the SDP answer type, the session Location, ICE servers in Link, ETag for
# trickle ICE, plus whatever CORS the relay adds for the browser.
RESPONSE_HEADERS = ("Content-Type", "Location", "Link", "ETag", "ID",
                    "Accept-Post", "Accept-Patch",
                    "Access-Control-Allow-Origin",
                    "Access-Control-Allow-Methods",
                    "Access-Control-Allow-Headers",
                    "Access-Control-Expose-Headers",
                    "Access-Control-Allow-Credentials",
                    "Access-Control-Max-Age")
BODY_CAP = 64 * 1024  # an SDP offer is a few KB; a trickle fragment less


def route(path: str) -> tuple[str, str | None] | None:
    """The proxy's route table, or None for anything else.

    ('status', None)         GET  /overlay/status
    ('whep', None|secret)    /overlay/whep[/<secret>]
    ('ingest-status', None)  GET  /ingest/status
    ('whip', None|secret)    /ingest/whip[/<secret>]
    ('source', None)         PUT/GET/DELETE /ingest/source
    ('view', None)           PUT/GET /ingest/view
    """
    if path == "/overlay/status":
        return "status", None
    if path == "/overlay/whep":
        return "whep", None
    if path == "/ingest/status":
        return "ingest-status", None
    if path == "/ingest/whip":
        return "whip", None
    if path == "/ingest/source":
        return "source", None
    if path == "/ingest/view":
        return "view", None
    for prefix, kind in (("/overlay/whep/", "whep"), ("/ingest/whip/", "whip")):
        if path.startswith(prefix) and WHEP_SECRET.match(path[len(prefix):]):
            return kind, path[len(prefix):]
    return None


def own_location(value: str, leg: str, public_origin: str = "") -> str:
    """The relay's session Location, in this proxy's namespace.

    MediaMTX answers a WHIP/WHEP POST with a Location that is the bare
    session secret (a relative reference), or a path ending in
    `/<verb>/<secret>`. Either becomes `/overlay/whep/<secret>` or
    `/ingest/whip/<secret>`; anything else is passed through. With a
    `public_origin` the result is absolute (`https://slot/ingest/whip/<s>`):
    a browser following it from another origin must not resolve it
    against the page.
    """
    prefix = "/overlay/whep/" if leg == "whep" else "/ingest/whip/"
    candidate = value.strip()
    if WHEP_SECRET.match(candidate):
        return public_origin + prefix + candidate
    try:
        path = urlsplit(candidate).path
    except ValueError:
        return value
    found = _LOCATION_SECRET.search(path)
    return public_origin + prefix + found.group(1) if found else value


def location_secret(value: str) -> str | None:
    """The session secret a (rewritten or raw) Location names, or None."""
    try:
        path = urlsplit(value.strip()).path
    except ValueError:
        return None
    found = _LOCATION_SECRET.search(path)
    return found.group(1) if found else None


class RelayProxy:
    def __init__(self, webrtc_base: str = "http://127.0.0.1:8889",
                 api_base: str = "http://127.0.0.1:9997",
                 path: str = "overlay", timeout_s: float = 10.0,
                 body_cap: int = BODY_CAP, opener=urllib.request.urlopen,
                 ingest_path: str = "cam", public_origin: str = "",
                 rtsp_base: str = "rtsp://127.0.0.1:8554"):
        self.webrtc_base = webrtc_base.rstrip("/")
        self.api_base = api_base.rstrip("/")
        self.path = path
        self.ingest_path = ingest_path
        self.timeout_s = timeout_s
        self.body_cap = body_cap
        self.opener = opener
        self.public_origin = public_origin.rstrip("/")
        # Where the producer reads a relay path back: the loopback RTSP
        # listener, so `/ingest/status` can name the stream to /produce.
        self.rtsp_base = rtsp_base.rstrip("/")

    # -- WHEP / WHIP -----------------------------------------------------------

    def forward(self, method: str, secret: str | None, headers, body: bytes,
                client: str = "", leg: str = "whep",
                ) -> tuple[int, list[tuple[str, str]], bytes]:
        """One signalling request to the relay; (status, headers, body) back.

        `leg` is 'whep' (the overlay out, default) or 'whip' (the camera in).
        """
        if leg not in LEGS:
            raise ValueError(f"unknown leg {leg!r}")
        if method not in WHEP_METHODS:
            return 405, [("Allow", ", ".join(sorted(WHEP_METHODS)))], b""
        if secret is None and method not in ("OPTIONS", "POST"):
            return 405, [("Allow", "OPTIONS, POST")], b""
        if secret is not None and method not in ("OPTIONS", "PATCH", "DELETE"):
            return 405, [("Allow", "OPTIONS, PATCH, DELETE")], b""
        if len(body) > self.body_cap:
            return 413, [], b""
        relay_path = self.ingest_path if leg == "whip" else self.path
        url = f"{self.webrtc_base}/{relay_path}/{LEGS[leg]}"
        if secret is not None:
            url += f"/{secret}"
        forwarded = {}
        for name in REQUEST_HEADERS:
            value = headers.get(name)
            if value:
                forwarded[name] = value
        if client:
            forwarded["X-Forwarded-For"] = client
        request = urllib.request.Request(
            url, data=body if body else None, headers=forwarded, method=method)
        try:
            with self.opener(request, timeout=self.timeout_s) as response:
                return (response.status, self._headers(response.headers, leg),
                        response.read())
        except urllib.error.HTTPError as error:
            # 4xx/5xx from the relay are answers, not failures: a stale
            # secret is a 404 the client must see.
            return error.code, self._headers(error.headers, leg), error.read()
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            return 502, [("Content-Type", "application/json")], json.dumps(
                {"error": "relay unreachable", "detail": str(error)}).encode()

    def webrtc_sessions(self, leg: str) -> list[str]:
        """The relay's WebRTC session ids currently on a leg: the publisher
        of the camera path, or the readers of the overlay. Empty when the
        relay is unreachable or the path is idle."""
        path = self.ingest_path if leg == "whip" else self.path
        try:
            with self.opener(f"{self.api_base}/v3/paths/get/{path}",
                             timeout=self.timeout_s) as response:
                info = json.loads(response.read() or b"{}")
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return []
        if leg == "whip":
            source = info.get("source") or {}
            if source.get("type") == "webRTCSession" and source.get("id"):
                return [str(source["id"])]
            return []
        return [str(reader["id"]) for reader in info.get("readers") or []
                if reader.get("type") == "webRTCSession" and reader.get("id")]

    def evict(self, leg: str) -> int:
        """Kick whatever WebRTC session holds a leg, so a new offer from the
        lease's holder takes it over: one publisher and one subscriber per
        lease, and a phone that reconnects (page reload, network change) is
        not locked out by its own lingering session for the relay's
        timeout. Returns how many sessions were kicked."""
        kicked = 0
        for session_id in self.webrtc_sessions(leg):
            request = urllib.request.Request(
                f"{self.api_base}/v3/webrtcsessions/kick/{session_id}",
                method="POST")
            try:
                with self.opener(request, timeout=self.timeout_s):
                    kicked += 1
            except urllib.error.HTTPError as error:
                if error.code != 404:  # already gone is fine
                    raise
            except (urllib.error.URLError, OSError, TimeoutError):
                break
        return kicked

    def _headers(self, headers, leg: str = "whep") -> list[tuple[str, str]]:
        out = []
        for name in RESPONSE_HEADERS:
            for value in headers.get_all(name) or []:
                if name == "Location":
                    value = own_location(value, leg, self.public_origin)
                out.append((name, value))
        return out

    # -- status ----------------------------------------------------------------

    def relay_status(self, path: str | None = None) -> dict | None:
        """The relay's view of a path (the overlay by default), or None if
        the relay is unreachable. A path the relay has never seen reads as
        not ready with no tracks."""
        url = f"{self.api_base}/v3/paths/get/{path or self.path}"
        try:
            with self.opener(url, timeout=self.timeout_s) as response:
                info = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {"ready": False, "readers": 0, "tracks": [],
                        "bytesReceived": 0}
            return None
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return None
        return {
            "ready": bool(info.get("ready")),
            "readers": len(info.get("readers") or []),
            "tracks": info.get("tracks") or [],
            "bytesReceived": int(info.get("bytesReceived") or 0),
            "source": (info.get("source") or {}).get("type"),
        }

    def status(self, session) -> tuple[int, dict]:
        """(status code, body) for GET /overlay/status.

        503 when no session is publishing: a subscriber polling before the
        session starts gets a clear 'not yet' rather than a WHEP 404 from
        the relay. 502 when the relay itself cannot be reached.
        """
        overlay = getattr(session, "overlay", None) if session else None
        relay = self.relay_status()
        body = {
            "session": (getattr(session, "run_name", "") or None) if session else None,
            "publishing": overlay is not None,
            "overlay": overlay.snapshot() if overlay is not None else None,
            "relay": relay,
            "whep": "/overlay/whep",
        }
        if relay is None:
            body["error"] = "relay unreachable"
            return 502, body
        if overlay is None:
            body["error"] = ("no session is running"
                             if session is None else "session has no overlay")
            return 503, body
        return 200, body

    def ingest_status(self, external=None) -> tuple[int, dict]:
        """(status code, body) for GET /ingest/status: the relay's view of
        the session's camera, flattened so `ready` and `tracks` sit at the
        top, plus which camera that is.

        The camera is the phone's (`cam`) unless `external` (an
        external_source.ExternalSource) is active, in which case the top
        level follows its path and `streamUrl` names it for /produce:
        `source` says which (`kind`, `path`, `ready`, `tracks`, the
        relay's source `type`, `since`) and `phone` keeps the phone's own
        path in view either way. On a slot with an external camera at all
        `source` also carries how it is reached (`mode`: direct, tunnel,
        or null) and the connector's `tunnel` {connected, sinceMs}, as the
        external source reports them.

        200 whether or not anything is publishing - the caller polls this
        to learn when the track has landed, so 'not yet' is a normal answer
        (`ready: false`), not an error. 502 only when the relay itself
        cannot be reached.
        """
        phone = self.relay_status(self.ingest_path)
        if phone is None:
            return 502, {"ready": False, "tracks": [], "path": self.ingest_path,
                         "whip": "/ingest/whip", "error": "relay unreachable"}
        kind, path, since = "phone", self.ingest_path, None
        active = phone
        if external is not None and external.active:
            kind, path, since = "external", external.path, external.since
            active = self.relay_status(path) or {
                "ready": False, "readers": 0, "tracks": [], "bytesReceived": 0,
                "source": None}
        body = dict(active)
        body["path"] = path
        body["streamUrl"] = f"{self.rtsp_base}/{path}"
        body["source"] = {"kind": kind, "path": path, "ready": bool(active.get("ready")),
                          "tracks": list(active.get("tracks") or []),
                          "type": active.get("source"), "since": since}
        if external is not None:
            # Passed through as the external source reports them: the trainer
            # reads `tunnel` to know whether a camera at home can be reached.
            body["source"]["mode"] = external.mode
            body["source"]["tunnel"] = external.tunnel_status()
        body["phone"] = {"ready": bool(phone.get("ready")),
                         "tracks": list(phone.get("tracks") or [])}
        body["whip"] = "/ingest/whip"
        return 200, body
