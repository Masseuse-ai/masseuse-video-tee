"""A camera the phone names instead of the one it carries.

Most people hold their phone during a session; the camera that should be
watching their back, hip region, legs and feet is a fixed one behind them,
and it speaks RTSPS. The phone pastes its link into the page and the page
hands it straight to the enclave (`PUT /ingest/source`, capability bearer,
over the slot's own TLS, after the phone has verified the attestation):
the trainer and Cloudflare never see it. The enclave asks the relay
(MediaMTX, loopback API) to pull the link as a path of its own, `ext`,
and the producer reads that path over loopback RTSP exactly as it reads
the phone's `cam` path. Nothing about the link - not the host - is
written to a log or handed to anyone.

What the enclave checks before it connects:

  scheme      rtsps:// only. A plain rtsp:// link would carry the video
              across the internet in the clear.
  address     the host must resolve to a public address: RFC 1918,
              loopback, link-local (the metadata server), multicast, the
              shared range and the VM's own public IP are refused with a
              reason the page can show. A camera on a home LAN is named
              by its LAN address or .local name instead and reached
              through the user's connector (tunnel mode, below).
  the leaf    a TLS handshake to the resolved address (SNI: the URL's
              host) fetches the camera's certificate; its SHA-256 goes to
              the relay as `sourceFingerprint`. Cameras ship self-signed
              certificates, so this is trust-on-first-use, and it is also
              what closes the DNS-rebinding gap between this check and the
              relay's own resolution: whatever the name resolves to a
              moment later, the relay completes the handshake only with
              the certificate seen here. (The handshake here doubles as
              the reachability probe: five seconds, then `unreachable`.)

Then `POST /v3/config/paths/add/ext {source, sourceFingerprint,
rtspTransport: tcp, sourceOnDemand: false}` and a poll of the path until
the relay reports it ready with a video track, fifteen seconds at most;
a failure deletes the path again and says why (`timeout`, `no-video`).
The path lives in the relay's memory for the life of the lease: cleared
on /teardown, on a lease for a different session, and on the phone's
DELETE. Not on /stop, which is how the trainer restarts a production.

A camera at home (tunnel mode). A host that is a private literal (RFC
1918, loopback, link-local, and their IPv6 kin), a `.local` name or a
bare name with no dot is not resolved or dialled from here at all: it is
reached through the user's connector (camlink_gateway.py). The gateway
must report a connector attached - `tunnel-offline` otherwise, before
anything else is touched - and is told the host and port as the target;
the TLS probe then goes to the gateway's relay listener, 127.0.0.1:7441,
with no SNI, and the leaf it returns is the camera's own, exactly the
bytes a direct dial would see, so the pin is the same pin. The relay is
pointed at `rtsps://[user:pass@]127.0.0.1:7441/<path>?<query>`: the
camera's address goes to the gateway and nowhere else. A public host
keeps the direct dial above, unchanged.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from camlink_gateway import OFFLINE, RELAY_HOST, RELAY_PORT, GatewayError

EXTERNAL_PATH = "ext"
URL_CAP = 2048
DEFAULT_RTSPS_PORT = 322
PROBE_TIMEOUT_S = 5.0
CONNECT_TIMEOUT_S = 15.0
POLL_S = 0.5
# The relay names tracks by codec (MediaMTX `tracks`); anything on this
# list is a picture the producer can decode.
VIDEO_CODECS = ("H264", "H265", "AV1", "VP9", "VP8", "M-JPEG", "MJPEG",
                "MPEG-4 VIDEO", "MPEG-1/2 VIDEO")
# What a camera on a home network is addressed by, and so what goes through
# the connector rather than a direct dial: the same list the connector
# itself accepts as a target (PROTOCOL.md section 4).
TUNNEL_NETWORKS = tuple(ipaddress.ip_network(cidr) for cidr in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8",
    "169.254.0.0/16", "::1/128", "fc00::/7", "fe80::/10"))
# How the probe's refusals read when the connector, not the camera, was
# what the enclave reached; the reasons stay the ones the page knows.
TUNNEL_PROBE_WORDING = {
    "not-tls": "the camera, reached through the connector, did not speak TLS: is "
               "the link rtsps, not rtsp, and the port the camera's secure one?",
    "unreachable": "the connector could not reach the camera: check the address "
                   "and port, and that the connector runs in the camera's network",
}


class SourceError(Exception):
    """A refusal or a failure, with a reason code the page shows the user
    and the HTTP status the handler answers with."""

    def __init__(self, reason: str, message: str, status: int = 400):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.status = status

    def body(self) -> dict:
        return {"status": "failed", "reason": self.reason, "error": self.message}


def is_video(codec: str) -> bool:
    upper = str(codec or "").upper()
    return any(upper == name or upper.startswith(name) for name in VIDEO_CODECS)


def _refused_address(ip) -> bool:
    """Anything that is not a public unicast address."""
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_unspecified or ip.is_reserved
            or not ip.is_global)


def is_private_host(host: str) -> bool:
    """Is this a host the connector is for: a literal RFC 1918, loopback or
    link-local address (v4, v6 or v4-mapped), a `.local` name, or a bare
    name with no dot - what a camera on a home network is called. Anything
    else is dialled directly and must pass `validate`."""
    name = str(host or "").strip().lower().rstrip(".")
    if not name:
        return False
    try:
        ip = ipaddress.ip_address(name.split("%", 1)[0])
    except ValueError:
        return name.endswith(".local") or "." not in name
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return any(ip in network for network in TUNNEL_NETWORKS)


def parse_link(url: str) -> tuple[str, str, int]:
    """(the link, stripped; its host; its port) for anything shaped like a
    camera's rtsps:// link, or a SourceError saying what is wrong with it.
    Nothing here touches the network."""
    if not isinstance(url, str) or not url.strip():
        raise SourceError("bad-url", "paste the camera's rtsps:// link")
    url = url.strip()
    if len(url) > URL_CAP:
        raise SourceError("bad-url", f"the link is longer than {URL_CAP} characters")
    if any(ch.isspace() or ord(ch) < 32 for ch in url):
        raise SourceError("bad-url", "the link contains spaces or control characters")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError as error:
        raise SourceError("bad-url", f"that is not a valid link ({error})") from None
    if parts.scheme.lower() != "rtsps":
        raise SourceError("not-rtsps", "only rtsps:// links are accepted: a plain "
                          "rtsp:// link would send the video across the internet "
                          "unencrypted")
    if not host:
        raise SourceError("bad-url", "the link names no host")
    return url, host, int(port or DEFAULT_RTSPS_PORT)


def resolve_public(host: str, port: int, own_ip: str = "",
                   resolver=socket.getaddrinfo) -> str:
    """The address a camera the enclave dials directly resolves to, or a
    SourceError saying why not. Every resolved address must pass: a name
    that also resolves to something private is refused whole."""
    try:
        found = resolver(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        raise SourceError("unreachable", f"{host} does not resolve", 502) from None
    addresses = []
    for entry in found:
        candidate = entry[4][0] if len(entry) >= 5 else None
        if not candidate:
            continue
        try:
            ip = ipaddress.ip_address(candidate.split("%", 1)[0])
        except ValueError:
            continue
        addresses.append(ip)
    if not addresses:
        raise SourceError("unreachable", f"{host} does not resolve", 502)
    own = None
    if own_ip:
        try:
            own = ipaddress.ip_address(own_ip)
        except ValueError:
            own = None
    for ip in addresses:
        if _refused_address(ip) or (own is not None and ip == own):
            raise SourceError("private-address",
                              "the camera must be reachable from the internet: "
                              "that address is private, local, or this slot's own "
                              "(a camera at home is named by its LAN address or "
                              ".local name and reached through the connector)")
    return str(addresses[0])


def validate(url: str, own_ip: str = "",
             resolver=socket.getaddrinfo) -> tuple[str, int, str]:
    """(host, port, resolved ip) for a link the enclave may dial directly,
    or a SourceError saying why not: `parse_link` then `resolve_public`."""
    _, host, port = parse_link(url)
    return host, port, resolve_public(host, port, own_ip, resolver)


def tunnel_source_url(url: str) -> str:
    """The pasted link with the camera's address replaced by the gateway's
    relay listener, `rtsps://[user:pass@]127.0.0.1:7441/<path>?<query>`:
    what the relay pulls in tunnel mode. Credentials, path and query go
    through as they were; the host does not go to the relay at all."""
    parts = urlsplit(url.strip())
    userinfo, at, _ = parts.netloc.rpartition("@")
    netloc = f"{RELAY_HOST}:{RELAY_PORT}"
    if at:
        netloc = f"{userinfo}@{netloc}"
    return urlunsplit(("rtsps", netloc, parts.path, parts.query, parts.fragment))


def tls_fingerprint(host: str, port: int, ip: str,
                    timeout_s: float = PROBE_TIMEOUT_S, *, sni: bool = True) -> str:
    """SHA-256 of the DER leaf the camera presents, as 64 uppercase hex
    characters (what MediaMTX's `sourceFingerprint` takes: `openssl x509
    -fingerprint -sha256` with the colons removed). The connection goes to
    the address `validate` resolved, with the URL's host as the SNI - or
    none, for a literal address or when `sni` is False (the probe through
    the gateway's relay listener, which the camera must not be named to)."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    server_name = host if sni else None
    if server_name is not None:
        try:
            ipaddress.ip_address(host)
            server_name = None  # no SNI for a literal address
        except ValueError:
            pass
    try:
        with socket.create_connection((ip, port), timeout=timeout_s) as raw:
            with context.wrap_socket(raw, server_hostname=server_name) as tls:
                der = tls.getpeercert(binary_form=True)
    except ssl.SSLError:
        raise SourceError("not-tls", f"{host}:{port} does not speak TLS: is the "
                          "link rtsps, not rtsp, and the port the camera's "
                          "secure one?", 502) from None
    except (TimeoutError, socket.timeout):
        raise SourceError("unreachable", f"{host}:{port} did not answer within "
                          f"{timeout_s:.0f}s", 502) from None
    except OSError as error:
        raise SourceError("unreachable", f"{host}:{port} refused the connection "
                          f"({type(error).__name__})", 502) from None
    if not der:
        raise SourceError("not-tls", f"{host}:{port} presented no certificate", 502)
    return hashlib.sha256(der).hexdigest().upper()


class ExternalSource:
    """The one external camera a slot may pull, and its relay path.

    `relay` is the RelayProxy: its API base, opener and timeout reach the
    relay's control API, and `relay_status` reads the path back. `own_ip`
    is TEE_PUBLIC_IP. `gateway` is the connector's CamlinkGateway (None on
    a slot with no tunnel: a camera at home is then `tunnel-offline`).
    `resolver` and `probe` are `validate`'s and `tls_fingerprint`'s
    network calls, replaceable by a test.
    """

    def __init__(self, relay, *, own_ip: str = "", path: str = EXTERNAL_PATH,
                 rtsp_base: str = "rtsp://127.0.0.1:8554",
                 connect_timeout_s: float = CONNECT_TIMEOUT_S,
                 probe_timeout_s: float = PROBE_TIMEOUT_S, poll_s: float = POLL_S,
                 resolver=socket.getaddrinfo, probe=tls_fingerprint, gateway=None,
                 clock=time.time, monotonic=time.monotonic, sleep=time.sleep,
                 log=print):
        self.relay = relay
        self.own_ip = own_ip or ""
        self.path = path
        self.rtsp_base = rtsp_base.rstrip("/")
        self.stream_url = f"{self.rtsp_base}/{path}"
        self.connect_timeout_s = float(connect_timeout_s)
        self.probe_timeout_s = float(probe_timeout_s)
        self.poll_s = float(poll_s)
        self.resolver = resolver
        self.probe = probe
        self.gateway = gateway
        self.clock = clock
        self.monotonic = monotonic
        self.sleep = sleep
        self.log = log
        # `_busy` serialises connect/clear (a connect blocks for seconds);
        # `_state_lock` guards the few fields status() reads meanwhile.
        self._busy = threading.Lock()
        self._state_lock = threading.Lock()
        self._active = False
        self._session_id = ""
        self._since: int | None = None
        self._tracks: list = []
        self._mode = ""  # "direct" or "tunnel" while active

    # -- what the rest of the process reads ------------------------------------

    @property
    def active(self) -> bool:
        with self._state_lock:
            return self._active

    @property
    def session_id(self) -> str:
        with self._state_lock:
            return self._session_id

    @property
    def since(self) -> int | None:
        with self._state_lock:
            return self._since

    @property
    def mode(self) -> str | None:
        """How the attached camera is reached: "direct" (dialled from the
        enclave) or "tunnel" (through the connector); None with none."""
        with self._state_lock:
            return (self._mode or None) if self._active else None

    @property
    def phone_stream_url(self) -> str:
        """The phone's own camera on the relay: the face view's stream
        while the external camera is the session's."""
        return f"{self.rtsp_base}/{self.relay.ingest_path}"

    def tunnel_status(self) -> dict:
        """{connected, sinceMs}: whether a camera at home could be reached
        right now, from the gateway (cached a couple of seconds there)."""
        if self.gateway is None:
            return dict(OFFLINE)
        return self.gateway.tunnel()

    def status(self) -> dict:
        """The phone's `GET /ingest/source` and the `source` object in the
        trainer's /ingest/status: which camera is the session's, whether
        the relay has its picture, how it is reached (`mode`) and whether
        the connector is there (`tunnel`). No URL, no host."""
        with self._state_lock:
            active, since, tracks = self._active, self._since, list(self._tracks)
            mode = self._mode or None
        tunnel = self.tunnel_status()
        if not active:
            return {"kind": "phone", "path": self.relay.ingest_path, "ready": None,
                    "tracks": [], "since": None, "mode": None, "tunnel": tunnel}
        live = self.relay.relay_status(self.path)
        ready = bool(live and live.get("ready"))
        return {"kind": "external", "path": self.path, "ready": ready,
                "tracks": list(live.get("tracks") or []) if live else tracks,
                "since": since, "mode": mode, "tunnel": tunnel}

    # -- connect / clear -------------------------------------------------------------

    def connect(self, url: str, session_id: str = "") -> dict:
        """Validate, probe, add the relay path and wait for its picture.
        The connected status on success; a SourceError otherwise (and no
        path left behind). A private host goes through the connector, a
        public one is dialled directly (the module docstring)."""
        link, host, port = parse_link(url)
        if is_private_host(host):
            mode, source = "tunnel", tunnel_source_url(link)
            fingerprint = self._tunnel_fingerprint(host, port)
        else:
            mode, source = "direct", link
            ip = resolve_public(host, port, self.own_ip, self.resolver)
            fingerprint = self.probe(host, port, ip, self.probe_timeout_s)
        started = self.monotonic()
        with self._busy:
            self._delete_path()  # a replaced link, or a failed earlier attempt
            self._add_path(source, fingerprint)
            try:
                tracks = self._await_ready()
            except SourceError:
                self._delete_path()
                self.log(f"external camera: failed to start for session "
                         f"{session_id or '?'}", flush=True)
                raise
            since = int(self.clock())
            with self._state_lock:
                self._active = True
                self._session_id = session_id
                self._since = since
                self._tracks = list(tracks)
                self._mode = mode
        self.log(f"external camera: connected ({', '.join(tracks) or 'no tracks'}, "
                 f"{mode}) for session {session_id or '?'} in "
                 f"{self.monotonic() - started:.1f}s", flush=True)
        return {"status": "connected", "kind": "external", "path": self.path,
                "tracks": list(tracks), "since": since, "mode": mode}

    def clear(self, reason: str = "") -> bool:
        """Drop the external camera: the relay path goes, the session's
        camera is the phone's again, and a camera that was reached through
        the connector takes the connector with it (the gateway's target is
        stale; best effort). True when there was one."""
        with self._busy:
            with self._state_lock:
                had, mode = self._active, self._mode
                self._active = False
                self._session_id = ""
                self._since = None
                self._tracks = []
                self._mode = ""
            self._delete_path()
        if had:
            self.log(f"external camera: removed ({reason or 'cleared'})", flush=True)
            if mode == "tunnel" and self.gateway is not None:
                self.gateway.clear_quietly(reason or "external camera cleared")
        return had

    # -- the connector's tunnel ------------------------------------------------------

    def _tunnel_fingerprint(self, host: str, port: int) -> str:
        """Point the gateway at the camera and probe it through the relay
        listener: the leaf seen there is the camera's own, so the pin is
        the one a direct dial would take. `tunnel-offline` when no
        connector is attached, or the gateway is not answering."""
        gateway = self.gateway
        if gateway is None:
            raise SourceError("tunnel-offline", "this slot has no tunnel for a "
                              "camera on a home network", 502)
        live = gateway.status()
        if live is None:
            raise SourceError("tunnel-offline", "the slot's tunnel gateway is not "
                              "answering", 502)
        if not live.get("connected"):
            raise SourceError("tunnel-offline", "the connector is not attached: run "
                              "masseuse-camlink on a computer in the camera's network "
                              "and pair it with this session", 502)
        try:
            gateway.target(host, port)
        except GatewayError as error:
            if error.status == 409:  # the connector left between the two calls
                message = "the connector went away before the camera could be reached"
            else:
                message = "the slot's tunnel gateway is not answering"
            raise SourceError("tunnel-offline", message, 502) from None
        self.log("external camera: tunnel target set", flush=True)
        try:
            return self.probe(RELAY_HOST, RELAY_PORT, RELAY_HOST, self.probe_timeout_s,
                              sni=False)
        except SourceError as error:
            raise SourceError(error.reason,
                              TUNNEL_PROBE_WORDING.get(error.reason, error.message),
                              error.status) from None

    # -- the relay's API -------------------------------------------------------------

    def _api(self, method: str, route: str, body: dict | None = None) -> int:
        payload = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.relay.api_base}{route}", data=payload, method=method,
            headers={"Content-Type": "application/json"} if payload else {})
        try:
            with self.relay.opener(request, timeout=self.relay.timeout_s) as response:
                return int(response.status)
        except urllib.error.HTTPError as error:
            return int(error.code)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            raise SourceError("relay", "the slot's relay is not answering", 503) from None

    def _add_path(self, url: str, fingerprint: str) -> None:
        # useAbsoluteTimestamp: the relay keeps the camera's own frame
        # times (its RTCP sender reports) and gives them to its readers
        # rather than restamping at arrival, so the producer can line this
        # view up with the phone's (producer.Decoder, sync.py). The relay
        # skips the packets before the camera's first report.
        code = self._api("POST", f"/v3/config/paths/add/{self.path}", {
            "source": url,
            "sourceFingerprint": fingerprint,
            "rtspTransport": "tcp",
            "sourceOnDemand": False,
            "useAbsoluteTimestamp": True,
        })
        if code != 200:
            raise SourceError("relay", f"the relay refused the path ({code})", 503)

    def _delete_path(self) -> None:
        try:
            self._api("DELETE", f"/v3/config/paths/delete/{self.path}")
        except SourceError:
            pass  # a relay that is down has nothing to delete

    def _await_ready(self) -> list:
        deadline = self.monotonic() + self.connect_timeout_s
        while True:
            live = self.relay.relay_status(self.path)
            if live is None:
                raise SourceError("relay", "the slot's relay is not answering", 503)
            if live.get("ready"):
                tracks = [str(t) for t in live.get("tracks") or []]
                if not any(is_video(t) for t in tracks):
                    raise SourceError("no-video", "the camera answered but sent no "
                                      "video track (" + (", ".join(tracks) or "nothing")
                                      + ")", 502)
                return tracks
            if self.monotonic() >= deadline:
                raise SourceError("timeout", "the camera did not start streaming "
                                  f"within {self.connect_timeout_s:.0f}s: check the "
                                  "link's path and credentials", 502)
            self.sleep(self.poll_s)
