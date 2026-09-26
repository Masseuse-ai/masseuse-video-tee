"""What changes when the slot is a Confidential Space VM (`--tee`).

On Cloud Run the producer trusted its front end: IAM decided who could
call `/warmup` or `/produce`, and the trainer proxied the phone's WHIP/WHEP
signalling with its own token. Inside the enclave there is no front end -
Caddy terminates TLS on the VM's public IP with a certificate minted in the
enclave and hands every request to this process on loopback - so the
producer does the gating itself, and does one more thing Cloud Run never
could: it proves to the phone that the peer it is about to send its camera
to is *this* enclave.

Three pieces, all in memory, nothing on disk:

  ControlAuth   the trainer's Google-signed OIDC ID token (audience: this
                slot's origin, email: the trainer's runtime service account)
                on the control routes. Same check the trainer makes on the
                readings the producer posts back, in the other direction.
  Lease         `POST /lease {sessionId, capabilityHash, expiresAt}` from
                the trainer parks the SHA-256 of a 32-byte capability the
                phone generated. The phone presents the capability itself as
                a bearer on /ingest/whip*, /overlay/whep* and /ingest/source
                (an external camera's link, external_source.py); the trainer
                and Cloudflare only ever saw the hash, so none of them can
                publish to, read from, or point a camera at this slot.
  EvidenceKey   an Ed25519 key generated at boot. Its public key's SHA-256
  Attestation   is a nonce in the Google Cloud Attestation token requested
                from the launcher (with the TLS leaf's SPKI hash as the
                second nonce), which binds the key to *this* enclave image
                on *this* hardware. Every WHIP/WHEP answer carries an
                `X-Masseuse-Evidence` signature over its DTLS fingerprint;
                the phone checks it before setRemoteDescription, and the
                browser's own DTLS handshake then pins the media to the
                attested enclave. The attestation token itself takes 10-15 s
                to mint, so it is fetched at boot and refreshed in the
                background rather than per request; `/attestation?nonce=`
                is the slow path for a verifier that wants freshness.

Nothing here logs an SDP body, a capability, or a token: the stdout of the
debug image reaches Cloud Logging.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

LAUNCHER_SOCKET = "/run/container_launcher/teeserver.sock"
CLAIMS_TOKEN_FILE = "/run/container_launcher/attestation_verifier_claims_token"
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
TOKEN_REFRESH_S = 600.0
# Google Cloud Attestation accepts nonces of 10..74 bytes; base64url of a
# SHA-256 is 43 characters and every nonce this module mints is one.
NONCE_MIN, NONCE_MAX = 10, 74
CLIENT_NONCE = re.compile(r"^[A-Za-z0-9_.:-]{10,74}$")
CAPABILITY_HASH = re.compile(r"^[0-9a-f]{64}$")
CAPABILITY = re.compile(r"^[A-Za-z0-9_-]{43}$")  # base64url of 32 bytes, unpadded
SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
FINGERPRINT = re.compile(rb"^a=fingerprint:\s*([A-Za-z0-9-]+)\s+([0-9A-Fa-f:]+)\s*$", re.M)
# What the phone reads off a WHIP/WHEP answer, and may send with its offer.
EXPOSED_HEADERS = "Location, ETag, ID, Link, Accept-Patch, X-Masseuse-Evidence"
ALLOWED_REQUEST_HEADERS = ("Authorization, Content-Type, If-Match, "
                           "X-Masseuse-Client-Nonce")
LEASE_MAX_S = 6 * 3600.0
# IdleExit defaults (TEE_IDLE_EXIT_S, TEE_BOOT_IDLE_S): a minute after the
# session for a page refresh to reconnect, five for a boot nobody leases.
IDLE_EXIT_S = 60.0
BOOT_IDLE_S = 300.0


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded)


def sha256_b64url(data: bytes) -> str:
    return b64url(hashlib.sha256(data).digest())


def parse_fingerprint(sdp: bytes) -> str | None:
    """The DTLS fingerprint an SDP answer commits to, normalised as
    `<hash-func> <UPPER:HEX:...>` (RFC 8122). None if there is none; the
    caller treats that as an answer it cannot vouch for."""
    found = FINGERPRINT.search(sdp)
    if not found:
        return None
    return f"{found.group(1).decode().lower()} {found.group(2).decode().upper()}"


def bearer(headers) -> str:
    value = headers.get("Authorization") or ""
    scheme, _, token = value.strip().partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return ""
    return token.strip()


# -- configuration -----------------------------------------------------------------


@dataclass(frozen=True)
class TeeConfig:
    """The allow-listed `tee-env-*` a slot boots with (Dockerfile.tee)."""

    public_host: str
    trainer_invoker_service_accounts: tuple[str, ...]
    allowed_origins: tuple[str, ...] = ("https://masseuse.ai",)
    tls_cert_dir: str = "/run/tee/caddy/certificates"
    launcher_socket: str = LAUNCHER_SOCKET
    refresh_s: float = TOKEN_REFRESH_S
    # IdleExit: how long the slot stays up with no lease and no session
    # after it has served one, and how long a boot waits for its first.
    idle_exit_s: float = IDLE_EXIT_S
    boot_idle_s: float = BOOT_IDLE_S

    @classmethod
    def from_env(cls, env=os.environ) -> "TeeConfig":
        host = (env.get("TEE_PUBLIC_HOST") or "").strip().lower()
        if not host:
            raise SystemExit("--tee needs TEE_PUBLIC_HOST")
        invokers = tuple(
            s.strip().lower()
            for s in (env.get("TRAINER_INVOKER_SERVICE_ACCOUNT") or "").split(",")
            if s.strip())
        if not invokers:
            raise SystemExit("--tee needs TRAINER_INVOKER_SERVICE_ACCOUNT")
        origins = tuple(
            o.strip().rstrip("/")
            for o in (env.get("TEE_ALLOWED_ORIGINS") or "https://masseuse.ai").split(",")
            if o.strip())
        return cls(public_host=host,
                   trainer_invoker_service_accounts=invokers,
                   allowed_origins=origins,
                   tls_cert_dir=env.get("TEE_TLS_CERT_DIR", cls.tls_cert_dir),
                   launcher_socket=env.get("TEE_LAUNCHER_SOCKET", LAUNCHER_SOCKET),
                   refresh_s=float(env.get("TEE_TOKEN_REFRESH_S", TOKEN_REFRESH_S)),
                   idle_exit_s=float(env.get("TEE_IDLE_EXIT_S") or IDLE_EXIT_S),
                   boot_idle_s=float(env.get("TEE_BOOT_IDLE_S") or BOOT_IDLE_S))

    @property
    def origin(self) -> str:
        return f"https://{self.public_host}"


# -- the evidence key --------------------------------------------------------------


class EvidenceKey:
    """An Ed25519 key that lives exactly as long as this process.

    Its public key is what the attestation token vouches for (through the
    eat_nonce) and what the phone verifies WHIP/WHEP evidence against.
    """

    def __init__(self, private_key=None):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519

        self._key = private_key or ed25519.Ed25519PrivateKey.generate()
        self.public_bytes = self._key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_b64url = b64url(self.public_bytes)
        # What goes into the attestation token: the hash, not the key, so
        # the token stays small and the binding is a plain SHA-256 check.
        self.nonce = sha256_b64url(self.public_bytes)

    def sign(self, payload: dict) -> str:
        """`v1.<b64url json>.<b64url signature>`: the wire form of a piece
        of evidence, verified by the phone with the public key above."""
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        signature = self._key.sign(body)
        return f"v1.{b64url(body)}.{b64url(signature)}"

    def evidence(self, *, role: str, fingerprint: str, client_nonce: str,
                 session_secret: str, session_id: str, iat: int | None = None,
                 ) -> str:
        return self.sign({
            "v": 1,
            "role": role,
            "fingerprint": fingerprint,
            "clientNonce": client_nonce,
            "sessionSecret": session_secret,
            "sessionId": session_id,
            "iat": int(iat if iat is not None else time.time()),
        })

    def describe(self) -> dict:
        return {"alg": "Ed25519", "publicKey": self.public_b64url,
                "nonce": self.nonce}


def verify_evidence(public_key_raw: bytes, evidence: str) -> dict:
    """The phone's side of `EvidenceKey.sign`, for tests and the verifier.
    Raises on a bad signature or shape; returns the payload."""
    from cryptography.hazmat.primitives.asymmetric import ed25519

    version, _, rest = evidence.partition(".")
    if version != "v1":
        raise ValueError("unknown evidence version")
    body_b64, _, signature_b64 = rest.partition(".")
    body = b64url_decode(body_b64)
    ed25519.Ed25519PublicKey.from_public_bytes(public_key_raw).verify(
        b64url_decode(signature_b64), body)
    return json.loads(body)


# -- the TLS leaf ------------------------------------------------------------------


def tls_spki_nonce(cert_dir: str | Path, host: str) -> str | None:
    """SHA-256 of the SubjectPublicKeyInfo of the newest certificate Caddy
    has stored for `host`, base64url; None until the first issuance lands.

    Caddy keeps `<host>/<host>.crt` under `certificates/<issuer>/`; the
    key never leaves the tmpfs, and the SPKI hash is what a verifier
    compares with the live connection's leaf (`openssl s_client`).
    """
    root = Path(cert_dir)
    if not root.is_dir():
        return None
    candidates = sorted(root.rglob(f"{host}.crt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import serialization

            pem = candidate.read_bytes()
            certificate = x509.load_pem_x509_certificate(pem)
            spki = certificate.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo)
            return sha256_b64url(spki)
        except Exception as error:  # noqa: BLE001 - a half-written file, say so
            print(f"tee: unreadable certificate {candidate.name}: {error!r}",
                  flush=True)
    return None


# -- the launcher ------------------------------------------------------------------


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


class LauncherClient:
    """`POST /v1/token` on the Confidential Space launcher socket: a Google
    Cloud Attestation OIDC token for `audience` carrying `nonces`."""

    def __init__(self, socket_path: str = LAUNCHER_SOCKET, timeout_s: float = 60.0):
        self.socket_path = socket_path
        self.timeout_s = timeout_s

    def token(self, audience: str, nonces: list[str]) -> str:
        for nonce in nonces:
            if not NONCE_MIN <= len(nonce.encode()) <= NONCE_MAX:
                raise ValueError("nonce length out of range")
        connection = _UnixHTTPConnection(self.socket_path, self.timeout_s)
        body = json.dumps({"audience": audience, "token_type": "OIDC",
                           "nonces": nonces})
        connection.request("POST", "/v1/token", body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        connection.close()
        if response.status != 200:
            raise RuntimeError(
                f"launcher /v1/token {response.status}: {data[:200]!r}")
        token = data.decode().strip()
        if token.count(".") != 2:
            raise RuntimeError("launcher returned something that is not a JWT")
        return token


def decode_claims(token: str) -> dict:
    """The payload of a JWT, unverified: for logging what the enclave was
    told about itself and for tests. Never a substitute for verification."""
    return json.loads(b64url_decode(token.split(".")[1]))


class Attestation:
    """The cached boot token and its refresh loop.

    Nonces: [sha256(evidence public key), sha256(TLS leaf SPKI)] - the
    second joins once Caddy has a certificate, which is also what triggers
    an early refresh. `live(nonce)` mints a fresh token with a caller's
    nonce appended, for a verifier that wants freshness (10-15 s).
    """

    def __init__(self, config: TeeConfig, evidence: EvidenceKey,
                 launcher: LauncherClient | None = None,
                 clock=time.time, sleep=time.sleep, log=print):
        self.config = config
        self.evidence = evidence
        self.launcher = launcher or LauncherClient(config.launcher_socket)
        self.clock = clock
        self.sleep = sleep
        self.log = log
        self._lock = threading.Lock()
        self._current: dict | None = None
        self._spki_nonce: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- the cache ---------------------------------------------------------------

    def nonces(self, extra: str | None = None) -> list[str]:
        out = [self.evidence.nonce]
        if self._spki_nonce:
            out.append(self._spki_nonce)
        if extra:
            out.append(extra)
        return out

    def refresh(self) -> dict:
        self._spki_nonce = tls_spki_nonce(self.config.tls_cert_dir,
                                          self.config.public_host)
        nonces = self.nonces()
        started = time.monotonic()
        token = self.launcher.token(self.config.origin, nonces)
        claims = decode_claims(token)
        current = {
            "token": token,
            "nonces": nonces,
            "issuedAt": int(self.clock()),
            "expiresAt": int(claims.get("exp") or 0),
            "tlsSpkiNonce": self._spki_nonce,
            "evidenceKey": self.evidence.describe(),
        }
        with self._lock:
            self._current = current
        digest = ((claims.get("submods") or {}).get("container") or {}).get(
            "image_digest")
        self.log(f"tee: attestation token refreshed in "
                 f"{time.monotonic() - started:.1f}s "
                 f"(image {digest}, hwmodel {claims.get('hwmodel')}, "
                 f"dbgstat {claims.get('dbgstat')}, "
                 f"tlsNonce={'yes' if self._spki_nonce else 'pending'})",
                 flush=True)
        return current

    def current(self) -> dict | None:
        with self._lock:
            return dict(self._current) if self._current else None

    def live(self, client_nonce: str) -> dict:
        if not CLIENT_NONCE.match(client_nonce):
            raise ValueError("nonce must be 10-74 characters of [A-Za-z0-9_.:-]")
        nonces = self.nonces(client_nonce)
        token = self.launcher.token(self.config.origin, nonces)
        return {"token": token, "nonces": nonces, "issuedAt": int(self.clock()),
                "tlsSpkiNonce": self._spki_nonce,
                "evidenceKey": self.evidence.describe()}

    # -- the loop ----------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="tee-attestation")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        backoff = 5.0
        while not self._stop.is_set():
            try:
                self.refresh()
                backoff = 5.0
            except Exception as error:  # noqa: BLE001 - retried, said aloud
                self.log(f"tee: attestation refresh failed: {error!r}", flush=True)
                self.sleep(backoff)
                backoff = min(backoff * 2, 120.0)
                continue
            # Wake early when the certificate (and so the SPKI nonce)
            # changes; otherwise on the refresh cadence.
            waited = 0.0
            while waited < self.config.refresh_s and not self._stop.is_set():
                self.sleep(5.0)
                waited += 5.0
                latest = tls_spki_nonce(self.config.tls_cert_dir,
                                        self.config.public_host)
                if latest != self._spki_nonce:
                    break


# -- the lease ---------------------------------------------------------------------


@dataclass
class Lease:
    """The trainer's grant of this slot to one phone: the SHA-256 (hex) of
    the phone's capability, which the phone alone holds in the clear."""

    session_id: str = ""
    capability_hash: str = ""
    expires_at: float = 0.0
    # True once any lease has been granted, and it stays true through
    # clear() and expiry: the slot has served a phone, so IdleExit uses the
    # short idle clock from here on, however the ticks fall.
    ever_granted: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    clock: object = field(default=time.time, repr=False)

    def grant(self, body: dict) -> tuple[int, dict]:
        """Decide a `POST /lease`: (status, body)."""
        session_id = str(body.get("sessionId") or "")
        capability_hash = str(body.get("capabilityHash") or "").lower()
        if not SESSION_ID.match(session_id):
            return 400, {"error": "sessionId must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"}
        if not CAPABILITY_HASH.match(capability_hash):
            return 400, {"error": "capabilityHash must be 64 hex characters (sha256)"}
        now = float(self.clock())
        try:
            expires_at = float(body.get("expiresAt") or (now + LEASE_MAX_S))
        except (TypeError, ValueError):
            return 400, {"error": "expiresAt must be a unix time in seconds"}
        if expires_at > 1e12:  # milliseconds; be forgiving about the unit
            expires_at /= 1000.0
        expires_at = min(expires_at, now + LEASE_MAX_S)
        if expires_at <= now:
            return 400, {"error": "expiresAt is in the past"}
        with self.lock:
            self.session_id = session_id
            self.capability_hash = capability_hash
            self.expires_at = expires_at
            self.ever_granted = True
        return 200, {"status": "leased", "sessionId": session_id,
                     "expiresAt": int(expires_at)}

    def clear(self) -> None:
        with self.lock:
            self.session_id = ""
            self.capability_hash = ""
            self.expires_at = 0.0

    def active(self) -> bool:
        with self.lock:
            return bool(self.capability_hash) and float(self.clock()) < self.expires_at

    def check(self, capability: str) -> tuple[bool, str]:
        """Does `capability` (the phone's bearer) open this lease?"""
        if not capability or not CAPABILITY.match(capability):
            return False, "missing or malformed capability"
        with self.lock:
            expected = self.capability_hash
            expires_at = self.expires_at
            session_id = self.session_id
        if not expected:
            return False, "no lease"
        if float(self.clock()) >= expires_at:
            return False, "lease expired"
        presented = hashlib.sha256(b64url_decode(capability)).hexdigest()
        if not hmac.compare_digest(presented, expected):
            return False, "capability does not match the lease"
        return True, session_id

    def snapshot(self) -> dict:
        with self.lock:
            return {"sessionId": self.session_id or None,
                    "active": bool(self.capability_hash)
                    and float(self.clock()) < self.expires_at,
                    "expiresAt": int(self.expires_at) if self.expires_at else None}


# -- the slot's lifetime -----------------------------------------------------------


class IdleExit:
    """Exit 0 when nobody wants the slot, so the VM stops billing.

    A Confidential Space slot is an a3-highgpu-1g the trainer starts for one
    session (masseuse-trainer/server/slot-pool.js); nothing but this process
    ends it. The trainer's /teardown after the session drains and exits
    (producer.Teardown), but the trainer may be gone - restarted, or never
    told about a boot a phone abandoned before its camera came up - so the
    slot decides for itself as well. The rule: with no lease held and
    nothing working (a boot, a /produce session), exit after `idle_s` once
    a lease has ever been granted, or after `boot_idle_s` when none ever
    was. A /warmup counts as interest (`touch`): a trainer that is polling
    has a phone waiting, and it leases within seconds of `ready`; one that
    never leases stops polling when its session is reaped, and the clock
    runs out from there.

    The exit ends the container, and with it Caddy and MediaMTX, so the
    slot stops answering: that is the signal the trainer's reaper acts on
    (instances.stop), and on the production image the launcher powers the
    VM off two minutes later on its own (tee-restart-policy=Never,
    masseuse-video-tee/terraform/vm.tf; the debug image holds the VM for
    the reaper). The trainer's next instances.start is a new boot, a new
    key, a new certificate.
    """

    def __init__(self, teardown, lease: Lease, *, idle_s: float = IDLE_EXIT_S,
                 boot_idle_s: float = BOOT_IDLE_S, clock=time.monotonic,
                 exit_impl=os._exit, sleep=time.sleep, log=print, tick_s: float = 1.0):
        self.teardown = teardown
        self.lease = lease
        self.idle_s = float(idle_s)
        self.boot_idle_s = float(boot_idle_s)
        self.clock = clock
        self.exit_impl = exit_impl
        self.sleep = sleep
        self.log = log
        self.tick_s = tick_s
        self._lock = threading.Lock()
        self._started = clock()
        self._idle_since: float | None = self._started
        self._last_touch = self._started
        self._stop = threading.Event()

    def touch(self) -> None:
        """A control request that shows interest (/warmup): the idle clock
        restarts from now."""
        with self._lock:
            self._last_touch = self.clock()

    def _busy(self) -> bool:
        return self.lease.active() or self.teardown.is_working()

    def due(self) -> str | None:
        """One tick: the reason to exit now, or None. Also advances the
        idle bookkeeping, so call it on a cadence."""
        now = self.clock()
        leased = self.lease.active()
        ever = self.lease.ever_granted
        with self._lock:
            if leased or self.teardown.is_working():
                self._idle_since = None
                return None
            if self._idle_since is None:
                self._idle_since = now
            since = max(self._idle_since, self._last_touch)
            limit = self.idle_s if ever else self.boot_idle_s
            idle_for = now - since
            if idle_for < limit:
                return None
            if ever:
                return f"no lease and no session for {idle_for:.0f}s"
            return f"no lease {idle_for:.0f}s after boot"

    def snapshot(self) -> dict:
        now = self.clock()
        ever = self.lease.ever_granted
        with self._lock:
            limit = self.idle_s if ever else self.boot_idle_s
            idle_since = self._idle_since
            since = max(idle_since, self._last_touch) if idle_since is not None else None
        idle_for = (now - since) if since is not None else 0.0
        return {"everLeased": ever,
                "idleForS": round(idle_for, 1),
                "exitInS": round(max(0.0, limit - idle_for), 1) if since is not None else None,
                "idleExitS": self.idle_s, "bootIdleS": self.boot_idle_s}

    def start(self) -> None:
        threading.Thread(target=self._run, name="tee-idle-exit", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            reason = self.due()
            if reason is not None:
                self.log(f"tee: idle exit: {reason}; exiting so the VM stops", flush=True)
                self.sleep(0.5)
                self.exit_impl(0)
                return
            self.sleep(self.tick_s)


# -- the trainer's token -----------------------------------------------------------


def google_id_token_verifier():
    """The default verifier: google-auth against Google's OAuth2 certs,
    checking the signature, expiry and audience. Issuer and email are
    checked by ControlAuth."""
    import google.auth.transport.requests
    import google.oauth2.id_token
    import requests

    session = requests.Session()
    request = google.auth.transport.requests.Request(session=session)

    def verify(token: str, audience: str) -> dict:
        return google.oauth2.id_token.verify_token(token, request, audience=audience)

    return verify


class ControlAuth:
    """The trainer's OIDC ID token on the control routes. In place of Cloud
    Run IAM: same token the trainer already mints per origin
    (masseuse-trainer/server/id-token.js), same checks the trainer applies
    to the producer's readings."""

    def __init__(self, config: TeeConfig, verifier=None, log=print):
        self.config = config
        self._verifier = verifier
        self.log = log

    def check(self, headers) -> tuple[bool, str]:
        token = bearer(headers)
        if not token:
            return False, "missing bearer token"
        if self._verifier is None:
            self._verifier = google_id_token_verifier()
        try:
            claims = self._verifier(token, self.config.origin)
        except Exception as error:  # noqa: BLE001 - every failure is a 401
            return False, f"token rejected: {type(error).__name__}"
        if claims.get("iss") not in GOOGLE_ISSUERS:
            return False, "unexpected issuer"
        email = str(claims.get("email") or "").lower()
        if not claims.get("email_verified", False) or not email:
            return False, "token carries no verified email"
        if email not in self.config.trainer_invoker_service_accounts:
            return False, "caller is not the trainer"
        return True, email


# -- CORS --------------------------------------------------------------------------


def cors_headers(config: TeeConfig, request_origin: str | None,
                 ) -> list[tuple[str, str]]:
    """The phone talks to the slot cross-origin from masseuse.ai, on a page
    served with COEP require-corp: allow exactly the configured origins,
    expose the WHIP/WHEP headers plus the evidence, and mark every response
    cross-origin-readable."""
    out = [("Cross-Origin-Resource-Policy", "cross-origin"),
           ("Vary", "Origin")]
    origin = (request_origin or "").rstrip("/")
    if origin and origin in config.allowed_origins:
        out.extend([
            ("Access-Control-Allow-Origin", origin),
            # PUT is /ingest/source, the external camera's link.
            ("Access-Control-Allow-Methods", "OPTIONS, GET, PUT, POST, PATCH, DELETE"),
            ("Access-Control-Allow-Headers", ALLOWED_REQUEST_HEADERS),
            ("Access-Control-Expose-Headers", EXPOSED_HEADERS),
            ("Access-Control-Max-Age", "600"),
        ])
    return out


# -- everything the handler needs, in one object ------------------------------------


class TeeMode:
    """The producer's TEE-mode collaborators, built once per process."""

    def __init__(self, config: TeeConfig, *, evidence: EvidenceKey | None = None,
                 attestation: Attestation | None = None,
                 control_auth: ControlAuth | None = None,
                 lease: Lease | None = None, launcher: LauncherClient | None = None,
                 clock=time.time, log=print):
        self.config = config
        self.evidence = evidence or EvidenceKey()
        self.attestation = attestation or Attestation(
            config, self.evidence, launcher=launcher, clock=clock, log=log)
        self.control_auth = control_auth or ControlAuth(config, log=log)
        self.lease = lease or Lease(clock=clock)
        self.clock = clock
        self.log = log

    @classmethod
    def from_env(cls, env=os.environ) -> "TeeMode":
        return cls(TeeConfig.from_env(env))

    @property
    def origin(self) -> str:
        return self.config.origin

    def start(self) -> None:
        self.attestation.start()

    def attestation_body(self, client_nonce: str | None) -> tuple[int, dict]:
        """`GET /attestation[?nonce=]`: the cached boot token, or a live one
        with the caller's nonce appended."""
        if client_nonce:
            try:
                return 200, self.attestation.live(client_nonce)
            except ValueError as error:
                return 400, {"error": str(error)}
            except Exception as error:  # noqa: BLE001 - the launcher is down
                self.log(f"tee: live attestation failed: {error!r}", flush=True)
                return 503, {"error": "attestation unavailable"}
        current = self.attestation.current()
        if current is None:
            return 503, {"error": "attestation not yet available",
                         "evidenceKey": self.evidence.describe()}
        return 200, current

    def evidence_for(self, *, role: str, answer: bytes, client_nonce: str,
                     session_secret: str, session_id: str) -> str | None:
        """The X-Masseuse-Evidence for a 201 answer, or None when the answer
        carries no fingerprint (the phone then refuses it)."""
        fingerprint = parse_fingerprint(answer)
        if fingerprint is None:
            return None
        return self.evidence.evidence(
            role=role, fingerprint=fingerprint, client_nonce=client_nonce,
            session_secret=session_secret, session_id=session_id,
            iat=int(self.clock()))

    def snapshot(self) -> dict:
        current = self.attestation.current()
        return {
            "origin": self.origin,
            "lease": self.lease.snapshot(),
            "attestation": ({"issuedAt": current["issuedAt"],
                             "expiresAt": current.get("expiresAt"),
                             "tlsNonce": bool(current.get("tlsSpkiNonce"))}
                            if current else None),
            "evidenceKeyNonce": self.evidence.nonce,
        }
