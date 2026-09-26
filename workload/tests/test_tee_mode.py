"""The Confidential Space slot's gates, against the same stub relay the
proxy tests use.

The trainer's token opens the control routes and nothing else does; the
phone's capability - the preimage of the hash the trainer leased - opens
WHIP/WHEP and nothing else does; every 201 answer carries an Ed25519
signature over its DTLS fingerprint that verifies with the key the
attestation token's nonce names; the preflight is answered here with the
page's origin only; a second publisher is refused; /stop and /teardown
clear the lease; and no capture may be configured in --tee mode.

The home connector's gateway (camlink_gateway.py) is driven from here too:
/tunnel/expect forwards the trainer's expectation for the leased session
only, /tunnel/clear drops it, a gateway that is down is a 502, and the
connector goes with the session (a teardown, a lease for another session,
the phone taking its camera back) but survives a /stop.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

import producer  # noqa: E402
import tee_mode  # noqa: E402
from camlink_gateway import CamlinkGateway  # noqa: E402
from egress import Egress  # noqa: E402
from external_source import ExternalSource  # noqa: E402
from hud_card import HudCard  # noqa: E402
from relay_proxy import RelayProxy  # noqa: E402
from share import Share  # noqa: E402
from telemetry import Telemetry  # noqa: E402
from test_camlink_gateway import CONNECTOR_KEY, TICKET_HASH, FakeGateway  # noqa: E402
from test_external_source import FINGERPRINT, GLOBAL, resolver_for  # noqa: E402
from test_relay_proxy import WHIP_SECRET, FakeOverlay, StubRelay  # noqa: E402

pytest.importorskip("cryptography")

TRAINER_SA = "trainer@example-project.iam.gserviceaccount.com"
HOST = "slot-0.tee.masseuse.ai"
ORIGIN = "https://masseuse.ai"
OFFER = (b"v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n"
         b"a=fingerprint:sha-256 AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:"
         b"AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99\r\n")


def capability() -> tuple[str, str]:
    raw = os.urandom(32)
    cap = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    return cap, hashlib.sha256(raw).hexdigest()


class FakeLauncher:
    """The launcher socket's /v1/token, minus the 1.2 s: an unsigned JWT
    carrying the requested audience and nonces."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def token(self, audience: str, nonces: list[str]) -> str:
        self.calls.append((audience, list(nonces)))
        header = tee_mode.b64url(json.dumps({"alg": "RS256", "kid": "fake"}).encode())
        now = int(time.time())
        payload = tee_mode.b64url(json.dumps({
            "aud": audience, "iss": "https://confidentialcomputing.googleapis.com",
            "iat": now, "exp": now + 3600, "eat_nonce": nonces,
            "hwmodel": "GCP_INTEL_TDX", "dbgstat": "enabled",
            "submods": {"container": {"image_digest": "sha256:feed"},
                        "nvidia_gpu": {"cc_mode": "ON"}},
        }).encode())
        return f"{header}.{payload}.{tee_mode.b64url(b'sig')}"


def trainer_verifier(token: str, audience: str) -> dict:
    """Stands in for google-auth: the token *is* its claims, as JSON."""
    claims = json.loads(token)
    if claims.get("aud") != audience:
        raise ValueError("wrong audience")
    if claims.get("exp", 0) < time.time():
        raise ValueError("expired")
    return claims


def trainer_token(email: str = TRAINER_SA, aud: str = f"https://{HOST}",
                  **extra) -> str:
    claims = {"iss": "https://accounts.google.com", "aud": aud, "email": email,
              "email_verified": True, "exp": time.time() + 300}
    claims.update(extra)
    return json.dumps(claims)


def make_tee(**overrides) -> tee_mode.TeeMode:
    config = tee_mode.TeeConfig(
        public_host=HOST, trainer_invoker_service_accounts=(TRAINER_SA,),
        allowed_origins=(ORIGIN,), tls_cert_dir="/nonexistent")
    launcher = FakeLauncher()
    evidence = tee_mode.EvidenceKey()
    attestation = tee_mode.Attestation(config, evidence, launcher=launcher,
                                       log=lambda *a, **k: None)
    tee = tee_mode.TeeMode(
        config, evidence=evidence, attestation=attestation,
        control_auth=tee_mode.ControlAuth(config, verifier=trainer_verifier),
        log=lambda *a, **k: None, **overrides)
    tee.launcher = launcher  # type: ignore[attr-defined]
    return tee


def server_args(**overrides) -> argparse.Namespace:
    base = dict(stream="", pose="sideload", pose_fps=9.0, device="cpu",
                track="", analysis_socket="", sink_dir="/tmp/x", run="",
                capture_bucket="", post_url="", post_interval_s=1.0,
                # A /teardown in a test must never reach its os._exit: the
                # drain outlasts the test, and the watcher is a daemon.
                duration=0.0, serve=True, tee=True, port=0, teardown_drain_s=600.0,
                overlay_publish="", overlay_record=False,
                overlay_relay_webrtc="", overlay_relay_api="")
    base.update(overrides)
    return argparse.Namespace(**base)


def request(port: int, method: str, path: str, body: bytes = b"",
            headers: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    payload = response.read()
    conn.close()
    return response.status, dict(response.getheaders()), payload


class FakeEgressProc:
    """The live stream's ffmpeg, scripted: it runs until terminated and
    never touches the network. `communicate` blocks on its end."""

    def __init__(self, argv):
        self.argv = argv
        self.returncode = None
        self.stdin = self
        self._done = threading.Event()

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        self._done.wait(timeout)
        return b"", b""

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self._done.set()

    def kill(self):
        self.returncode = -9
        self._done.set()

    def write(self, data):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class FakeBridge:
    """The share's TLS bridge without a socket: the URL alone."""

    def __init__(self, pin, log=print):
        self.pin = pin
        self.stopped = False

    def start(self):
        return 5555

    @property
    def url(self):
        return "rtsp://127.0.0.1:5555/phone"

    def stop(self):
        self.stopped = True


class Slot:
    """A TEE-mode producer server over a stub relay and a fake connector
    gateway, for one test. Its external camera resolves every name to a
    public address and finds a fixed certificate fingerprint without a
    network; `gateway_down` points it at a gateway that is not there."""

    def __init__(self, gateway_down: bool = False):
        self.relay = StubRelay()
        self.gateway = FakeGateway()
        self.tee = make_tee()
        self.probes: list[tuple] = []

        def probe(host, port, ip, timeout_s, **options):
            self.probes.append((host, port, ip) + ((options,) if options else ()))
            return FINGERPRINT

        self.external = ExternalSource(
            RelayProxy(self.relay.base, self.relay.base), own_ip="34.1.2.3",
            resolver=resolver_for(GLOBAL), probe=probe, sleep=lambda s: None,
            connect_timeout_s=0.5, log=lambda *a, **k: None,
            gateway=CamlinkGateway("http://127.0.0.1:1" if gateway_down else self.gateway.base,
                                   timeout_s=1.0, status_cache_s=0.0,
                                   log=lambda *a, **k: None))
        # The live stream over a scripted ffmpeg: every spawn is recorded,
        # none runs; the destination resolves to a public address.
        self.egress_procs: list[FakeEgressProc] = []

        def popen(argv, **kwargs):
            proc = FakeEgressProc(argv)
            self.egress_procs.append(proc)
            return proc

        self.egress = Egress(
            "rtsp://127.0.0.1:8554/overlay", "rtsp://127.0.0.1:8554/cam", own_ip="34.1.2.3",
            hud_card=HudCard(), resolver=resolver_for(GLOBAL), popen=popen,
            clock=lambda: 1_700_000_000.0, log=lambda *a, **k: None)
        # The phone's picture to the connector: the same scripted ffmpeg,
        # the connector's leaf found without a network, the bridge a stub.
        self.share_procs: list[FakeEgressProc] = []

        def share_popen(argv, **kwargs):
            proc = FakeEgressProc(argv)
            self.share_procs.append(proc)
            return proc

        self.share = Share(
            "rtsp://127.0.0.1:8554/cam", gateway=self.external.gateway,
            probe=lambda: FINGERPRINT.lower(), bridge_factory=FakeBridge, popen=share_popen,
            clock=lambda: 1_700_000_000.0, log=lambda *a, **k: None)
        # The connector's front-facing camera: a second external source on
        # the face path, which does not own the connector.
        self.face_source = ExternalSource(
            RelayProxy(self.relay.base, self.relay.base), own_ip="34.1.2.3", path="face-ext",
            resolver=resolver_for(GLOBAL), probe=probe, sleep=lambda s: None,
            connect_timeout_s=0.5, log=lambda *a, **k: None,
            gateway=self.external.gateway, owns_gateway=False)
        self.server = producer.build_server(
            server_args(overlay_publish="rtsp://127.0.0.1:8554/overlay",
                        overlay_renditions="hi,half,small,lean",
                        overlay_relay_webrtc=self.relay.base,
                        overlay_relay_api=self.relay.base),
            Telemetry(), tee=self.tee, external=self.external, egress=self.egress,
            share=self.share, face_source=self.face_source)
        # The process exits a slot may ask for (a `/teardown?mode=now`, the
        # idle exit) are recorded here, never taken: taken, os._exit(0)
        # ends the test runner half way with a clean exit code and no
        # summary, and every test after it silently never runs.
        self.exits: list[int] = []
        self.server.teardown.exit_impl = self.exits.append
        if self.server.idle_exit is not None:
            self.server.idle_exit.exit_impl = self.exits.append
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.current["session"] = None
        self.egress.clear("test over")
        self.share.clear("test over")
        self.server.shutdown()
        self.server.server_close()
        self.relay.stop()
        self.gateway.stop()

    def expect(self, session_id: str = "sess-1", **overrides) -> dict:
        body = {"connectorKey": CONNECTOR_KEY, "ticketHash": TICKET_HASH,
                "expiresAt": int(time.time()) + 600, "sessionId": session_id}
        body.update(overrides)
        return body

    def request(self, method, path, body=b"", headers=None):
        return request(self.port, method, path, body, headers)

    def as_trainer(self, method, path, body=b"", headers=None):
        merged = {"Authorization": f"Bearer {trainer_token()}"}
        merged.update(headers or {})
        return self.request(method, path, body, merged)

    def lease(self, session_id: str = "sess-1") -> str:
        cap, cap_hash = capability()
        code, _, body = self.as_trainer(
            "POST", "/lease", json.dumps({"sessionId": session_id,
                                          "capabilityHash": cap_hash}).encode(),
            {"Content-Type": "application/json"})
        assert code == 200 and json.loads(body)["status"] == "leased"
        return cap

    def as_phone(self, cap: str, method, path, body=b"", headers=None):
        merged = {"Authorization": f"Bearer {cap}", "Origin": ORIGIN}
        merged.update(headers or {})
        return self.request(method, path, body, merged)


# -- units ---------------------------------------------------------------------------


def test_fingerprint_parsing_normalises_case_and_refuses_answers_without_one():
    assert tee_mode.parse_fingerprint(OFFER) == (
        "sha-256 AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:"
        "AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99")
    assert tee_mode.parse_fingerprint(b"a=fingerprint:SHA-256 ab:cd\r\n") == "sha-256 AB:CD"
    assert tee_mode.parse_fingerprint(b"v=0\r\nc=IN IP4 0.0.0.0\r\n") is None


def test_evidence_verifies_with_the_key_the_nonce_names_and_not_another():
    key = tee_mode.EvidenceKey()
    assert key.nonce == tee_mode.sha256_b64url(key.public_bytes)
    assert len(key.nonce) == 43
    evidence = key.evidence(role="whip", fingerprint="sha-256 AB:CD",
                            client_nonce="phone-nonce-1234", session_secret="s",
                            session_id="sess", iat=1700000000)
    payload = tee_mode.verify_evidence(key.public_bytes, evidence)
    assert payload == {"v": 1, "role": "whip", "fingerprint": "sha-256 AB:CD",
                       "clientNonce": "phone-nonce-1234", "sessionSecret": "s",
                       "sessionId": "sess", "iat": 1700000000}
    other = tee_mode.EvidenceKey()
    with pytest.raises(Exception):
        tee_mode.verify_evidence(other.public_bytes, evidence)
    with pytest.raises(ValueError):
        tee_mode.verify_evidence(key.public_bytes, "v2." + evidence[3:])


def test_the_lease_opens_for_the_preimage_only_and_expires():
    clock = {"now": 1000.0}
    lease = tee_mode.Lease(clock=lambda: clock["now"])
    cap, cap_hash = capability()
    assert lease.check(cap) == (False, "no lease")
    code, body = lease.grant({"sessionId": "bad/slash", "capabilityHash": cap_hash})
    assert code == 400
    code, body = lease.grant({"sessionId": "s1", "capabilityHash": "nothex"})
    assert code == 400
    code, body = lease.grant({"sessionId": "s1", "capabilityHash": cap_hash,
                              "expiresAt": 1000.0 + 60})
    assert code == 200 and body["expiresAt"] == 1060
    assert lease.check(cap) == (True, "s1")
    other, _ = capability()
    assert lease.check(other) == (False, "capability does not match the lease")
    assert lease.check("short") == (False, "missing or malformed capability")
    assert lease.check("") == (False, "missing or malformed capability")
    clock["now"] = 1061.0
    assert lease.check(cap) == (False, "lease expired")
    assert lease.active() is False
    # Milliseconds are forgiven; the cap is enforced.
    code, body = lease.grant({"sessionId": "s2", "capabilityHash": cap_hash,
                              "expiresAt": (1061 + 10 * 3600) * 1000})
    assert code == 200 and body["expiresAt"] == 1061 + 6 * 3600
    lease.clear()
    assert lease.check(cap) == (False, "no lease")


def test_control_auth_wants_a_verified_trainer_email_for_this_origin():
    config = tee_mode.TeeConfig(public_host=HOST,
                                trainer_invoker_service_accounts=(TRAINER_SA,))
    auth = tee_mode.ControlAuth(config, verifier=trainer_verifier)
    assert auth.check({}) == (False, "missing bearer token")
    assert auth.check({"Authorization": "Basic abc"}) == (False, "missing bearer token")
    ok, who = auth.check({"Authorization": f"Bearer {trainer_token()}"})
    assert ok and who == TRAINER_SA
    ok, why = auth.check({"Authorization": f"Bearer {trainer_token(aud='https://other')}"})
    assert not ok and why.startswith("token rejected")
    ok, why = auth.check({"Authorization": f"Bearer {trainer_token(email='x@y.iam.gserviceaccount.com')}"})
    assert (ok, why) == (False, "caller is not the trainer")
    ok, why = auth.check({"Authorization": f"Bearer {trainer_token(email_verified=False)}"})
    assert (ok, why) == (False, "token carries no verified email")
    ok, why = auth.check({"Authorization": f"Bearer {trainer_token(iss='https://evil')}"})
    assert (ok, why) == (False, "unexpected issuer")


def test_attestation_refresh_binds_the_evidence_key_and_serves_a_live_nonce():
    tee = make_tee()
    current = tee.attestation.refresh()
    assert current["nonces"] == [tee.evidence.nonce]
    assert current["tlsSpkiNonce"] is None
    claims = tee_mode.decode_claims(current["token"])
    assert claims["aud"] == f"https://{HOST}"
    assert claims["eat_nonce"] == [tee.evidence.nonce]
    assert tee.attestation.current()["evidenceKey"] == tee.evidence.describe()
    live = tee.attestation.live("verifier-nonce-42")
    assert live["nonces"] == [tee.evidence.nonce, "verifier-nonce-42"]
    with pytest.raises(ValueError):
        tee.attestation.live("short")
    with pytest.raises(ValueError):
        tee.attestation.live("bad nonce with spaces!")


def test_tls_spki_nonce_reads_the_newest_leaf_caddy_stored(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from datetime import datetime, timedelta, timezone

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, HOST)])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(1)
            .not_valid_before(now).not_valid_after(now + timedelta(days=1))
            .sign(key, hashes.SHA256()))
    where = tmp_path / "acme-staging" / HOST
    where.mkdir(parents=True)
    (where / f"{HOST}.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    expected = tee_mode.sha256_b64url(key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo))
    assert tee_mode.tls_spki_nonce(tmp_path, HOST) == expected
    assert tee_mode.tls_spki_nonce(tmp_path, "other.example") is None
    assert tee_mode.tls_spki_nonce(tmp_path / "missing", HOST) is None


def test_cors_is_for_the_page_origin_only():
    config = tee_mode.TeeConfig(public_host=HOST,
                                trainer_invoker_service_accounts=(TRAINER_SA,),
                                allowed_origins=(ORIGIN,))
    allowed = dict(tee_mode.cors_headers(config, ORIGIN))
    assert allowed["Access-Control-Allow-Origin"] == ORIGIN
    assert "X-Masseuse-Evidence" in allowed["Access-Control-Expose-Headers"]
    assert "Authorization" in allowed["Access-Control-Allow-Headers"]
    assert allowed["Cross-Origin-Resource-Policy"] == "cross-origin"
    denied = dict(tee_mode.cors_headers(config, "https://evil.example"))
    assert "Access-Control-Allow-Origin" not in denied
    assert denied["Cross-Origin-Resource-Policy"] == "cross-origin"
    assert "Access-Control-Allow-Origin" not in dict(tee_mode.cors_headers(config, None))


def test_config_from_env_needs_the_host_and_the_trainer():
    env = {"TEE_PUBLIC_HOST": "Slot-0.TEE.masseuse.ai",
           "TRAINER_INVOKER_SERVICE_ACCOUNT": "A@x.iam, b@y.iam",
           "TEE_ALLOWED_ORIGINS": "https://masseuse.ai/, https://staging.masseuse.ai"}
    config = tee_mode.TeeConfig.from_env(env)
    assert config.origin == "https://slot-0.tee.masseuse.ai"
    assert config.trainer_invoker_service_accounts == ("a@x.iam", "b@y.iam")
    assert config.allowed_origins == ("https://masseuse.ai", "https://staging.masseuse.ai")
    with pytest.raises(SystemExit):
        tee_mode.TeeConfig.from_env({"TEE_PUBLIC_HOST": "h"})
    with pytest.raises(SystemExit):
        tee_mode.TeeConfig.from_env({"TRAINER_INVOKER_SERVICE_ACCOUNT": "a@b"})


# -- through the handler --------------------------------------------------------------


def test_control_routes_take_the_trainer_token_and_nothing_else():
    with Slot() as slot:
        assert slot.request("GET", "/healthz")[0] == 200
        for method, path in (("GET", "/statz"), ("GET", "/warmup"),
                             ("GET", "/ingest/status"), ("GET", "/overlay/status"),
                             ("POST", "/stop"), ("POST", "/teardown"),
                             ("POST", "/lease"), ("GET", "/produce"),
                             ("POST", "/tunnel/expect"), ("POST", "/tunnel/clear")):
            code, headers, body = slot.request(method, path)
            assert code == 401, (method, path, code)
            assert headers["WWW-Authenticate"] == "Bearer"
            assert json.loads(body)["reason"] == "missing bearer token"
        assert slot.gateway.requests == []  # nothing reached the gateway
        code, _, body = slot.request(
            "GET", "/statz",
            headers={"Authorization": f"Bearer {trainer_token(email='intruder@x.iam.gserviceaccount.com')}"})
        assert code == 401 and json.loads(body)["reason"] == "caller is not the trainer"
        code, _, body = slot.as_trainer("GET", "/statz")
        assert code == 200
        snapshot = json.loads(body)
        assert snapshot["tee"]["origin"] == f"https://{HOST}"
        assert snapshot["tee"]["lease"] == {"sessionId": None, "active": False, "stopped": False,
                                            "expiresAt": None, "record": None}
        assert snapshot["counters"].get("teeControlRejects") == 11
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        assert code == 200 and json.loads(body)["ready"] is False
        code, _, body = slot.as_trainer("POST", "/stop")
        assert code == 404  # authenticated, nothing running


def test_attestation_and_evidence_key_are_public_and_cors_for_the_page():
    with Slot() as slot:
        code, headers, body = slot.request("GET", "/attestation",
                                           headers={"Origin": ORIGIN})
        assert code == 503 and json.loads(body)["error"].startswith("attestation not yet")
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        slot.tee.attestation.refresh()
        code, headers, body = slot.request("GET", "/attestation",
                                           headers={"Origin": ORIGIN})
        assert code == 200
        doc = json.loads(body)
        assert doc["nonces"] == [slot.tee.evidence.nonce]
        assert tee_mode.decode_claims(doc["token"])["eat_nonce"] == [slot.tee.evidence.nonce]
        assert doc["evidenceKey"]["publicKey"] == slot.tee.evidence.public_b64url
        assert headers["Cache-Control"] == "no-store"
        assert headers["Cross-Origin-Resource-Policy"] == "cross-origin"
        code, _, body = slot.request("GET", "/attestation?nonce=phone-fresh-nonce-1",
                                     headers={"Origin": ORIGIN})
        assert code == 200
        assert json.loads(body)["nonces"] == [slot.tee.evidence.nonce, "phone-fresh-nonce-1"]
        assert slot.request("GET", "/attestation?nonce=x")[0] == 400
        code, headers, body = slot.request("GET", "/evidence-key",
                                           headers={"Origin": "https://evil.example"})
        assert code == 200 and json.loads(body) == slot.tee.evidence.describe()
        assert "Access-Control-Allow-Origin" not in headers
        code, headers, _ = slot.request("OPTIONS", "/attestation",
                                        headers={"Origin": ORIGIN,
                                                 "Access-Control-Request-Method": "GET"})
        assert code == 204 and headers["Access-Control-Allow-Origin"] == ORIGIN
        assert slot.request("POST", "/evidence-key")[0] == 405


def test_lease_then_whip_with_the_capability_yields_signed_evidence():
    with Slot() as slot:
        # Without a lease the phone is refused, whatever it presents.
        cap0, _ = capability()
        code, headers, body = slot.request(
            "POST", "/ingest/whip", OFFER,
            {"Content-Type": "application/sdp", "Authorization": f"Bearer {cap0}",
             "Origin": ORIGIN})
        assert code == 401 and json.loads(body)["reason"] == "no lease"
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        assert slot.relay.requests == []  # the relay never saw it

        cap = slot.lease()
        # A wrong capability, a missing one, a malformed nonce.
        assert slot.request("POST", "/ingest/whip", OFFER,
                            {"Content-Type": "application/sdp",
                             "Authorization": f"Bearer {cap0}"})[0] == 401
        assert slot.request("POST", "/ingest/whip", OFFER,
                            {"Content-Type": "application/sdp"})[0] == 401
        assert slot.request("POST", "/ingest/whip", OFFER,
                            {"Content-Type": "application/sdp",
                             "Authorization": f"Bearer {cap}",
                             "X-Masseuse-Client-Nonce": "bad nonce!"})[0] == 400
        assert slot.relay.requests == []

        # The preflight is answered here, for the page's origin.
        code, headers, _ = slot.request(
            "OPTIONS", "/ingest/whip", b"",
            {"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
             "Access-Control-Request-Headers": "authorization, content-type"})
        assert code == 204
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        assert "Authorization" in headers["Access-Control-Allow-Headers"]
        assert slot.relay.requests == []

        # The real thing.
        code, headers, body = slot.request(
            "POST", "/ingest/whip", OFFER,
            {"Content-Type": "application/sdp", "Authorization": f"Bearer {cap}",
             "Origin": ORIGIN, "X-Masseuse-Client-Nonce": "phone-nonce-0001"})
        assert code == 201
        assert body.startswith(b"v=0\r\nwhip-answer-for:")
        assert headers["Location"] == f"https://{HOST}/ingest/whip/{WHIP_SECRET}"
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        assert "X-Masseuse-Evidence" in headers["Access-Control-Expose-Headers"]
        # The relay never saw the capability.
        forwarded = slot.relay.requests[-1]["headers"]
        assert "Authorization" not in forwarded
        assert "X-Masseuse-Client-Nonce" not in forwarded
        payload = tee_mode.verify_evidence(slot.tee.evidence.public_bytes,
                                           headers["X-Masseuse-Evidence"])
        assert payload["role"] == "whip"
        assert payload["fingerprint"] == tee_mode.parse_fingerprint(body)
        assert payload["clientNonce"] == "phone-nonce-0001"
        assert payload["sessionSecret"] == WHIP_SECRET
        assert payload["sessionId"] == "sess-1"
        assert abs(payload["iat"] - time.time()) < 5

        # One publisher per lease: the camera is up, so the holder's next
        # offer (a reload) kicks the lingering session and takes the leg.
        assert slot.relay.kicks == []
        code, _, body = slot.request(
            "POST", "/ingest/whip", OFFER,
            {"Content-Type": "application/sdp", "Authorization": f"Bearer {cap}"})
        assert code == 201
        assert slot.relay.kicks == ["cam-1"]

        # Trickle ICE and hang-up follow the absolute Location's path.
        path = headers["Location"][len(f"https://{HOST}"):]
        assert slot.request("PATCH", path, b"a=candidate",
                            {"Content-Type": "application/trickle-ice-sdpfrag",
                             "Authorization": f"Bearer {cap}"})[0] == 204
        assert slot.request("PATCH", path, b"a=candidate",
                            {"Content-Type": "application/trickle-ice-sdpfrag"})[0] == 401
        assert slot.request("DELETE", path, headers={"Authorization": f"Bearer {cap}"})[0] == 200

        # The overlay leg: same capability, gated on a session as before.
        code, _, body = slot.request(
            "POST", "/overlay/whep", OFFER,
            {"Content-Type": "application/sdp", "Authorization": f"Bearer {cap}"})
        assert code == 503
        slot.server.current["session"] = argparse.Namespace(
            overlay=FakeOverlay(), run_name="sess-1")
        code, headers, body = slot.request(
            "POST", "/overlay/whep", OFFER,
            {"Content-Type": "application/sdp", "Authorization": f"Bearer {cap}",
             "X-Masseuse-Client-Nonce": "phone-nonce-0002"})
        assert code == 201
        assert headers["Location"].startswith(f"https://{HOST}/overlay/whep/")
        payload = tee_mode.verify_evidence(slot.tee.evidence.public_bytes,
                                           headers["X-Masseuse-Evidence"])
        assert payload["role"] == "whep" and payload["clientNonce"] == "phone-nonce-0002"
        # The stub relay always reports one reader (r1) on the overlay: it
        # was kicked before this offer went through, one subscriber per lease.
        assert slot.relay.kicks == ["cam-1", "r1"]

        # A rendition of the view: the same capability and gate, the
        # relay's overlay-half path, a Location under /overlay-half/whep/,
        # evidence for the whep leg (the phone verifies it as it does the
        # view's own), and only that path's reader evicted: the reader of
        # the view the phone is moving away from lives on until it hangs
        # up, so the picture never gaps.
        code, headers, body = slot.request(
            "POST", "/overlay-half/whep", OFFER,
            {"Content-Type": "application/sdp", "Authorization": f"Bearer {cap}",
             "X-Masseuse-Client-Nonce": "phone-nonce-0003"})
        assert code == 201 and body.startswith(b"v=0\r\nanswer-for:")
        assert headers["Location"].startswith(f"https://{HOST}/overlay-half/whep/")
        assert slot.relay.requests[-1]["path"] == "/overlay-half/whep"
        payload = tee_mode.verify_evidence(slot.tee.evidence.public_bytes,
                                           headers["X-Masseuse-Evidence"])
        assert payload["role"] == "whep" and payload["clientNonce"] == "phone-nonce-0003"
        assert payload["fingerprint"] == tee_mode.parse_fingerprint(body)
        assert slot.relay.kicks == ["cam-1", "r1", "r1-half"]
        half = headers["Location"][len(f"https://{HOST}"):]
        assert slot.request("DELETE", half, headers={"Authorization": f"Bearer {cap}"})[0] == 200
        assert slot.relay.requests[-1]["path"].startswith("/overlay-half/whep/")
        # Without the capability a rendition is as closed as the view.
        assert slot.request("POST", "/overlay-lean/whep", OFFER,
                            {"Content-Type": "application/sdp"})[0] == 401
        # The status names every rendition's route.
        code, _, body = slot.as_trainer("GET", "/overlay/status")
        assert code == 200 and json.loads(body)["renditions"] == {
            "hi": "/overlay/whep", "half": "/overlay-half/whep",
            "small": "/overlay-small/whep", "lean": "/overlay-lean/whep"}

        # /stop clears the lease; the capability is dead.
        slot.server.current["session"] = None
        slot.as_trainer("POST", "/stop")  # 404: nothing running, lease untouched
        assert slot.tee.lease.active() is True
        code, _, _ = slot.as_trainer("POST", "/teardown?mode=drain")
        assert code == 200
        assert slot.tee.lease.active() is False
        assert slot.request("POST", "/ingest/whip", OFFER,
                            {"Content-Type": "application/sdp",
                             "Authorization": f"Bearer {cap}"})[0] == 401
        code, _, body = slot.as_trainer("GET", "/statz")
        counters = json.loads(body)["counters"]
        assert counters["teeEvidenceIssued"] == 4
        assert counters["teeLegEvicted"] == 3
        assert counters["teeCapabilityRejects"] >= 5


def test_lease_validation_and_a_stop_that_leaves_the_lease_standing():
    with Slot() as slot:
        code, _, body = slot.as_trainer("POST", "/lease", b"not json",
                                        {"Content-Type": "application/json"})
        assert code == 400
        code, _, body = slot.as_trainer(
            "POST", "/lease", json.dumps({"sessionId": "s", "capabilityHash": "zz"}).encode())
        assert code == 400 and "capabilityHash" in json.loads(body)["error"]
        assert slot.as_trainer("POST", "/lease", b"x" * 5000)[0] == 413
        cap = slot.lease()
        assert slot.tee.lease.active() and slot.tee.lease.busy()
        running = argparse.Namespace(stopping=threading.Event(), run_name="sess-1", overlay=None)
        slot.server.current["session"] = running
        # Another session's /stop is refused, and the run is left alone
        # (2026-09-17: a release's stop and teardown landed on the slot the
        # next session had just been leased).
        code, _, body = slot.as_trainer("POST", "/stop?session=sess-9")
        assert code == 409 and json.loads(body) == {"error": "lease mismatch"}
        assert running.stopping.is_set() is False
        assert slot.tee.lease.active() and slot.tee.lease.busy()
        code, _, body = slot.as_trainer("POST", "/teardown?session=sess-9")
        assert code == 409 and json.loads(body) == {"error": "lease mismatch"}
        assert slot.tee.lease.active()
        # The lease's own /stop ends the run and leaves the lease standing:
        # the phone's capability still opens it (its legs, the connector),
        # the record stays the session's, and nothing works under it until
        # the next /produce, so the idle clock runs.
        code, _, _ = slot.as_trainer("POST", "/stop?session=sess-1")
        assert code == 200
        assert running.stopping.is_set()
        assert slot.tee.lease.active() is True
        assert slot.tee.lease.busy() is False
        assert slot.tee.lease.check(cap) == (True, "sess-1")
        assert slot.tee.lease.snapshot()["stopped"] is True
        code, _, statz = slot.as_trainer("GET", "/statz")
        assert json.loads(statz)["tee"]["lease"]["stopped"] is True
        assert json.loads(statz)["counters"].get("leaseMismatch") == 2
        # A caller that names no session is taken at its word, as before.
        slot.server.current["session"] = argparse.Namespace(
            stopping=threading.Event(), run_name="sess-1", overlay=None)
        assert slot.as_trainer("POST", "/stop")[0] == 200
        # The lease's own /teardown clears it.
        slot.server.current["session"] = None
        assert slot.as_trainer("POST", "/teardown?session=sess-1&mode=drain")[0] == 200
        assert slot.tee.lease.active() is False
        assert slot.tee.lease.check(cap) == (False, "no lease")
        # With no lease held, anyone's teardown is nobody's to refuse.
        assert slot.as_trainer("POST", "/teardown?session=sess-9&mode=drain")[0] == 200


def test_the_lease_names_the_sessions_record_when_the_slot_has_a_bucket():
    """The trainer's lease may carry `record` (the prefix its half of the
    session record lives beside, producer/record.py): a slot with a
    capture bucket takes it and says where the parts will land; a slot
    without refuses the lease, so nothing is silently not kept; /stop
    clears it with the lease."""
    prefix = ("019966a0-0000-7000-8000-000000000001/estim_sessions/"
              "019966a0-0000-7000-8000-000000000002/enclave")
    json_type = {"Content-Type": "application/json"}
    with Slot() as slot:
        _, cap_hash = capability()
        body = {"sessionId": "sess-1", "capabilityHash": cap_hash,
                "record": {"prefix": prefix, "partSeconds": 30}}
        code, _, answer = slot.as_trainer("POST", "/lease", json.dumps(body).encode(), json_type)
        assert code == 400 and "capture bucket" in json.loads(answer)["error"]
        assert not slot.tee.lease.active()
        # The same slot with the attested bucket.
        slot.tee.lease.capture_bucket = "masseuse-ai-prod"
        code, _, answer = slot.as_trainer(
            "POST", "/lease", json.dumps({**body, "record": {"prefix": "runs/x"}}).encode(), json_type)
        assert code == 400 and "record.prefix" in json.loads(answer)["error"]
        code, _, answer = slot.as_trainer("POST", "/lease", json.dumps(body).encode(), json_type)
        assert code == 200
        assert json.loads(answer)["record"] == {"prefix": prefix, "partSeconds": 30,
                                                "bucket": "masseuse-ai-prod"}
        assert slot.tee.lease.record_for() == {"prefix": prefix, "partSeconds": 30,
                                               "bucket": "masseuse-ai-prod", "sessionId": "sess-1"}
        code, _, statz = slot.as_trainer("GET", "/statz")
        assert json.loads(statz)["tee"]["lease"]["record"]["prefix"] == prefix
        slot.server.current["session"] = argparse.Namespace(
            stopping=threading.Event(), run_name="sess-1", overlay=None)
        assert slot.as_trainer("POST", "/stop")[0] == 200
        # The lease stands through a /stop, its record with it: the next
        # /produce of the same session (a camera change) records there.
        assert slot.tee.lease.record_for() == {"prefix": prefix, "partSeconds": 30,
                                               "bucket": "masseuse-ai-prod", "sessionId": "sess-1"}
        slot.server.current["session"] = None
        assert slot.as_trainer("POST", "/teardown?mode=drain")[0] == 200
        assert slot.tee.lease.record_for() is None
        # A lease without a record is a session without one, as before.
        code, _, answer = slot.as_trainer(
            "POST", "/lease", json.dumps({k: v for k, v in body.items() if k != "record"}).encode(),
            json_type)
        assert code == 200 and "record" not in json.loads(answer)
        assert slot.tee.lease.record_for() is None


def test_the_leases_record_outlives_its_runs_and_closes_with_the_lease(tmp_path):
    """The record is the lease's (producer/record.py RecordKeeper): open
    between a session's production runs, listed in /statz, closed by the
    trainer's /stop once the run has let go, by /teardown, and by a lease
    for another session - while the same session's re-lease keeps it."""
    from test_record import BUCKET, PREFIX, FakeGcs  # noqa: PLC0415
    from record import Record  # noqa: PLC0415

    other = PREFIX.replace("019966a0-0000-7000-8000-000000000002",
                           "019966a0-0000-7000-8000-000000000003")
    json_type = {"Content-Type": "application/json"}
    gcs = FakeGcs()

    def opened(prefix, name):
        return slot.server.records.open(BUCKET, prefix, lambda: Record(
            tmp_path / name, BUCKET, prefix, name, client_factory=lambda: gcs))

    def lease(session_id, prefix):
        _, cap_hash = capability()
        body = {"sessionId": session_id, "capabilityHash": cap_hash,
                "record": {"prefix": prefix, "partSeconds": 30}}
        code, _, _ = slot.as_trainer("POST", "/lease", json.dumps(body).encode(), json_type)
        assert code == 200

    def settled(predicate, timeout_s=5.0):
        deadline = time.monotonic() + timeout_s
        while not predicate():
            assert time.monotonic() < deadline, "timed out"
            time.sleep(0.02)

    with Slot() as slot:
        slot.tee.lease.capture_bucket = BUCKET
        lease("sess-1", PREFIX)
        record = opened(PREFIX, "r1")
        record.append("posts", {"atS": 1.0})
        code, _, statz = slot.as_trainer("GET", "/statz")
        listed = json.loads(statz)["records"]
        assert [r["prefix"] for r in listed] == [PREFIX] and listed[0]["rows"] == 1
        # The same session leasing again (a trainer restart) keeps it.
        lease("sess-1", PREFIX)
        assert record.closed is False and slot.server.records.open_records() == [record]
        # A /stop that finds nothing running leaves it (the lease stays).
        slot.server.current["session"] = None
        assert slot.as_trainer("POST", "/stop")[0] == 404
        assert record.closed is False
        # A /stop that ends a run leaves it too: the record is the lease's,
        # and the trainer stops a run to start the next one under the same
        # session (a camera change). Until 2026-09-17 this closed it, with
        # summary.json written at the first camera change and the later
        # runs' summaries never landing.
        slot.server.current["session"] = argparse.Namespace(
            stopping=threading.Event(), run_name="sess-1", overlay=None)
        assert slot.as_trainer("POST", "/stop")[0] == 200
        time.sleep(0.1)
        assert record.closed is False and slot.server.records.open_records() == [record]
        assert f"{PREFIX}/summary.json" not in gcs.objects
        # The lease's /teardown closes it, off the request.
        slot.server.current["session"] = None
        assert slot.as_trainer("POST", "/teardown?mode=drain")[0] == 200
        settled(lambda: record.closed)
        settled(lambda: f"{PREFIX}/summary.json" in gcs.objects)
        assert json.loads(gcs.objects[f"{PREFIX}/summary.json"][0])["ended"] == "teardown"
        assert f"{PREFIX}/posts/part-" in "".join(gcs.objects)
        settled(lambda: slot.server.records.open_records() == [])

        # Another session's lease closes what the last left open.
        lease("sess-1", PREFIX)
        second = opened(PREFIX, "r2")
        lease("sess-2", other)
        settled(lambda: second.closed)
        assert json.loads(gcs.objects[f"{PREFIX}/summary.json"][0])["ended"] == "teardown", \
            "the first record's summary is untouched: objects are written once"
        # (the second record's summary was refused as already there, and
        # counted: the prefix was reused within one test only.)
        third = opened(other, "r3")
        third.append("onsets", {"atS": 2.0})
        # /teardown closes it with the lease.
        assert slot.as_trainer("POST", "/teardown?mode=drain")[0] == 200
        settled(lambda: third.closed)
        settled(lambda: f"{other}/summary.json" in gcs.objects)
        assert json.loads(gcs.objects[f"{other}/summary.json"][0])["ended"] == "teardown"
        settled(lambda: slot.server.records.open_records() == [])


def test_an_answer_without_a_fingerprint_is_refused_and_hung_up(monkeypatch):
    with Slot() as slot:
        cap = slot.lease()
        monkeypatch.setattr(tee_mode, "FINGERPRINT",
                            tee_mode.re.compile(rb"^a=never-matches$", tee_mode.re.M))
        code, headers, body = slot.request(
            "POST", "/ingest/whip", OFFER,
            {"Content-Type": "application/sdp", "Authorization": f"Bearer {cap}",
             "Origin": ORIGIN})
        assert code == 502 and "fingerprint" in json.loads(body)["error"]
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        assert "X-Masseuse-Evidence" not in headers
        # The relay session the stub created was closed behind the phone's back.
        assert slot.relay.requests[-1]["method"] == "DELETE"
        assert slot.relay.requests[-1]["path"] == f"/cam/whip/{WHIP_SECRET}"


def test_the_phone_names_an_external_camera_with_its_capability():
    link = b'{"url": "rtsps://viewer:s3cret@cam.example:7441/back?enableSrtp"}'
    json_type = {"Content-Type": "application/json"}
    with Slot() as slot:
        # Without a lease the link is refused before anything looks at it.
        cap0, _ = capability()
        code, headers, body = slot.as_phone(cap0, "PUT", "/ingest/source", link, json_type)
        assert code == 401 and json.loads(body)["reason"] == "no lease"
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        assert slot.probes == [] and slot.relay.ext_config is None
        # The preflight for a PUT is answered here, for the page.
        code, headers, _ = slot.request(
            "OPTIONS", "/ingest/source", b"",
            {"Origin": ORIGIN, "Access-Control-Request-Method": "PUT",
             "Access-Control-Request-Headers": "authorization, content-type"})
        assert code == 204 and "PUT" in headers["Access-Control-Allow-Methods"]

        cap = slot.lease()
        code, headers, body = slot.as_phone(cap, "GET", "/ingest/source")
        assert code == 200 and json.loads(body)["kind"] == "phone"
        assert headers["Cache-Control"] == "no-store"
        assert slot.as_phone(cap, "PATCH", "/ingest/source")[0] == 405
        # A wrong capability, a plain rtsp link, not JSON: each says why.
        assert slot.as_phone(cap0, "PUT", "/ingest/source", link, json_type)[0] == 401
        code, headers, body = slot.as_phone(
            cap, "PUT", "/ingest/source", b'{"url": "rtsp://cam.example/back"}', json_type)
        assert code == 400
        assert json.loads(body)["status"] == "failed"
        assert json.loads(body)["reason"] == "not-rtsps"
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/source", b"not json", json_type)
        assert code == 400 and json.loads(body)["reason"] == "bad-url"
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/source", b'["x"]', json_type)
        assert code == 400 and json.loads(body)["reason"] == "bad-url"
        assert slot.as_phone(cap, "PUT", "/ingest/source", b"x" * 5000, json_type)[0] == 413
        assert slot.relay.ext_config is None

        # The real thing: probed at the resolved address with the URL's
        # host, added to the relay pinned to that certificate, reported.
        code, headers, body = slot.as_phone(cap, "PUT", "/ingest/source", link, json_type)
        assert code == 200, body
        answer = json.loads(body)
        assert answer["status"] == "connected" and answer["kind"] == "external"
        assert answer["path"] == "ext" and answer["tracks"] == ["H264"]
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        assert slot.probes == [("cam.example", 7441, GLOBAL)]
        assert slot.relay.ext_config == {
            "source": "rtsps://viewer:s3cret@cam.example:7441/back?enableSrtp",
            "sourceFingerprint": FINGERPRINT, "rtspTransport": "tcp",
            "sourceOnDemand": False, "useAbsoluteTimestamp": True}
        code, _, body = slot.as_phone(cap, "GET", "/ingest/source")
        assert json.loads(body)["kind"] == "external"

        # The trainer's poll now follows the external camera and learns the
        # stream to produce from; the phone's own path stays in view.
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        status = json.loads(body)
        assert code == 200 and status["ready"] is True and status["tracks"] == ["H264"]
        assert status["path"] == "ext"
        assert status["streamUrl"] == "rtsp://127.0.0.1:8554/ext"
        assert status["source"]["kind"] == "external"
        assert status["source"]["type"] == "rtspSource"
        assert status["phone"] == {"ready": False, "tracks": []}
        code, _, body = slot.as_trainer("GET", "/statz")
        assert json.loads(body)["tee"]["externalCamera"]["kind"] == "external"

        # /stop restarts a production; the camera stays.
        slot.server.current["session"] = argparse.Namespace(
            stopping=threading.Event(), run_name="sess-1", overlay=None)
        assert slot.as_trainer("POST", "/stop")[0] == 200
        slot.server.current["session"] = None
        assert slot.external.active is True and slot.relay.ext_config is not None

        # A lease for another session drops it: a new phone inherits no camera.
        cap2 = slot.lease("sess-2")
        assert slot.external.active is False and slot.relay.ext_config is None
        code, _, body = slot.as_phone(cap2, "GET", "/ingest/source")
        assert json.loads(body)["kind"] == "phone"
        assert json.loads(slot.as_trainer("GET", "/ingest/status")[2])["path"] == "cam"
        # The same session re-leasing keeps it.
        assert slot.as_phone(cap2, "PUT", "/ingest/source", link, json_type)[0] == 200
        cap2b = slot.lease("sess-2")
        assert slot.external.active is True
        # The phone's DELETE goes back to its own camera.
        code, _, body = slot.as_phone(cap2b, "DELETE", "/ingest/source")
        assert code == 200 and json.loads(body)["status"] == "removed"
        assert json.loads(body)["kind"] == "phone"
        assert slot.relay.ext_config is None
        code, _, body = slot.as_phone(cap2b, "DELETE", "/ingest/source")
        assert code == 200 and json.loads(body)["status"] == "none"
        # /teardown clears whatever is attached.
        assert slot.as_phone(cap2b, "PUT", "/ingest/source", link, json_type)[0] == 200
        assert slot.as_trainer("POST", "/teardown?mode=drain")[0] == 200
        assert slot.external.active is False and slot.relay.ext_config is None

        code, _, body = slot.as_trainer("GET", "/statz")
        counters = json.loads(body)["counters"]
        assert counters["externalSourceConnected"] == 3
        assert counters["externalSourceFailed"] == 1
        assert counters["externalSourceRemoved"] == 1


def test_a_camera_that_does_not_stream_is_a_502_with_no_path_left_behind():
    with Slot() as slot:
        cap = slot.lease()
        slot.relay.ext_ready_after = 10 ** 6
        code, headers, body = slot.as_phone(
            cap, "PUT", "/ingest/source", b'{"url": "rtsps://cam.example/back"}',
            {"Content-Type": "application/json"})
        assert code == 502
        assert json.loads(body) == {
            "status": "failed", "reason": "timeout",
            "error": json.loads(body)["error"]}
        assert "did not start streaming" in json.loads(body)["error"]
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        assert slot.relay.ext_config is None and slot.relay.ext_deletes == 1
        assert slot.external.active is False
        assert json.loads(slot.as_trainer("GET", "/ingest/status")[2])["path"] == "cam"


def test_the_trainer_tells_the_slot_which_connector_to_expect_for_its_lease():
    json_type = {"Content-Type": "application/json"}
    with Slot() as slot:
        # No lease: nobody's connector can be expected yet.
        code, _, body = slot.as_trainer("POST", "/tunnel/expect",
                                        json.dumps(slot.expect()).encode(), json_type)
        assert code == 409 and json.loads(body) == {"error": "lease mismatch"}
        slot.lease("sess-1")
        # Another session's, a malformed body, a bad shape: each says why,
        # and none of it reaches the gateway.
        code, _, body = slot.as_trainer("POST", "/tunnel/expect",
                                        json.dumps(slot.expect("sess-2")).encode(), json_type)
        assert code == 409 and json.loads(body) == {"error": "lease mismatch"}
        code, _, body = slot.as_trainer("POST", "/tunnel/expect", b"not json", json_type)
        assert code == 400 and "invalid JSON" in json.loads(body)["error"]
        code, _, body = slot.as_trainer("POST", "/tunnel/expect", b'["x"]', json_type)
        assert code == 400
        for bad in ({"connectorKey": "short"}, {"ticketHash": "xyz"},
                    {"expiresAt": int(time.time()) - 5}, {"expiresAt": "soon"},
                    {"sessionId": "bad/slash"}):
            code, _, body = slot.as_trainer(
                "POST", "/tunnel/expect", json.dumps(slot.expect(**bad)).encode(), json_type)
            assert code == 400, bad
            assert next(iter(bad)) in json.loads(body)["error"], bad
        assert slot.as_trainer("POST", "/tunnel/expect", b"x" * 5000, json_type)[0] == 413
        assert slot.gateway.requests == []

        # The real thing: forwarded without the session id, answered expecting.
        expectation = slot.expect()
        code, _, body = slot.as_trainer("POST", "/tunnel/expect",
                                        json.dumps(expectation).encode(), json_type)
        assert code == 200, body
        assert json.loads(body) == {"status": "expecting"}
        assert slot.gateway.requests[-1] == {
            "method": "POST", "path": "/expect",
            "body": {"connectorKey": CONNECTOR_KEY, "ticketHash": TICKET_HASH,
                     "expiresAt": expectation["expiresAt"]}}
        assert slot.gateway.expectation["connectorKey"] == CONNECTOR_KEY
        assert slot.external.gateway.session_id == "sess-1"
        # A GET is not a control verb here.
        assert slot.as_trainer("GET", "/tunnel/expect")[0] == 404
        # The trainer's poll reads the connector's state on the source.
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        assert json.loads(body)["source"]["tunnel"] == {"connected": False, "sinceMs": None}
        assert json.loads(body)["source"]["mode"] is None
        slot.gateway.connected = True
        slot.gateway.since_ms = 1_700_000_000_000
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        assert json.loads(body)["source"]["tunnel"] == {"connected": True,
                                                        "sinceMs": 1_700_000_000_000}
        code, _, body = slot.as_trainer("GET", "/statz")
        assert json.loads(body)["tee"]["externalCamera"]["tunnel"]["connected"] is True

        # /tunnel/clear drops it.
        code, _, body = slot.as_trainer("POST", "/tunnel/clear")
        assert code == 200 and json.loads(body) == {"status": "cleared"}
        assert slot.gateway.requests[-1]["path"] == "/clear"
        assert slot.gateway.expectation is None
        assert slot.external.gateway.session_id == ""
        code, _, body = slot.as_trainer("GET", "/statz")
        counters = json.loads(body)["counters"]
        assert counters["tunnelExpected"] == 1 and counters["tunnelCleared"] == 1

    # A gateway that is not answering: 502 either way, and the status reads
    # offline rather than failing.
    with Slot(gateway_down=True) as slot:
        slot.lease("sess-1")
        code, _, body = slot.as_trainer("POST", "/tunnel/expect",
                                        json.dumps(slot.expect()).encode(), json_type)
        assert code == 502 and json.loads(body) == {"error": "gateway unavailable"}
        code, _, body = slot.as_trainer("POST", "/tunnel/clear")
        assert code == 502 and json.loads(body) == {"error": "gateway unavailable"}
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        assert code == 200
        assert json.loads(body)["source"]["tunnel"] == {"connected": False, "sinceMs": None}
        counters = json.loads(slot.as_trainer("GET", "/statz")[2])["counters"]
        assert counters["tunnelGatewayUnavailable"] == 2


def test_the_connector_goes_with_the_session_and_survives_a_stop():
    json_type = {"Content-Type": "application/json"}
    home = b'{"url": "rtsps://viewer:s3cret@192.168.1.108:7441/back?enableSrtp"}'
    with Slot() as slot:
        cap = slot.lease("sess-1")
        # A camera at home before the connector is there: refused, with the
        # reason the page words, and the gateway was only asked.
        code, headers, body = slot.as_phone(cap, "PUT", "/ingest/source", home, json_type)
        assert code == 502
        assert json.loads(body)["status"] == "failed"
        assert json.loads(body)["reason"] == "tunnel-offline"
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        assert slot.gateway.paths() == ["/status"] and slot.probes == []
        assert slot.relay.ext_config is None

        # The trainer expects the connector, the connector attaches, the phone
        # pastes the link again: target set, probed through the listener with
        # no SNI, pulled at the listener with the credentials kept.
        assert slot.as_trainer("POST", "/tunnel/expect", json.dumps(slot.expect()).encode(),
                               json_type)[0] == 200
        slot.gateway.connected = True
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/source", home, json_type)
        assert code == 200, body
        answer = json.loads(body)
        assert answer["status"] == "connected" and answer["mode"] == "tunnel"
        assert slot.gateway.target == {"host": "192.168.1.108", "port": 7441}
        assert slot.probes == [("127.0.0.1", 7441, "127.0.0.1", {"sni": False})]
        assert slot.relay.ext_config["source"] == (
            "rtsps://viewer:s3cret@127.0.0.1:7441/back?enableSrtp")
        assert slot.relay.ext_config["sourceFingerprint"] == FINGERPRINT
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        status = json.loads(body)
        assert status["streamUrl"] == "rtsp://127.0.0.1:8554/ext"
        assert status["source"]["mode"] == "tunnel"
        assert status["source"]["tunnel"]["connected"] is True
        code, _, body = slot.as_phone(cap, "GET", "/ingest/source")
        assert json.loads(body)["mode"] == "tunnel"

        # /stop restarts a production: camera and connector both stay.
        slot.server.current["session"] = argparse.Namespace(
            stopping=threading.Event(), run_name="sess-1", overlay=None)
        assert slot.as_trainer("POST", "/stop")[0] == 200
        slot.server.current["session"] = None
        assert slot.external.active is True
        assert slot.gateway.paths("POST") == ["/expect", "/target"]
        assert slot.gateway.connected is True

        # A lease for another session drops the camera and, with it, the
        # connector: a new phone inherits neither.
        cap2 = slot.lease("sess-2")
        assert slot.external.active is False
        assert slot.gateway.paths("POST") == ["/expect", "/target", "/clear"]
        assert slot.gateway.expectation is None and slot.gateway.target is None
        assert slot.external.gateway.session_id == ""
        # An expectation with no camera attached goes the same way on the
        # next lease for someone else.
        assert slot.as_trainer("POST", "/tunnel/expect",
                               json.dumps(slot.expect("sess-2")).encode(),
                               json_type)[0] == 200
        slot.lease("sess-2")  # the same session again: kept
        assert slot.gateway.paths("POST")[-1] == "/expect"
        cap3 = slot.lease("sess-3")
        assert slot.gateway.paths("POST")[-1] == "/clear"
        del cap2

        # The phone taking its own camera back drops the connector too...
        assert slot.as_trainer("POST", "/tunnel/expect",
                               json.dumps(slot.expect("sess-3")).encode(),
                               json_type)[0] == 200
        slot.gateway.connected = True
        assert slot.as_phone(cap3, "PUT", "/ingest/source", home, json_type)[0] == 200
        posted = len(slot.gateway.paths("POST"))
        code, _, body = slot.as_phone(cap3, "DELETE", "/ingest/source")
        assert code == 200 and json.loads(body)["status"] == "removed"
        assert slot.gateway.paths("POST")[posted:] == ["/clear"]
        # ...and a teardown drops whatever is left.
        assert slot.as_trainer("POST", "/tunnel/expect",
                               json.dumps(slot.expect("sess-3")).encode(),
                               json_type)[0] == 200
        assert slot.as_trainer("POST", "/teardown?mode=drain")[0] == 200
        assert slot.gateway.paths("POST")[-2:] == ["/expect", "/clear"]
        assert slot.gateway.expectation is None
        # A second teardown with nothing posted since has nothing to clear.
        assert slot.as_trainer("POST", "/teardown?mode=drain")[0] == 200
        assert slot.gateway.paths("POST")[-2:] == ["/expect", "/clear"]

        # A public camera never involves the connector, coming or going.
        cap4 = slot.lease("sess-4")
        posted = len(slot.gateway.paths("POST"))
        assert slot.as_phone(cap4, "PUT", "/ingest/source",
                             b'{"url": "rtsps://cam.example/back"}', json_type)[0] == 200
        assert slot.probes[-1] == ("cam.example", 322, GLOBAL)
        assert json.loads(slot.as_phone(cap4, "GET", "/ingest/source")[2])["mode"] == "direct"
        assert slot.as_phone(cap4, "DELETE", "/ingest/source")[0] == 200
        assert len(slot.gateway.paths("POST")) == posted


def test_a_produce_on_the_external_stream_is_landscape_and_unmirrored():
    with Slot() as slot:
        phone = argparse.Namespace(stream="rtsp://127.0.0.1:8554/cam",
                                   overlay_mirror=True, overlay_size="720x1280")
        assert producer.external_view_args(phone, slot.external) is False
        assert (phone.overlay_mirror, phone.overlay_size) == (True, "720x1280")
        fixed = argparse.Namespace(stream="rtsp://127.0.0.1:8554/ext",
                                   overlay_mirror=True, overlay_size="720x1280")
        assert producer.external_view_args(fixed, slot.external) is True
        assert (fixed.overlay_mirror, fixed.overlay_size) == (False, "1280x720")
        # The phone's camera stays live beside the fixed one: the face view
        # and the microphone are its.
        assert fixed.face_stream == "rtsp://127.0.0.1:8554/cam"
        assert fixed.audio_stream == "rtsp://127.0.0.1:8554/cam"
        assert not hasattr(phone, "face_stream")
        assert producer.external_view_args(fixed, None) is False


def test_the_phone_sets_how_its_own_picture_is_drawn_as_the_inset():
    """/ingest/view: the phone's capability sets the face inset's mirror
    and what is drawn over the pictures (`overlay`: clean until the phone
    asks for the keypoints) for the running session and the next;
    anything else is refused."""
    json_type = {"Content-Type": "application/json"}
    with Slot() as slot:
        cap = slot.lease("sess-1")
        # Read before set: the slot's defaults, no session to show them on.
        code, headers, body = slot.as_phone(cap, "GET", "/ingest/view")
        assert code == 200 and json.loads(body) == {"mirror": False, "overlay": "clean", "view": None}
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        # Preflight is answered here.
        code, headers, _ = slot.request("OPTIONS", "/ingest/view", b"", {"Origin": ORIGIN})
        assert code == 204 and headers["Access-Control-Allow-Origin"] == ORIGIN
        # A running session's overlay is told at once...
        told, layers = [], []
        slot.server.current["session"] = argparse.Namespace(
            overlay=argparse.Namespace(
                snapshot=lambda: {"view": {"layout": "inset", "mirror": True}},
                set_mirror=told.append, set_overlay=layers.append),
            run_name="sess-1")
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/view",
                                      b'{"mirror": true}', json_type)
        assert code == 200
        assert json.loads(body) == {"mirror": True, "overlay": "clean",
                                    "view": {"layout": "inset", "mirror": True}}
        assert told == [True] and layers == []
        # The keypoints, asked for on their own: the mirror stands.
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/view",
                                      b'{"overlay": "keypoints"}', json_type)
        assert code == 200 and json.loads(body)["overlay"] == "keypoints"
        assert json.loads(body)["mirror"] is True
        assert layers == ["keypoints"] and told == [True]
        # Both at once.
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/view",
                                      b'{"mirror": false, "overlay": "clean"}', json_type)
        assert code == 200 and json.loads(body)["overlay"] == "clean"
        assert told == [True, False] and layers == ["keypoints", "clean"]
        slot.as_phone(cap, "PUT", "/ingest/view", b'{"mirror": true, "overlay": "keypoints"}', json_type)
        # ...and the next session starts with them (the /produce handler
        # copies the slot's preferences into its args).
        slot.server.current["session"] = None
        code, _, body = slot.as_phone(cap, "GET", "/ingest/view")
        assert json.loads(body) == {"mirror": True, "overlay": "keypoints", "view": None}
        # The trainer reads them beside the camera in /ingest/status.
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        assert code == 200 and json.loads(body)["view"] == {"mirror": True, "overlay": "keypoints"}
        # Not a boolean, not a layer, neither key, not JSON, not the phone:
        # refused, nothing changed.
        for bad in (b'{"mirror": "yes"}', b'{"overlay": "boxes"}', b'{"overlay": 1}',
                    b'{"overlay": "keypoints", "mirror": "yes"}', b'{}', b'{"other": 1}', b'nope'):
            assert slot.as_phone(cap, "PUT", "/ingest/view", bad, json_type)[0] == 400, bad
        assert slot.request("PUT", "/ingest/view", b'{"mirror": false}',
                            {**json_type, "Origin": ORIGIN})[0] == 401
        assert slot.as_phone(cap, "DELETE", "/ingest/view")[0] == 405
        assert json.loads(slot.as_phone(cap, "GET", "/ingest/view")[2]) == {
            "mirror": True, "overlay": "keypoints", "view": None}
        # The trainer's token is not the phone's capability.
        assert slot.as_trainer("PUT", "/ingest/view", b'{"overlay": "clean"}', json_type)[0] == 401


def test_tee_mode_refuses_capture_configuration(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["producer.py", "--tee", "--analysis-socket", "",
                                      "--capture-bucket", "some-bucket"])
    with pytest.raises(SystemExit) as excinfo:
        producer.main()
    assert "capture" in str(excinfo.value)
    monkeypatch.setattr(sys, "argv", ["producer.py", "--tee", "--analysis-socket", "",
                                      "--overlay-record"])
    with pytest.raises(SystemExit) as excinfo:
        producer.main()
    assert "overlay-record" in str(excinfo.value)


def test_the_default_verifier_tolerates_a_small_clock_skew(monkeypatch):
    """A confidential VM's clock can lag the token service by a few seconds
    after boot; a token minted and presented in the same second must not
    be refused as used too early."""
    id_token = pytest.importorskip("google.oauth2.id_token")
    seen = {}

    def fake_verify_token(token, request, audience=None, **kwargs):
        seen.update(token=token, audience=audience, **kwargs)
        return {"iss": "https://accounts.google.com"}

    monkeypatch.setattr(id_token, "verify_token", fake_verify_token)
    verify = tee_mode.google_id_token_verifier()
    assert verify("t.o.k", "https://slot-0.example") == {"iss": "https://accounts.google.com"}
    assert seen["audience"] == "https://slot-0.example"
    assert seen["clock_skew_in_seconds"] == tee_mode.TOKEN_CLOCK_SKEW_S == 30


def test_the_phone_opens_a_live_stream_with_its_capability_and_the_trainer_may_only_stop_it():
    json_type = {"Content-Type": "application/json"}
    destination = b'{"url": "rtmps://live.example.com/app/sk_live_1234", "audio": true}'
    with Slot() as slot:
        # Without a lease the destination is refused before anything looks at it.
        cap0, _ = capability()
        code, headers, body = slot.as_phone(cap0, "PUT", "/ingest/egress", destination, json_type)
        assert code == 401 and json.loads(body)["reason"] == "no lease"
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        assert slot.egress_procs == []
        code, headers, _ = slot.request(
            "OPTIONS", "/ingest/egress", b"",
            {"Origin": ORIGIN, "Access-Control-Request-Method": "PUT",
             "Access-Control-Request-Headers": "authorization, content-type"})
        assert code == 204 and "PUT" in headers["Access-Control-Allow-Methods"]

        cap = slot.lease()
        code, headers, body = slot.as_phone(cap, "GET", "/ingest/egress")
        assert code == 200 and json.loads(body)["active"] is False
        assert headers["Cache-Control"] == "no-store"
        assert slot.as_phone(cap, "PATCH", "/ingest/egress")[0] == 405
        # A plain rtmp address, a private host, not JSON: each says why, and nothing starts.
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/egress", b'{"url": "rtmp://live.example.com/app/k"}', json_type)
        assert code == 400 and json.loads(body) == {"status": "failed", "reason": "not-rtmps",
                                                     "error": json.loads(body)["error"]}
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/egress", b"not json", json_type)
        assert code == 400 and json.loads(body)["reason"] == "bad-url"
        assert slot.as_phone(cap, "PUT", "/ingest/egress", b"x" * 5000, json_type)[0] == 413
        assert slot.egress_procs == []

        # The real thing: started, the answer and the status name the host and never the address.
        code, headers, body = slot.as_phone(cap, "PUT", "/ingest/egress", destination, json_type)
        assert code == 200, body
        answer = json.loads(body)
        assert answer["status"] == "connected" and answer["active"] is True
        assert answer["host"] == "live.example.com" and answer["audio"] is True and answer["hud"] is True
        assert "sk_live" not in body.decode()
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        assert len(slot.egress_procs) == 1
        assert slot.egress_procs[0].argv[-1] == "rtmps://live.example.com/app/sk_live_1234"
        assert "-c:a" in slot.egress_procs[0].argv and "pipe:0" in slot.egress_procs[0].argv

        # The trainer's poll says a stream is on, to which host; nothing more.
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        status = json.loads(body)
        assert code == 200 and status["egress"]["active"] is True
        assert status["egress"]["host"] == "live.example.com"
        assert "sk_live" not in body.decode()
        # The trainer's HUD card state is taken (its identity, PUT only); the phone's capability is not.
        card = json.dumps({"tiles": [{"key": "clench", "label": "Clench rate", "value": "36", "unit": "/min", "state": "live"}],
                           "unit": {"name": "MK-312BT", "detail": "Stroke", "tone": "live", "level": 35, "max": 70},
                           "fans": {"watching": 4, "controlling": 1}}).encode()
        code, _, body = slot.as_trainer("PUT", "/overlay/hud", card, json_type)
        assert code == 200 and json.loads(body) == {"status": "ok", "tiles": 1, "egress": True}
        assert slot.as_trainer("POST", "/overlay/hud", card, json_type)[0] == 405
        assert slot.as_phone(cap, "PUT", "/overlay/hud", card, json_type)[0] == 401
        assert slot.request("PUT", "/overlay/hud", card, json_type)[0] == 401
        code, _, body = slot.as_trainer("PUT", "/overlay/hud", b"[1,2]", json_type)
        assert code == 400
        assert slot.egress.status()["card"]["held"] is True

        # The trainer may stop the stream, and only stop it: no PUT for it on the phone's route.
        assert slot.as_trainer("PUT", "/ingest/egress", destination, json_type)[0] == 401
        assert slot.request("POST", "/egress/stop")[0] == 401
        assert slot.as_trainer("GET", "/egress/stop")[0] == 405
        code, _, body = slot.as_trainer("POST", "/egress/stop")
        assert code == 200 and json.loads(body)["status"] == "stopped" and json.loads(body)["active"] is False
        assert slot.egress_procs[0].returncode == -15
        code, _, body = slot.as_trainer("POST", "/egress/stop")
        assert json.loads(body)["status"] == "none"

        # The phone's own DELETE; a stream survives /stop and goes with a teardown.
        slot.as_phone(cap, "PUT", "/ingest/egress", destination, json_type)
        assert len(slot.egress_procs) == 2
        code, _, body = slot.as_phone(cap, "DELETE", "/ingest/egress")
        assert code == 200 and json.loads(body)["status"] == "removed"
        assert slot.egress_procs[1].returncode == -15
        code, _, body = slot.as_phone(cap, "DELETE", "/ingest/egress")
        assert json.loads(body)["status"] == "none"
        slot.as_phone(cap, "PUT", "/ingest/egress", destination, json_type)
        assert len(slot.egress_procs) == 3
        slot.as_trainer("POST", "/stop")  # 404: nothing running, and the lease stands
        assert slot.egress.active is True, "a /stop restarts a production; the stream stays"
        code, _, _ = slot.as_trainer("POST", "/teardown?mode=now")
        assert code == 200
        assert slot.egress.active is False
        assert slot.egress_procs[2].returncode == -15
        # The teardown asks the process to exit half a second later; the
        # harness records the ask instead of taking it.
        deadline = time.time() + 3.0
        while not slot.exits and time.time() < deadline:
            time.sleep(0.05)
        assert slot.exits == [0]


def test_a_lease_for_another_session_takes_the_live_stream_with_it():
    json_type = {"Content-Type": "application/json"}
    destination = b'{"url": "rtmps://live.example.com/app/sk_live_1234"}'
    with Slot() as slot:
        cap = slot.lease("sess-1")
        assert slot.as_phone(cap, "PUT", "/ingest/egress", destination, json_type)[0] == 200
        assert slot.egress.active is True and slot.egress.session_id == "sess-1"
        # The same session leasing again (a trainer restart) keeps it.
        slot.lease("sess-1")
        assert slot.egress.active is True
        # Another session's lease does not inherit it.
        slot.lease("sess-2")
        assert slot.egress.active is False
        assert slot.egress_procs[0].returncode == -15


def test_the_phone_sends_its_picture_to_the_connector_with_its_capability_when_the_connector_is_there():
    with Slot() as slot:
        # Without a lease, nothing; the preflight is answered.
        cap0, _ = capability()
        code, headers, body = slot.as_phone(cap0, "PUT", "/ingest/share")
        assert code == 401 and json.loads(body)["reason"] == "no lease"
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        code, headers, _ = slot.request(
            "OPTIONS", "/ingest/share", b"",
            {"Origin": ORIGIN, "Access-Control-Request-Method": "PUT",
             "Access-Control-Request-Headers": "authorization"})
        assert code == 204 and "PUT" in headers["Access-Control-Allow-Methods"]

        cap = slot.lease()
        code, headers, body = slot.as_phone(cap, "GET", "/ingest/share")
        assert code == 200 and json.loads(body)["active"] is False
        assert headers["Cache-Control"] == "no-store"
        assert slot.as_phone(cap, "PATCH", "/ingest/share")[0] == 405
        # No connector attached: refused before anything is probed or spawned.
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/share")
        assert code == 502 and json.loads(body)["reason"] == "connector-offline"
        assert slot.share_procs == []
        # The trainer has no PUT here, nor anyone without the capability.
        assert slot.as_trainer("PUT", "/ingest/share")[0] == 401
        assert slot.request("PUT", "/ingest/share")[0] == 401

        # The connector attached: the picture goes, copied off the phone's
        # path into the bridge to the connector's phone path.
        slot.gateway.connected = True
        code, headers, body = slot.as_phone(cap, "PUT", "/ingest/share")
        assert code == 200, body
        answer = json.loads(body)
        assert answer["status"] == "connected" and answer["active"] is True
        assert answer["since"] == 1_700_000_000.0 and answer["refused"] is False
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        deadline = time.time() + 3.0
        while len(slot.share_procs) < 1 and time.time() < deadline:
            time.sleep(0.02)
        assert len(slot.share_procs) == 1
        argv = slot.share_procs[0].argv
        assert argv[argv.index("-i") + 1] == "rtsp://127.0.0.1:8554/cam"
        assert argv[-1] == "rtsp://127.0.0.1:5555/phone" and "copy" in argv and "-an" in argv
        assert slot.share.pin == FINGERPRINT.lower()
        # The gateway was asked whether the connector is there; it was not
        # given a target (the own listener needs none).
        assert slot.gateway.target is None

        # The trainer's poll says the picture is going; the statz too.
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        status = json.loads(body)
        assert code == 200 and status["share"]["active"] is True
        assert status["share"]["since"] == 1_700_000_000.0
        assert status["faceSource"]["kind"] == "phone" and status["faceSource"]["streamUrl"] is None
        code, _, body = slot.as_trainer("GET", "/statz")
        assert json.loads(body)["share"]["active"] is True

        # The phone's DELETE stops it; it survives /stop; a teardown ends it.
        code, _, body = slot.as_phone(cap, "DELETE", "/ingest/share")
        assert code == 200 and json.loads(body)["status"] == "removed"
        assert slot.share_procs[0].returncode == -15
        code, _, body = slot.as_phone(cap, "DELETE", "/ingest/share")
        assert json.loads(body)["status"] == "none"
        assert slot.as_phone(cap, "PUT", "/ingest/share")[0] == 200
        slot.as_trainer("POST", "/stop")
        assert slot.share.active is True, "a /stop restarts a production; the share stays"
        code, _, _ = slot.as_trainer("POST", "/teardown?mode=now")
        assert code == 200
        assert slot.share.active is False
        deadline = time.time() + 3.0
        while not slot.exits and time.time() < deadline:
            time.sleep(0.05)
        assert slot.exits == [0]


def test_a_lease_for_another_session_takes_the_share_and_the_face_camera_with_it():
    json_type = {"Content-Type": "application/json"}
    with Slot() as slot:
        slot.gateway.connected = True
        cap = slot.lease("sess-1")
        assert slot.as_phone(cap, "PUT", "/ingest/share")[0] == 200
        slot.relay.face_ready_after = 1
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/face-source",
                                      b'{"url": "rtsps://127.0.0.1:7443/face"}', json_type)
        assert code == 200, body
        assert slot.share.active and slot.face_source.active
        # The same session leasing again (a trainer restart) keeps both.
        slot.lease("sess-1")
        assert slot.share.active and slot.face_source.active
        # Another session's lease inherits neither.
        slot.lease("sess-2")
        assert not slot.share.active and not slot.face_source.active
        assert slot.relay.face_config is None


def test_the_connectors_face_camera_is_the_face_view_while_attached_and_never_the_body():
    json_type = {"Content-Type": "application/json"}
    with Slot() as slot:
        cap = slot.lease()
        code, _, body = slot.as_phone(cap, "GET", "/ingest/face-source")
        assert code == 200 and json.loads(body)["kind"] == "phone"
        # Not while the connector is away.
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/face-source",
                                      b'{"url": "rtsps://127.0.0.1:7443/face"}', json_type)
        assert code == 502 and json.loads(body)["reason"] == "tunnel-offline"
        # A link that is not rtsps, or not JSON, says why.
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/face-source",
                                      b'{"url": "rtsp://127.0.0.1:7443/face"}', json_type)
        assert code == 400
        assert slot.as_phone(cap, "PUT", "/ingest/face-source", b"nope", json_type)[0] == 400
        assert slot.as_trainer("PUT", "/ingest/face-source", b"{}", json_type)[0] == 401

        slot.gateway.connected = True
        slot.relay.face_ready_after = 2
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/face-source",
                                      b'{"url": "rtsps://127.0.0.1:7443/face"}', json_type)
        assert code == 200, body
        answer = json.loads(body)
        assert answer["status"] == "connected" and answer["path"] == "face-ext" and answer["mode"] == "tunnel"
        # Through the own listener, pinned; no target set for it.
        assert slot.probes[-1] == ("127.0.0.1", 7442, "127.0.0.1", {"sni": False})
        assert slot.gateway.target is None
        assert slot.relay.face_config["source"] == "rtsps://127.0.0.1:7442/face"
        # The body camera is still the phone's: the top of /ingest/status is
        # the phone's path, and the face source rides beside it with its
        # stream for /produce.
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        status = json.loads(body)
        assert status["path"] == "cam" and status["source"]["kind"] == "phone"
        assert status["faceSource"]["kind"] == "external" and status["faceSource"]["path"] == "face-ext"
        assert status["faceSource"]["streamUrl"] == "rtsp://127.0.0.1:8554/face-ext"
        assert status["faceSource"]["mode"] == "tunnel"
        # The phone puts its own camera back; the connector stays attached
        # (the body camera may be its), and the path is gone.
        code, _, body = slot.as_phone(cap, "DELETE", "/ingest/face-source")
        assert code == 200 and json.loads(body)["status"] == "removed"
        assert slot.gateway.connected is True and slot.relay.face_config is None
        code, _, body = slot.as_trainer("GET", "/ingest/status")
        assert json.loads(body)["faceSource"]["streamUrl"] is None
        assert json.loads(slot.as_phone(cap, "DELETE", "/ingest/face-source")[2])["status"] == "none"
