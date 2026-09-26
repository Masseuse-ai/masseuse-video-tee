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
from external_source import ExternalSource  # noqa: E402
from relay_proxy import RelayProxy  # noqa: E402
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
        self.server = producer.build_server(
            server_args(overlay_publish="rtsp://127.0.0.1:8554/overlay",
                        overlay_renditions="hi,half,small,lean",
                        overlay_relay_webrtc=self.relay.base,
                        overlay_relay_api=self.relay.base),
            Telemetry(), tee=self.tee, external=self.external)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.current["session"] = None
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
        assert snapshot["tee"]["lease"] == {"sessionId": None, "active": False,
                                            "expiresAt": None}
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


def test_lease_validation_and_stop_clearing():
    with Slot() as slot:
        code, _, body = slot.as_trainer("POST", "/lease", b"not json",
                                        {"Content-Type": "application/json"})
        assert code == 400
        code, _, body = slot.as_trainer(
            "POST", "/lease", json.dumps({"sessionId": "s", "capabilityHash": "zz"}).encode())
        assert code == 400 and "capabilityHash" in json.loads(body)["error"]
        assert slot.as_trainer("POST", "/lease", b"x" * 5000)[0] == 413
        cap = slot.lease()
        assert slot.tee.lease.active()
        slot.server.current["session"] = argparse.Namespace(
            stopping=threading.Event(), run_name="sess-1", overlay=None)
        code, _, _ = slot.as_trainer("POST", "/stop")
        assert code == 200
        assert slot.tee.lease.active() is False
        assert slot.tee.lease.check(cap) == (False, "no lease")


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
    for the running session and the next; anything else is refused."""
    json_type = {"Content-Type": "application/json"}
    with Slot() as slot:
        cap = slot.lease("sess-1")
        # Read before set: the slot's default, no session to show it on.
        code, headers, body = slot.as_phone(cap, "GET", "/ingest/view")
        assert code == 200 and json.loads(body) == {"mirror": False, "view": None}
        assert headers["Access-Control-Allow-Origin"] == ORIGIN
        # Preflight is answered here.
        code, headers, _ = slot.request("OPTIONS", "/ingest/view", b"", {"Origin": ORIGIN})
        assert code == 204 and headers["Access-Control-Allow-Origin"] == ORIGIN
        # A running session's overlay is told at once...
        told = []
        slot.server.current["session"] = argparse.Namespace(
            overlay=argparse.Namespace(
                snapshot=lambda: {"view": {"layout": "inset", "mirror": True}},
                set_mirror=told.append),
            run_name="sess-1")
        code, _, body = slot.as_phone(cap, "PUT", "/ingest/view",
                                      b'{"mirror": true}', json_type)
        assert code == 200
        assert json.loads(body) == {"mirror": True,
                                    "view": {"layout": "inset", "mirror": True}}
        assert told == [True]
        # ...and the next session starts with it (the /produce handler
        # copies the slot's preference into its args).
        slot.server.current["session"] = None
        code, _, body = slot.as_phone(cap, "GET", "/ingest/view")
        assert json.loads(body) == {"mirror": True, "view": None}
        # Not a boolean, not JSON, not the phone: refused, nothing changed.
        assert slot.as_phone(cap, "PUT", "/ingest/view", b'{"mirror": "yes"}', json_type)[0] == 400
        assert slot.as_phone(cap, "PUT", "/ingest/view", b'nope', json_type)[0] == 400
        assert slot.request("PUT", "/ingest/view", b'{"mirror": false}',
                            {**json_type, "Origin": ORIGIN})[0] == 401
        assert slot.as_phone(cap, "DELETE", "/ingest/view")[0] == 405
        assert json.loads(slot.as_phone(cap, "GET", "/ingest/view")[2])["mirror"] is True
        # The trainer's token is not the phone's capability.
        assert slot.as_trainer("PUT", "/ingest/view", b'{"mirror": false}', json_type)[0] == 401


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
