"""tee_eab.py without the network: the endpoint follows the directory, the
REST `b64MacKey` is decoded exactly once, the answer is parsed, and the env
file for the entrypoint is written 0600 with nothing echoed."""

from __future__ import annotations

import base64
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))

import tee_eab  # noqa: E402

PROJECT = "prod-masseuse-video-tee"
# What GTS hands out: 43+ base64url characters of HS256 key.
MAC_TEXT = base64.urlsafe_b64encode(b"\x01" * 48).rstrip(b"=").decode()


def test_endpoint_follows_the_directory():
    production = tee_eab.eab_endpoint("https://dv.acme-v02.api.pki.goog/directory", PROJECT)
    staging = tee_eab.eab_endpoint("https://dv.acme-v02.test-api.pki.goog/directory", PROJECT)
    assert production == ("https://publicca.googleapis.com/v1/projects/"
                          f"{PROJECT}/locations/global/externalAccountKeys")
    assert staging.startswith("https://preprod-publicca.googleapis.com/v1/projects/")
    # Anything that is not GTS staging mints from production: a typo in the
    # directory never silently binds a production account to a preprod key.
    assert tee_eab.eab_endpoint("https://example.test/dir", PROJECT) == production


def test_mac_key_is_decoded_exactly_once():
    wire = base64.b64encode(MAC_TEXT.encode()).decode()
    assert tee_eab.mac_key_for_acme(wire) == MAC_TEXT
    # Already the text (a future API that stops double-encoding): unchanged.
    assert tee_eab.mac_key_for_acme(MAC_TEXT) == MAC_TEXT
    # Standard base64 of random bytes is not base64url text: left alone.
    binary = base64.b64encode(b"\xff\xfe" * 24).decode()
    assert tee_eab.mac_key_for_acme(binary) == binary


class _Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_mint_parses_the_answer(monkeypatch):
    seen = {}

    def urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        seen["body"] = request.data
        body = {"name": f"projects/{PROJECT}/locations/global/externalAccountKeys/k1",
                "keyId": "k1",
                "b64MacKey": base64.b64encode(MAC_TEXT.encode()).decode()}
        return _Answer(json.dumps(body).encode())

    monkeypatch.setattr(tee_eab.urllib.request, "urlopen", urlopen)
    key_id, mac = tee_eab.mint("https://publicca.googleapis.com/v1/x", "tok")
    assert (key_id, mac) == ("k1", MAC_TEXT)
    assert seen == {"url": "https://publicca.googleapis.com/v1/x",
                    "auth": "Bearer tok", "body": b"{}"}


def test_mint_surfaces_the_api_error(monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {},
                                     io.BytesIO(b'{"error":{"message":"denied"}}'))

    monkeypatch.setattr(tee_eab.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="403 from publicca.googleapis.com: .*denied"):
        tee_eab.mint("https://publicca.googleapis.com/v1/x", "tok")


def test_mint_refuses_an_answer_without_a_key(monkeypatch):
    monkeypatch.setattr(tee_eab.urllib.request, "urlopen",
                        lambda request, timeout: _Answer(b'{"name":"x"}'))
    with pytest.raises(RuntimeError, match="without keyId/b64MacKey"):
        tee_eab.mint("https://publicca.googleapis.com/v1/x", "tok")


def test_env_file_is_private_and_sourceable(tmp_path):
    out = tmp_path / "eab.env"
    tee_eab.write_env(str(out), "k1", MAC_TEXT)
    assert oct(out.stat().st_mode & 0o777) == "0o600"
    assert out.read_text() == f"EAB_KID=k1\nEAB_HMAC={MAC_TEXT}\n"


def test_main_never_prints_the_hmac(monkeypatch, tmp_path, capsys):
    class Credential:
        token = "tok"

    monkeypatch.setattr(tee_eab, "credentials", lambda impersonate: Credential())
    monkeypatch.setattr(tee_eab, "mint", lambda endpoint, token: ("k1", MAC_TEXT))
    out = tmp_path / "eab.env"
    assert tee_eab.main(["--directory", "https://dv.acme-v02.api.pki.goog/directory",
                         "--project", PROJECT, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "k1" in printed and "publicca.googleapis.com" in printed
    assert MAC_TEXT not in printed
    assert out.read_text().endswith(f"{MAC_TEXT}\n")


def test_main_fails_closed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tee_eab, "credentials",
                        lambda impersonate: (_ for _ in ()).throw(RuntimeError("no token yet")))
    assert tee_eab.main(["--directory", "https://dv.acme-v02.api.pki.goog/directory",
                         "--project", PROJECT, "--out", str(tmp_path / "eab.env")]) == 1
    assert "no token yet" in capsys.readouterr().out
    assert not (tmp_path / "eab.env").exists()
    assert tee_eab.main(["--out", str(tmp_path / "e"), "--directory", "x"]) == 2
