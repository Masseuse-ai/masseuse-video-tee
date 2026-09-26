"""The ACME External Account Binding for this boot, minted inside the enclave.

Google Trust Services (through Cloud Public CA) issues the slot's TLS
certificate, and its ACME directory admits only accounts that present an
External Account Binding minted with `externalAccountKeys.create` on the
project. The enclave registers a fresh ACME account every boot - the key
and the certificate live on the /run/tee tmpfs and die with it, and the
slot boots once per session - so the EAB is minted here, at boot, with the
enclave's own credential: the `external_account` file the entrypoint
renders, whose subject token is the launcher's attestation claims token.
The WIF provider admits it only for this image digest on TDX with the GPU
in CC mode, and the digest-pinned principal is the only identity in the
project with `roles/publicca.externalAccountKeyCreator`
(masseuse-video-tee/terraform/acme.tf). No operator can mint an EAB and
register an account in the slot's name.

Run once by tee/entrypoint.sh before Caddy starts:

    python3 tee_eab.py --directory "$ACME_DIRECTORY_URL" --out /run/tee/eab.env

`--out` receives `EAB_KID=...` and `EAB_HMAC=...` (mode 0600) for the
entrypoint to export into Caddy's environment, where the Caddyfile's
`eab {$EAB_KID} {$EAB_HMAC}` reads them; the HMAC is never printed. The
endpoint follows the directory: GTS's staging directory
(dv.acme-v02.test-api.pki.goog) takes EABs from preprod-publicca, the
production one from publicca. `b64MacKey` in the REST answer is a protobuf
bytes field - standard base64 of the base64url HMAC the ACME client wants -
so it is decoded exactly once (masseuse-video-tee/tools/gts-acme-test.sh,
which proved both the decode and Caddy's `eab` against GTS).

`--impersonate` is the contingency for a Public CA API that will not take a
federated token directly: mint through a service account the principal may
impersonate (`roles/iam.serviceAccountTokenCreator` on it), which keeps
the digest pinning. Unused unless ACME_EAB_IMPERSONATE is set.

Exit code non-zero on any failure so the entrypoint can retry (the claims
token can race the first attempt).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

SCOPE = "https://www.googleapis.com/auth/cloud-platform"
PRODUCTION_HOST = "publicca.googleapis.com"
STAGING_HOST = "preprod-publicca.googleapis.com"
# A base64url HS256 key as ACME clients take it: at least 32 characters.
MAC_KEY_TEXT = re.compile(r"^[A-Za-z0-9_-]{32,}$")


def eab_endpoint(directory_url: str, project: str) -> str:
    """The externalAccountKeys.create URL that pairs with an ACME directory:
    preprod- for GTS staging (any test-api.pki.goog host), production
    otherwise."""
    host = (urlparse(directory_url).hostname or "").lower()
    api = STAGING_HOST if host.endswith("test-api.pki.goog") else PRODUCTION_HOST
    return f"https://{api}/v1/projects/{project}/locations/global/externalAccountKeys"


def mac_key_for_acme(b64_mac_key: str) -> str:
    """`b64MacKey` decoded once: the JSON carries standard base64 of the
    base64url HMAC text. If the decode does not read as such text the raw
    value is returned (the API has been consistent; this is the seat belt)."""
    try:
        decoded = base64.b64decode(b64_mac_key, validate=True).decode("ascii").strip()
    except (ValueError, UnicodeDecodeError):
        return b64_mac_key.strip()
    if MAC_KEY_TEXT.match(decoded):
        return decoded
    return b64_mac_key.strip()


def credentials(impersonate: str | None):
    import google.auth
    from google.auth.transport.requests import Request

    source, _ = google.auth.default(scopes=[SCOPE])
    if impersonate:
        from google.auth import impersonated_credentials

        source = impersonated_credentials.Credentials(
            source_credentials=source, target_principal=impersonate,
            target_scopes=[SCOPE], lifetime=300)
    source.refresh(Request())
    return source


def mint(endpoint: str, token: str, timeout_s: float = 20.0) -> tuple[str, str]:
    """POST an empty ExternalAccountKey; (keyId, mac key for the client)."""
    request = urllib.request.Request(
        endpoint, data=b"{}", method="POST",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"{error.code} from {urlparse(endpoint).hostname}: {detail}") from None
    key_id = str(body.get("keyId") or "").strip()
    b64_mac_key = str(body.get("b64MacKey") or "").strip()
    if not key_id or not b64_mac_key:
        raise RuntimeError(f"answer without keyId/b64MacKey: {json.dumps(body)[:400]}")
    return key_id, mac_key_for_acme(b64_mac_key)


def write_env(path: str, key_id: str, mac_key: str) -> None:
    """`EAB_KID=..`/`EAB_HMAC=..` for `set -a; . path`, created 0600 so the
    HMAC is readable by this uid only, and never echoed."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(f"EAB_KID={key_id}\nEAB_HMAC={mac_key}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--directory", default=os.environ.get("ACME_DIRECTORY_URL", ""),
                        help="the ACME directory URL Caddy will use; picks the "
                             "Public CA endpoint")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
                        help="the project whose externalAccountKeys are minted")
    parser.add_argument("--endpoint", default="",
                        help="override the externalAccountKeys.create URL")
    parser.add_argument("--impersonate",
                        default=os.environ.get("ACME_EAB_IMPERSONATE", ""),
                        help="service account to mint through (contingency)")
    parser.add_argument("--out", required=True,
                        help="where EAB_KID/EAB_HMAC are written (0600)")
    args = parser.parse_args(argv)
    if not args.project:
        print("tee-eab: no project (GOOGLE_CLOUD_PROJECT)", flush=True)
        return 2
    if not args.directory and not args.endpoint:
        print("tee-eab: no ACME directory (ACME_DIRECTORY_URL)", flush=True)
        return 2
    endpoint = args.endpoint or eab_endpoint(args.directory, args.project)
    started = time.monotonic()
    try:
        token = credentials(args.impersonate or None).token
        key_id, mac_key = mint(endpoint, token)
        write_env(args.out, key_id, mac_key)
    except Exception as error:  # noqa: BLE001 - the entrypoint retries on any failure
        print(f"tee-eab: failed: {error}", flush=True)
        return 1
    via = f" via {args.impersonate}" if args.impersonate else ""
    print(f"tee-eab: EAB {key_id} from {urlparse(endpoint).hostname}{via} in "
          f"{(time.monotonic() - started) * 1000:.0f} ms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
