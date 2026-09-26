"""Where the producer's outputs land: JSONL capture, and optionally a POST.

The capture files are the session's record - the analysis process's onsets,
paired events and payloads, the posted readings and the raw pose rows, one
JSON object per line, written as they are emitted so a crash loses nothing
already decided. Nothing in them is a frame. `upload` copies them to GCS
when a named test session ends (never in --tee mode, where no capture
leaves the enclave); a named session's capture is what `SideloadPose`
replays. The POST client sends the readings to the trainer; it stays
optional and failures are counted, never fatal - a sink must not be able to
stall the pipeline.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class Capture:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._files = {
            name: open(directory / f"{name}.jsonl", "a", buffering=1)
            for name in ("onsets", "events", "payloads", "posts", "poses")
        }

    def pose(self, row: dict) -> None:
        """Every pose row the worker produced, in production numerics.

        Dropped (queue full) and errored steps are written too, keypoint-less
        and flagged, because the gaps are part of what the analysis saw: a
        spike is a displacement over the real dt between posed rows. Nothing
        else keeps the keypoints - the assembler interpolates them away and
        the readings carry only derived numbers - so without this file a
        session can never be replayed or explained.
        """
        self._files["poses"].write(json.dumps(row) + "\n")

    def onset(self, at_s: float) -> None:
        self._files["onsets"].write(json.dumps({"atS": round(at_s, 3)}) + "\n")

    def event(self, start_s: float, release_s: float) -> None:
        self._files["events"].write(json.dumps(
            {"startS": round(start_s, 3),
             "releaseS": round(release_s, 3)}) + "\n")

    def payload(self, payload: dict) -> None:
        self._files["payloads"].write(json.dumps(payload) + "\n")

    def post(self, body: dict) -> None:
        """A reading from the analysis process, at its true availability.

        Recorded whether or not a POST sink is configured: these rows are
        what a replay of the session's consumers runs against, with real
        availability rather than modeled lags.
        """
        self._files["posts"].write(json.dumps(body) + "\n")

    def summary(self, summary: dict) -> None:
        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()

    def upload(self, bucket: str, prefix: str, telemetry=None,
               client_factory=None) -> dict:
        """Copy the capture files to gs://bucket/prefix/, file by file.

        The capture dir dies with the instance, so this is how a session's
        rows leave the machine. Best effort like every sink: a missing
        client library, a refused bucket or one failed file are counted and
        reported, never raised into the request that is winding down.
        """
        prefix = prefix.strip("/")
        files = sorted(p for p in self.directory.iterdir() if p.is_file())
        uploaded: list[str] = []
        failed: list[str] = []
        try:
            if client_factory is None:
                from google.cloud import storage  # noqa: PLC0415
                client_factory = storage.Client
            target = client_factory().bucket(bucket)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            failed = [p.name for p in files]
            if telemetry:
                telemetry.count("captureUploadFailed", len(failed))
            return {"bucket": bucket, "prefix": prefix, "uploaded": uploaded,
                    "failed": failed, "error": repr(error)}
        for path in files:
            try:
                target.blob(f"{prefix}/{path.name}").upload_from_filename(
                    str(path))
                uploaded.append(path.name)
            except Exception:  # noqa: BLE001 - one bad file, keep going
                failed.append(path.name)
        if telemetry:
            if uploaded:
                telemetry.count("captureUploaded", len(uploaded))
            if failed:
                telemetry.count("captureUploadFailed", len(failed))
        return {"bucket": bucket, "prefix": prefix, "uploaded": uploaded,
                "failed": failed}


METADATA_IDENTITY_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/identity"
)

# Google-signed ID tokens live an hour; re-mint with margin.
TOKEN_REFRESH_S = 50 * 60.0


def metadata_identity_token(audience: str) -> str | None:
    """A Google-signed OIDC ID token for this instance's service account.

    The deployed ingest route verifies audience and caller identity; on a
    machine without a metadata server (the local loop) this returns None and
    the POST goes unauthenticated, which the route accepts only outside
    production - the same loopback/dev bypass its tests use.
    """
    # format=full is load-bearing: without it the metadata server omits the
    # email claim, and the ingest route verifies the caller BY email - the
    # deployed smoke measured exactly this refusal ("producer identity:
    # unknown") on a token minted without it.
    request = urllib.request.Request(
        f"{METADATA_IDENTITY_URL}?audience={urllib.parse.quote(audience)}"
        "&format=full",
        headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.read().decode().strip()
    except (urllib.error.URLError, OSError):
        return None


class Poster:
    """Best-effort POST of the analysis's readings to the trainer.

    The analysis process assembles the body at its own cadence; this class
    only signs and sends. Authenticates like the other machine callers in
    this stack: a metadata OIDC ID token with the post URL's origin as
    audience, refreshed before the hour is up. A sink must never stall the
    pipeline, so token failures and post failures are counted, not raised.

    Sends happen on the poster's own thread. Each post is a fresh HTTPS
    connection, 50-100 ms of handshake even when the far end answers in
    2 ms, and at 1 Hz that is a tenth of a second the descriptors thread
    cannot spare. `post` hands the body over and returns; a body still
    unsent when the next arrives is replaced, since the consumer wants the
    newest reading, not a backlog of stale ones.
    """

    def __init__(self, url: str, telemetry=None, timeout_s: float = 2.0,
                 token_supplier=None):
        self.url = url
        self.telemetry = telemetry
        self.timeout_s = timeout_s
        parts = urllib.parse.urlsplit(url)
        self.audience = f"{parts.scheme}://{parts.netloc}"
        self._token_supplier = (token_supplier
                                or (lambda: metadata_identity_token(
                                    self.audience)))
        self._token: str | None = None
        self._token_at = 0.0
        self._state = threading.Condition()
        self._pending: dict | None = None
        self._closed = False
        self._sender = threading.Thread(target=self._send_loop, daemon=True)
        self._sender.start()

    def _headers(self) -> dict:
        import time
        headers = {"Content-Type": "application/json"}
        if (self._token is None
                or time.monotonic() - self._token_at > TOKEN_REFRESH_S):
            self._token = self._token_supplier()
            self._token_at = time.monotonic()
            if self._token is None and self.telemetry:
                self.telemetry.count("postsUnauthenticated")
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def post(self, body: dict) -> None:
        """Hand a body to the sender; returns at once. Newest wins."""
        with self._state:
            if self._closed:
                return
            if self._pending is not None and self.telemetry:
                self.telemetry.count("postsCoalesced")
            self._pending = body
            self._state.notify()

    def close(self, timeout_s: float | None = None) -> None:
        """Send whatever is pending, then stop the sender."""
        with self._state:
            self._closed = True
            self._state.notify()
        self._sender.join(timeout_s)

    def _send_loop(self) -> None:
        while True:
            with self._state:
                while self._pending is None and not self._closed:
                    self._state.wait()
                if self._pending is None:
                    return
                body, self._pending = self._pending, None
            self._send(body)

    def _send(self, body: dict) -> None:
        try:
            request = urllib.request.Request(
                self.url, data=json.dumps(body).encode(),
                headers=self._headers())
            with urllib.request.urlopen(request, timeout=self.timeout_s):
                pass
            if self.telemetry:
                self.telemetry.count("postsOk")
        except (urllib.error.URLError, OSError, ValueError):
            if self.telemetry:
                self.telemetry.count("postsFailed")
