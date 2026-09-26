"""A session with two views (producer.Session with `face_stream`).

The pose, the decoders and the analysis link are stubs; what is pinned is
the wiring: the face stream gets its own decoder that waits for its track,
its frames are posed as the `face` view at the face cadence through the
same pose, its rows go out as `facePose` with the body view's moment for
them, the audio stage reads the stream named for it, `hello` says which
views there are, and /statz's `views` says how they are timed.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

WORKLOAD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOAD / "producer"))
sys.path.insert(0, str(WORKLOAD / "pixel"))

import producer  # noqa: E402
from live_pose import ViewState  # noqa: E402
from telemetry import Telemetry  # noqa: E402

BODY_URL = "rtsp://127.0.0.1:8554/ext"
FACE_URL = "rtsp://127.0.0.1:8554/cam"
WIDTH, HEIGHT = 32, 16


class StubPose:
    """GpuPose's surface as the session uses it."""

    def __init__(self):
        self.views = {"body": ViewState()}
        self.steps: list[tuple] = []
        self.lock = threading.Lock()

    @property
    def on_full(self):
        return self.views["body"].on_full

    @on_full.setter
    def on_full(self, value):
        self.views["body"].on_full = value

    def view(self, name):
        return self.views.setdefault(name or "body", ViewState())

    def reset_session_state(self):
        pass

    def boot(self):
        pass

    def step(self, rgb, index, at_s, view=None):
        with self.lock:
            self.steps.append((view or "body", index, round(at_s, 4)))
        return {"frame": index, "atS": round(at_s, 4),
                "keypoints": {"nose": [1.0, 2.0, 0.9]}}


class StubDecoder:
    """Decoder's surface: scripted frames per URL, sender-timed."""

    made: list["StubDecoder"] = []
    scripts: dict[str, list[tuple[int, float]]] = {}
    epochs: dict[str, float] = {}

    def __init__(self, url, telemetry, **kwargs):
        self.url = url
        self.kwargs = kwargs
        self.timing = "ntp"
        self.epoch = None
        self.stopped = False
        StubDecoder.made.append(self)

    def frames(self):
        yuv = np.full((HEIGHT * 3 // 2, WIDTH), 128, np.uint8)
        for index, at_s in StubDecoder.scripts.get(self.url, []):
            self.epoch = StubDecoder.epochs[self.url]
            yield index, at_s, yuv
            time.sleep(0.05)  # the workers keep up, as a paced stream lets them

    def stop(self):
        self.stopped = True


class FakeLink:
    def __init__(self, fields):
        self.fields = fields
        self.sent: list[dict] = []
        self.connected = False
        self.ready = {}

    def send(self, message):
        self.sent.append(message)

    def stop(self, last_at_s):
        return {}


def session_args(tmp_path, **overrides):
    args = argparse.Namespace(
        stream=BODY_URL, pose="gpu", device="cpu", track="", pose_fps=6.0,
        analysis_socket="", sink_dir=str(tmp_path), run="", capture_bucket="",
        post_url="", post_interval_s=1.0, audio=False, duration=0.0,
        input_lost_after=0.0, overlay_publish="", overlay_record=False,
        face_stream=FACE_URL, audio_stream=FACE_URL)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def run_session(monkeypatch, tmp_path, **overrides):
    pose = StubPose()
    links = []

    def open_link(path, on_message, on_error, **fields):
        link = FakeLink(fields)
        links.append(link)
        return link

    monkeypatch.setattr(producer, "acquire_gpu_pose", lambda args, telemetry: pose)
    monkeypatch.setattr(producer, "Decoder", StubDecoder)
    monkeypatch.setattr(producer, "open_link", open_link)
    StubDecoder.made = []
    StubDecoder.scripts = {BODY_URL: [(0, 0.0), (1, 1 / 30), (2, 2 / 30)],
                           FACE_URL: [(0, 0.0), (10, 10 / 30)]}
    StubDecoder.epochs = {BODY_URL: 1_000.0, FACE_URL: 1_002.0}
    session = producer.Session(session_args(tmp_path, **overrides), Telemetry())
    session.run()
    return session, pose, links[0]


def test_two_views_are_decoded_posed_and_reported(monkeypatch, tmp_path):
    session, pose, link = run_session(monkeypatch, tmp_path)
    assert session.face_stream == FACE_URL and session.audio_stream == FACE_URL
    assert link.fields["views"] == ["body", "face"]
    assert link.fields["facePoseFps"] == producer.FACE_POSE_FPS
    # Two decoders: the face's waits for its track rather than failing.
    by_url = {d.url: d for d in StubDecoder.made}
    assert set(by_url) == {BODY_URL, FACE_URL}
    assert by_url[FACE_URL].kwargs["wait_for_track"] is True
    assert not by_url[BODY_URL].kwargs.get("wait_for_track")
    assert by_url[BODY_URL].stopped and by_url[FACE_URL].stopped
    # The body's frame 0 was posed as the body, the face's frames as the
    # face view, both through the one pose.
    assert ("body", 0, 0.0) in pose.steps
    assert ("face", 0, 0.0) in pose.steps and ("face", 10, round(10 / 30, 4)) in pose.steps
    # The face rows went out as facePose with the body's moment: the face
    # timeline starts 2 s after the body's on the senders' clock.
    face_rows = [m for m in link.sent if m["kind"] == "facePose"]
    assert [row["frame"] for row in face_rows] == [0, 10]
    assert face_rows[0]["atS"] == 0.0 and face_rows[1]["atS"] == round(10 / 30, 4)
    # A face row posed before the body's first frame placed the body's
    # clock has no body moment yet (protocol.md: null until lined up);
    # from then on it is the face moment shifted by the epochs' gap.
    placed = [row for row in face_rows if row["bodyAtS"] is not None]
    assert placed and all(row["bodyAtS"] == round(row["atS"] + 2.0, 4) for row in placed)
    assert face_rows[0]["keypoints"] == {"nose": [1.0, 2.0, 0.9]}
    assert face_rows[0]["frameSize"] == [WIDTH, HEIGHT]
    assert any(m["kind"] == "pose" for m in link.sent)
    # /statz's views.
    views = session.views()
    assert views["body"] == {"timing": "ntp", "epoch": 1_000.0, "url": BODY_URL}
    assert views["face"] == {"timing": "ntp", "epoch": 1_002.0, "url": FACE_URL}
    assert views["sync"] == {"timing": "ntp", "skewMs": 2000}
    assert session.telemetry.snapshot()["counters"]["faceFramesIn"] == 2


def test_one_view_without_a_face_stream(monkeypatch, tmp_path):
    session, pose, link = run_session(monkeypatch, tmp_path, face_stream="",
                                      audio_stream="")
    assert session.face_stream == "" and session.sync is None
    assert session.audio_stream == BODY_URL
    assert link.fields["views"] == ["body"] and link.fields["facePoseFps"] is None
    assert [d.url for d in StubDecoder.made] == [BODY_URL]
    assert all(view == "body" for view, _, _ in pose.steps)
    assert not any(m["kind"] == "facePose" for m in link.sent)
    assert set(session.views()) == {"body"}


def test_the_face_stream_needs_the_gpu_pose(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(producer, "SideloadPose", lambda path, telemetry: StubPose())
    monkeypatch.setattr(producer, "Decoder", StubDecoder)
    monkeypatch.setattr(producer, "open_link",
                        lambda path, on_message, on_error, **fields: FakeLink(fields))
    StubDecoder.made = []
    StubDecoder.scripts = {BODY_URL: [(0, 0.0)]}
    StubDecoder.epochs = {BODY_URL: 1_000.0}
    session = producer.Session(
        session_args(tmp_path, pose="sideload", track=str(tmp_path)), Telemetry())
    assert session.face_stream == "" and session.sync is None
    assert "needs --pose gpu" in capsys.readouterr().out
    session.run()
    assert [d.url for d in StubDecoder.made] == [BODY_URL]
