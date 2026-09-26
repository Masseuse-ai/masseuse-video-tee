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
    """Decoder's surface: scripted frames per URL, sender-timed.

    A script entry is `(index, at_s)` or `(index, at_s, picture)`: frames
    with the same `picture` are the same buffer contents, as the decode's
    `fps=30` conform repeats a slow source's frames; without it every
    frame is its own picture, as a 30 fps camera's are.
    """

    made: list["StubDecoder"] = []
    scripts: dict[str, list[tuple]] = {}
    epochs: dict[str, float] = {}

    def __init__(self, url, telemetry, **kwargs):
        self.url = url
        self.kwargs = kwargs
        self.timing = "ntp"
        self.epoch = None
        self.paced = True  # a live stream: decided rows never block on it
        self.stopped = False
        StubDecoder.made.append(self)

    def frames(self):
        for entry in StubDecoder.scripts.get(self.url, []):
            index, at_s = entry[0], entry[1]
            picture = entry[2] if len(entry) > 2 else index
            # A fresh buffer per frame, as the decoder hands out.
            yuv = np.full((HEIGHT * 3 // 2, WIDTH), 128, np.uint8)
            yuv[0, 0] = picture % 256
            yuv[0, 1] = (picture // 256) % 256
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


def run_session(monkeypatch, tmp_path, scripts=None, epochs=None, **overrides):
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
    StubDecoder.scripts = scripts or {
        BODY_URL: [(0, 0.0), (1, 1 / 30), (2, 2 / 30)],
        FACE_URL: [(0, 0.0), (10, 10 / 30)]}
    StubDecoder.epochs = epochs or {BODY_URL: 1_000.0, FACE_URL: 1_002.0}
    session = producer.Session(session_args(tmp_path, **overrides), Telemetry())
    session.run()
    return session, pose, links[0]


def test_two_views_are_decoded_posed_and_reported(monkeypatch, tmp_path):
    session, pose, link = run_session(monkeypatch, tmp_path)
    assert session.face_stream == FACE_URL and session.audio_stream == FACE_URL
    assert link.fields["views"] == ["body", "face"]
    # The face view follows the body's cadence unless told otherwise.
    assert session.face_pose_fps == 6.0
    assert link.fields["poseFps"] == 6.0 and link.fields["facePoseFps"] == 6.0
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


def test_the_face_view_takes_its_own_cadence_when_given(monkeypatch, tmp_path):
    """--face-pose-fps 3 with the body at 6: the face's frames 0 and 10 are
    both on the 3 fps picker's slots (0, 10, 20, ...), and `hello` says
    which cadence each view runs at."""
    session, pose, link = run_session(monkeypatch, tmp_path, face_pose_fps=3.0)
    assert session.pose_fps == 6.0 and session.face_pose_fps == 3.0
    assert link.fields["poseFps"] == 6.0 and link.fields["facePoseFps"] == 3.0
    assert [(index, at) for view, index, at in pose.steps if view == "face"] == [
        (0, 0.0), (10, round(10 / 30, 4))]


def test_a_nine_fps_body_submits_the_picker_slots(monkeypatch, tmp_path):
    """pose_fps=9 on the 30 fps grid: slots 0, 3, 7, 10 go to the pose and
    the frames between do not; the face view at the same cadence picks the
    same slots of its own grid."""
    grid = [(i, i / 30) for i in range(12)]
    session, pose, link = run_session(
        monkeypatch, tmp_path, pose_fps=9.0,
        scripts={BODY_URL: grid, FACE_URL: grid})
    assert session.pose_fps == 9.0 and session.face_pose_fps == 9.0
    assert link.fields["poseFps"] == 9.0 and link.fields["facePoseFps"] == 9.0
    body = sorted(index for view, index, _ in pose.steps if view == "body")
    face = sorted(index for view, index, _ in pose.steps if view == "face")
    assert body == [0, 3, 7, 10] and face == [0, 3, 7, 10]
    # The pose rows the analysis sees are the picked slots, in order, at
    # their grid times.
    rows = [m for m in link.sent if m["kind"] == "pose"]
    assert [row["frame"] for row in rows] == [0, 3, 7, 10]
    assert [row["atS"] for row in rows] == [0.0, 0.1, round(7 / 30, 4), round(10 / 30, 4)]


def test_a_slow_upload_conformed_to_the_grid_is_posed_on_distinct_frames(monkeypatch, tmp_path):
    """A phone uploading 10 fps arrives as every frame three times on the
    30 fps grid (`fps=30`), here with one frame held a slot longer than
    its share. Both views' 9 fps picks are distinct pictures: a pick that
    lands on a repeat of the frame last posed moves to the next frame
    that differs, and each move is counted per view."""
    pictures = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6,
                7, 7, 7, 8, 8, 8, 9, 9]
    grid = [(i, i / 30, picture) for i, picture in enumerate(pictures)]
    session, pose, link = run_session(
        monkeypatch, tmp_path, pose_fps=9.0,
        scripts={BODY_URL: grid, FACE_URL: grid})
    for view in ("body", "face"):
        indices = sorted(index for name, index, _ in pose.steps if name == view)
        # Slot 3 shows picture 0 again (posed at slot 0): the pick moves to
        # slot 4, the first frame that differs; the rest are on their slots.
        assert indices == [0, 4, 7, 10, 13, 17, 20, 23, 27], view
        assert len({pictures[i] for i in indices}) == len(indices), view
    counters = session.telemetry.snapshot()["counters"]
    assert counters["poseRepeatDeferred"] == 1
    assert counters["facePoseRepeatDeferred"] == 1
    # The rows carry the frame actually posed, at its own grid time.
    rows = [m for m in link.sent if m["kind"] == "pose"]
    assert [row["frame"] for row in rows] == [0, 4, 7, 10, 13, 17, 20, 23, 27]
    assert rows[1]["atS"] == round(4 / 30, 4)


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


class BreakingLink(FakeLink):
    """A link whose sends fail once told to, and which comes back on
    `reconnect` after a scripted number of refusals (analysis_link.py's
    shape; the real thing is tested in test_analysis_link.py)."""

    connected = True

    def __init__(self, fields, refusals: int = 1):
        super().__init__(fields)
        self.broken = False
        self.dropped = 0
        self.reconnects = 0
        self.refusals = refusals
        self.attempts = 0
        self.on_error = None
        self.resumed_answer = True

    def send(self, message):
        if self.broken:
            self.dropped += 1
            return
        self.sent.append(message)

    def reconnect(self, connect_timeout_s=None):
        self.attempts += 1
        if self.attempts <= self.refusals:
            import analysis_link
            raise analysis_link.LinkError("analysis socket: refused (test)")
        self.broken = False
        self.reconnects += 1
        return {"protocol": 1, "resumed": self.resumed_answer}


def test_a_broken_analysis_link_is_reconnected_with_backoff_and_said_on_the_stream(monkeypatch, tmp_path):
    pose = StubPose()
    links: list[BreakingLink] = []

    def open_link(path, on_message, on_error, **fields):
        link = BreakingLink(fields, refusals=1)
        link.on_error = on_error
        links.append(link)
        return link

    monkeypatch.setattr(producer, "acquire_gpu_pose", lambda args, telemetry: pose)
    monkeypatch.setattr(producer, "Decoder", StubDecoder)
    monkeypatch.setattr(producer, "open_link", open_link)
    StubDecoder.made = []
    StubDecoder.scripts = {BODY_URL: [(0, 0.0), (1, 1 / 30), (2, 2 / 30)],
                           FACE_URL: [(0, 0.0), (10, 10 / 30)]}
    StubDecoder.epochs = {BODY_URL: 1_000.0, FACE_URL: 1_002.0}
    telemetry = Telemetry()
    queue, _flag = telemetry.subscribe()
    session = producer.Session(session_args(tmp_path), telemetry)
    link = links[0]
    # The hello names the session the analysis may be asked to resume
    # (the lease's id when there is a record; a fresh one otherwise).
    assert isinstance(link.fields["sessionId"], str) and len(link.fields["sessionId"]) >= 8
    # No waiting in the test: the backoff is the loop's own.
    session.stopping = threading.Event()
    waits: list[float] = []
    real_wait = session.stopping.wait

    def instant_wait(timeout=None):
        waits.append(timeout)
        return real_wait(0.001)

    session.stopping.wait = instant_wait  # type: ignore[method-assign]

    # The link breaks: the reader's word arrives through on_error.
    link.broken = True
    link.on_error("analysis send failed: BrokenPipeError(32, 'Broken pipe')")
    assert session._relink_thread is not None
    session._relink_thread.join(5.0)
    assert not link.broken and link.reconnects == 1 and link.attempts == 2
    assert waits[:2] == [1.0, 2.0], "1 s, then 2 s before the attempt that took"
    counters = telemetry.snapshot()["counters"]
    assert counters["analysisErrors"] == 1
    assert counters["analysisReconnectFailures"] == 1
    assert counters["analysisReconnects"] == 1
    assert "analysisRestarts" not in counters, "the analysis kept the session"
    # The session's stream (what the trainer reads) heard it: lost, lost
    # again with the refusal, then resumed.
    import json
    said = [json.loads(m) for m in queue if '"analysis"' in m]
    assert [(e["kind"], e["state"]) for e in said] == [
        ("analysis", "lost"), ("analysis", "lost"), ("analysis", "resumed")]
    assert said[1]["error"].startswith("analysis socket") and said[2]["attempt"] == 2
    # A second break while the first relink is still running does not start another.
    link.broken = True
    link.refusals = 10 ** 6
    link.attempts = 0
    link.on_error("analysis send failed: again")
    first = session._relink_thread
    link.on_error("analysis send failed: and again")
    assert session._relink_thread is first
    session.stopping.set()
    first.join(5.0)
    assert not first.is_alive()
    assert telemetry.snapshot()["counters"]["analysisErrors"] == 3


# -- the connector's camera as the face view ------------------------------------------

FACE_VIEW_URL = "rtsp://127.0.0.1:8554/face-ext"


class FakeFaceSource:
    def __init__(self, active: bool):
        self.active = active
        self.stream_url = FACE_VIEW_URL


def test_external_view_args_name_the_connectors_camera_only_while_one_is_attached():
    class External:
        stream_url = "rtsp://127.0.0.1:8554/ext"
        phone_stream_url = "rtsp://127.0.0.1:8554/cam"

    # With none attached the args are today's: no face_view_stream at all.
    args = argparse.Namespace(stream="rtsp://127.0.0.1:8554/ext")
    assert producer.external_view_args(args, External(), FakeFaceSource(False)) is True
    assert args.face_stream == "rtsp://127.0.0.1:8554/cam"
    assert args.audio_stream == "rtsp://127.0.0.1:8554/cam"
    assert not hasattr(args, "face_view_stream")
    args = argparse.Namespace(stream="rtsp://127.0.0.1:8554/ext")
    assert producer.external_view_args(args, External(), None) is True
    assert not hasattr(args, "face_view_stream")
    # Attached: the face view shown is the connector's; the face stream
    # (the keypoints', the microphone's) stays the phone's.
    args = argparse.Namespace(stream="rtsp://127.0.0.1:8554/ext")
    assert producer.external_view_args(args, External(), FakeFaceSource(True)) is True
    assert args.face_view_stream == FACE_VIEW_URL
    assert args.face_stream == "rtsp://127.0.0.1:8554/cam"
    assert args.audio_stream == "rtsp://127.0.0.1:8554/cam"
    # Not the external camera's stream: nothing is touched.
    args = argparse.Namespace(stream="rtsp://127.0.0.1:8554/cam")
    assert producer.external_view_args(args, External(), FakeFaceSource(True)) is False
    assert not hasattr(args, "face_view_stream")


def test_without_a_face_source_the_session_runs_exactly_two_decoders(monkeypatch, tmp_path):
    session, pose, link = run_session(monkeypatch, tmp_path)
    assert session.face_view_stream == "" and session.face_view_decoder is None
    assert {d.url for d in StubDecoder.made} == {BODY_URL, FACE_URL}
    assert link.fields["faceView"] == "phone"
    assert session.views()["faceViewSource"] == "phone"
    assert "faceView" not in session.views()
    # No post URL: the face frames would go nowhere, so none are asked for.
    assert link.fields["faceFrameIntervalS"] is None


def test_face_frames_are_asked_for_only_with_a_post_url_and_a_face_view(monkeypatch, tmp_path):
    """The hello's faceFrameIntervalS (analysis/protocol.md): the flag's
    value when the frames have somewhere to go, null otherwise."""
    class QuietPoster:
        timeout_s = 0.1

        def __init__(self, url, telemetry):
            pass

        def post(self, body):
            pass

        def post_face(self, body):
            pass

        def close(self, timeout_s=None):
            pass

    monkeypatch.setattr(producer, "Poster", QuietPoster)
    _, _, link = run_session(monkeypatch, tmp_path, post_url="https://t.example/api/pose-signals/slot/s/readings",
                             face_frame_interval_s=0.2)
    assert link.fields["faceFrameIntervalS"] == 0.2
    _, _, link = run_session(monkeypatch, tmp_path, post_url="https://t.example/api/pose-signals/slot/s/readings",
                             face_frame_interval_s=0.0)
    assert link.fields["faceFrameIntervalS"] is None
    _, _, link = run_session(monkeypatch, tmp_path, post_url="", face_frame_interval_s=0.2)
    assert link.fields["faceFrameIntervalS"] is None


def test_the_connectors_camera_is_decoded_for_the_inset_alone_and_the_face_pose_reads_the_phone(monkeypatch, tmp_path):
    scripts = {
        BODY_URL: [(0, 0.0), (1, 1 / 30), (2, 2 / 30)],
        FACE_URL: [(0, 0.0), (10, 10 / 30)],
        FACE_VIEW_URL: [(0, 0.0), (5, 5 / 30), (6, 6 / 30)],
    }
    session, pose, link = run_session(
        monkeypatch, tmp_path, scripts=scripts,
        epochs={BODY_URL: 1_000.0, FACE_URL: 1_002.0, FACE_VIEW_URL: 1_003.0},
        face_view_stream=FACE_VIEW_URL)
    assert session.face_view_stream == FACE_VIEW_URL and session.face_stream == FACE_URL
    # Three decoders: the connector's waits for its track like the phone's.
    by_url = {d.url: d for d in StubDecoder.made}
    assert set(by_url) == {BODY_URL, FACE_URL, FACE_VIEW_URL}
    assert by_url[FACE_VIEW_URL].kwargs["wait_for_track"] is True
    assert by_url[FACE_VIEW_URL].stopped
    # The face pose still reads the phone's frames, and only those: the
    # connector's frames were never posed.
    face_steps = [(index, at) for view, index, at in pose.steps if view == "face"]
    assert face_steps == [(0, 0.0), (10, round(10 / 30, 4))]
    assert all(view in ("body", "face") for view, _, _ in pose.steps)
    # The analysis was told which picture the face view is.
    assert link.fields["views"] == ["body", "face"] and link.fields["faceView"] == "connector"
    counters = session.telemetry.snapshot()["counters"]
    assert counters["faceFramesIn"] == 2 and counters["faceViewFramesIn"] == 3
    # The face clock is the connector's (the frames shown), not the phone's.
    views = session.views()
    assert views["faceView"] == {"timing": "ntp", "epoch": 1_003.0, "url": FACE_VIEW_URL}
    assert views["faceViewSource"] == "connector"
    assert views["sync"] == {"timing": "ntp", "skewMs": 3000}


def test_a_face_view_stream_that_is_the_phones_or_the_bodys_is_no_third_view(monkeypatch, tmp_path):
    session, _, link = run_session(monkeypatch, tmp_path, face_view_stream=FACE_URL)
    assert session.face_view_stream == "" and link.fields["faceView"] == "phone"
    assert {d.url for d in StubDecoder.made} == {BODY_URL, FACE_URL}
    session, _, _ = run_session(monkeypatch, tmp_path, face_view_stream=BODY_URL)
    assert session.face_view_stream == ""
