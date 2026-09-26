"""The decoder's reconnect loop: stoppable, and bounded when asked.

No ffmpeg here - `_spawn` is replaced with a stub process whose stdout
delivers whatever the test scripts, so the loop's decisions are tested on
their own: a `/stop` while the camera is gone must land (the orphaned slot
of 2026-09-04 kept its GPU because it did not), and `--input-lost-after`
must end the stream after that long without frames while frames that do
arrive reset the deadline. And the pinned geometry: a live stream decodes
at `pipe_size` of its probe with ffmpeg told to deliver exactly that, a
file at its own size.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

WORKLOAD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOAD / "producer"))

from producer import Decoder, pipe_size  # noqa: E402
from telemetry import Telemetry  # noqa: E402

WIDTH, HEIGHT = 4, 2
FRAME = bytes(range(WIDTH * HEIGHT * 3 // 2))


class _Stdout:
    def __init__(self, reads):
        self.reads = reads

    def read(self, _n):
        return self.reads.pop(0) if self.reads else b""

    def close(self):
        pass


class _Proc:
    """A stand-in ffmpeg: alive until its scripted reads run out."""

    def __init__(self, reads):
        self.stdout = _Stdout(list(reads))
        self.killed = False

    def poll(self):
        return None  # alive until the decoder reads its EOF and waits on it

    def wait(self):
        return 0

    def kill(self):
        self.killed = True


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


def _decoder(spawns, *, stopping=None, lost_after_s=0.0, on_sleep=None):
    """A Decoder whose successive `_spawn`s deliver `spawns[i]` reads each."""
    clock = _Clock()

    def sleep(s):
        clock.sleep(s)
        if on_sleep:
            on_sleep(decoder)

    decoder = Decoder("rtsp://127.0.0.1:8554/cam", Telemetry(),
                      stopping=stopping, lost_after_s=lost_after_s,
                      clock=clock, sleep=sleep)
    decoder.size = (WIDTH, HEIGHT)
    queue = [list(reads) for reads in spawns]
    decoder.spawned = 0

    def spawn():
        decoder.spawned += 1
        decoder.proc = _Proc(queue.pop(0) if queue else [])

    decoder._spawn = spawn
    return decoder, clock


def test_a_stop_lands_while_the_stream_is_gone():
    stopping = threading.Event()
    sleeps = {"n": 0}

    def on_sleep(_decoder):
        sleeps["n"] += 1
        if sleeps["n"] == 3:
            stopping.set()

    decoder, _clock = _decoder([[]] * 100, stopping=stopping,
                               on_sleep=on_sleep)
    frames = list(decoder.frames())
    assert frames == []
    # Three empty spawns, three reconnect sleeps, then the flag was seen at
    # the top of the loop rather than after another spawn.
    assert sleeps["n"] == 3
    assert decoder.spawned == 3
    assert decoder.telemetry.snapshot()["counters"]["reconnects"] == 3


def test_without_a_deadline_the_loop_reconnects_until_stopped():
    stopping = threading.Event()
    sleeps = {"n": 0}

    def on_sleep(_decoder):
        sleeps["n"] += 1
        if sleeps["n"] == 50:
            stopping.set()

    decoder, clock = _decoder([[]] * 100, stopping=stopping, on_sleep=on_sleep)
    assert list(decoder.frames()) == []
    assert sleeps["n"] == 50
    assert clock.now == 1050.0
    assert "inputLost" not in decoder.telemetry.snapshot()["counters"]


def test_the_stream_is_treated_as_ended_after_the_input_loss_deadline():
    decoder, clock = _decoder([[]] * 100, lost_after_s=30.0)
    assert list(decoder.frames()) == []
    # First empty read at t=1000 starts the clock; 30 sleeps of 1 s later
    # the 31st empty read is past the deadline.
    assert clock.now == 1030.0
    assert decoder.spawned == 31
    assert decoder.telemetry.snapshot()["counters"]["inputLost"] == 1


def test_frames_that_arrive_reset_the_input_loss_deadline():
    # 20 s dark, then two frames, then dark again: the second outage gets a
    # fresh 30 s rather than the 10 s left from the first.
    spawns = [[]] * 20 + [[FRAME, FRAME]] + [[]] * 100
    decoder, clock = _decoder(spawns, lost_after_s=30.0)
    frames = list(decoder.frames())
    assert [index for index, _at, _yuv in frames] == [0, 1]
    assert all(isinstance(yuv, np.ndarray) and yuv.shape == (HEIGHT * 3 // 2, WIDTH)
               for _i, _a, yuv in frames)
    # 20 sleeps before the frames, then 30 sleeps after the frame spawn ran
    # dry (its short read is the first empty one of the second outage).
    assert clock.now == 1000.0 + 20 + 30
    assert decoder.telemetry.snapshot()["counters"]["inputLost"] == 1


def test_a_file_that_ends_still_returns_at_once():
    decoder, clock = _decoder([[FRAME]], lost_after_s=30.0)
    decoder.url = "/mnt/clips/session.mp4"
    frames = list(decoder.frames())
    assert len(frames) == 1
    assert clock.now == 1000.0  # no reconnect sleep for a file
    assert decoder.spawned == 1


# -- the pinned geometry -------------------------------------------------------
#
# The iPad of 2026-09-08: probed at 360x640 while its encoder was ramping,
# decoded at 720x1280 once it had, every frame read as four. The pipe now
# carries pipe_size(probe) and ffmpeg is told to deliver exactly that.


def test_pipe_size_is_the_probed_aspect_at_the_long_side():
    assert pipe_size((360, 640)) == (720, 1280)   # the ramp-up frame
    assert pipe_size((640, 360)) == (1280, 720)   # landscape
    assert pipe_size((480, 640)) == (960, 1280)   # 3:4 stays 3:4
    assert pipe_size((720, 1280)) == (720, 1280)  # already there


def test_pipe_size_never_shrinks_a_larger_stream():
    assert pipe_size((1080, 1920)) == (1080, 1920)
    assert pipe_size((2160, 3840)) == (2160, 3840)


def test_pipe_size_is_even_on_both_sides():
    assert pipe_size((359, 641)) == (716, 1280)
    assert all(side % 2 == 0 for side in pipe_size((333, 777)))


def _stream(url, pipe):
    """A Decoder probed at 360x640 whose stub ffmpeg delivers one frame of
    `pipe` (what a real ffmpeg, told the pinned size, would write)."""
    decoder = Decoder(url, Telemetry(), realtime=True)
    decoder._probe = lambda: (360, 640)
    frame = bytes(pipe[0] * pipe[1] * 3 // 2)
    decoder._spawn = lambda: setattr(decoder, "proc", _Proc([frame]))
    return decoder


def test_a_live_stream_decodes_at_the_pinned_size_and_ffmpeg_is_told_so():
    decoder = _stream("rtsp://127.0.0.1:8554/cam", (720, 1280))
    _index, _at, yuv = next(decoder.frames())  # probes, spawns, one read
    assert decoder.probed == (360, 640)
    assert decoder.size == (720, 1280)
    assert yuv.shape == (1280 * 3 // 2, 720)  # one whole frame per read
    argv = decoder._argv()
    assert argv[argv.index("-vf") + 1] == (
        "fps=30,scale=720:1280:force_original_aspect_ratio=decrease:"
        "force_divisible_by=2,pad=720:1280:-1:-1")
    assert "-re" not in argv  # the camera paces a network stream


def test_a_file_keeps_its_own_geometry_and_only_the_cadence_filter():
    decoder = _stream("/mnt/clips/session.mp4", (360, 640))
    _index, _at, yuv = next(decoder.frames())
    assert decoder.size == (360, 640) == decoder.probed
    assert yuv.shape == (640 * 3 // 2, 360)
    argv = decoder._argv()
    assert argv[argv.index("-vf") + 1] == "fps=30"
    assert "-re" in argv  # -re stands in for the camera on a file


def test_probe_names_a_stream_without_a_video_track(monkeypatch):
    """A relay path finalised as audio-only (the publisher's video packets
    came after the gather timeout) is said plainly, not as an IndexError
    from ffprobe's empty stream list."""
    import json
    import subprocess

    import producer as producer_module
    import pytest

    class Done:
        def __init__(self, stdout):
            self.stdout = stdout
            self.returncode = 0

    monkeypatch.setattr(producer_module.subprocess, "run",
                        lambda *a, **k: Done(json.dumps({"programs": [], "streams": []})))
    decoder = Decoder("rtsp://127.0.0.1:8554/cam", Telemetry())
    with pytest.raises(RuntimeError, match="no video track on rtsp://127.0.0.1:8554/cam"):
        decoder._probe()

    monkeypatch.setattr(producer_module.subprocess, "run",
                        lambda *a, **k: Done(json.dumps({"streams": [{"width": 960, "height": 540}]})))
    assert decoder._probe() == (960, 540)
    assert subprocess is producer_module.subprocess


# -- a video track without frames yet ----------------------------------------
#
# The connector drops whole video frames while its tunnel is behind and
# resumes at the next keyframe; ffprobe against the relay then reports the
# track (the description names the codec) at 0x0. That is not a geometry
# to pin the pipe to - pipe_size divided by it and every production run of
# 2026-09-10's session ended on a ZeroDivisionError - but a wait.


def test_pipe_size_refuses_a_probe_without_geometry():
    import pytest

    with pytest.raises(ValueError, match="probe without a geometry: 0x0"):
        pipe_size((0, 0))


def _probe_result(monkeypatch, *payloads):
    """subprocess.run replaced by successive ffprobe outputs, the last one
    repeating."""
    import json

    import producer as producer_module

    class Done:
        def __init__(self, stdout):
            self.stdout = stdout
            self.returncode = 0

    queue = [json.dumps(payload) for payload in payloads]

    def run(*_a, **_k):
        return Done(queue.pop(0) if len(queue) > 1 else queue[0])

    monkeypatch.setattr(producer_module.subprocess, "run", run)


def test_probe_names_a_starved_track_rather_than_dividing_by_zero(monkeypatch):
    import pytest

    from producer import StarvedTrack

    _probe_result(monkeypatch, {"streams": [{"width": 0, "height": 0}]})
    decoder = Decoder("rtsp://127.0.0.1:8554/ext", Telemetry())
    with pytest.raises(StarvedTrack, match="carries no frames yet"):
        decoder._probe()

    _probe_result(monkeypatch, {"streams": [{"codec_type": "video"}]})  # no size keys
    with pytest.raises(StarvedTrack):
        decoder._probe()


def test_a_starved_track_is_waited_for_then_decoded(monkeypatch):
    """Two probes of 0x0, then the geometry: the decoder waits a second
    between them and pins the pipe to what the third one said."""
    from producer import STARVED_RETRY_S

    _probe_result(monkeypatch,
                  {"streams": [{"width": 0, "height": 0}]},
                  {"streams": [{"width": 0, "height": 0}]},
                  {"streams": [{"width": 1280, "height": 720}]})
    clock = _Clock()
    telemetry = Telemetry()
    decoder = Decoder("rtsp://127.0.0.1:8554/ext", telemetry,
                      clock=clock, sleep=clock.sleep)
    start = clock.now
    assert decoder._await_probe() == (1280, 720)
    assert clock.now - start == 2 * STARVED_RETRY_S
    assert telemetry.snapshot()["counters"]["starvedProbes"] == 2


def test_a_starved_track_is_given_up_on_after_the_bound(monkeypatch):
    import pytest

    from producer import STARVED_TRACK_MAX_S

    _probe_result(monkeypatch, {"streams": [{"width": 0, "height": 0}]})
    clock = _Clock()
    decoder = Decoder("rtsp://127.0.0.1:8554/ext", Telemetry(),
                      clock=clock, sleep=clock.sleep)
    start = clock.now
    with pytest.raises(RuntimeError, match="carries no frames yet after 30 s"):
        decoder._await_probe()
    assert STARVED_TRACK_MAX_S <= clock.now - start < STARVED_TRACK_MAX_S + 2


def test_a_primary_view_waits_for_an_absent_track_then_decodes(monkeypatch):
    """The fixed camera's path with no track for two probes (the relay
    pulling it again after the tunnel came back), then the geometry: the
    session is not lost to one bad probe, it waits TRACK_RETRY_S at a
    time and pins the pipe to what the third one said."""
    from producer import TRACK_RETRY_S

    _probe_result(monkeypatch, {"streams": []}, {"streams": []},
                  {"streams": [{"width": 1280, "height": 720}]})
    clock = _Clock()
    telemetry = Telemetry()
    decoder = Decoder("rtsp://127.0.0.1:8554/ext", telemetry,
                      clock=clock, sleep=clock.sleep)
    start = clock.now
    assert decoder._await_probe() == (1280, 720)
    assert clock.now - start == 2 * TRACK_RETRY_S
    assert telemetry.snapshot()["counters"]["trackWaits"] == 2


def test_a_primary_view_gives_up_on_an_absent_track_after_the_bound(monkeypatch):
    import pytest

    from producer import TRACK_ABSENT_MAX_S, TRACK_RETRY_S

    _probe_result(monkeypatch, {"streams": []})
    clock = _Clock()
    decoder = Decoder("rtsp://127.0.0.1:8554/ext", Telemetry(),
                      clock=clock, sleep=clock.sleep)
    start = clock.now
    with pytest.raises(RuntimeError,
                       match=r"no video track on rtsp://127.0.0.1:8554/ext after 15 s"):
        decoder._await_probe()
    assert TRACK_ABSENT_MAX_S <= clock.now - start < TRACK_ABSENT_MAX_S + TRACK_RETRY_S


def test_a_secondary_view_waits_out_a_starved_or_absent_track(monkeypatch):
    """With `wait_for_track` (the phone's camera beside a fixed one) a
    starved track is not given up on at the bound, and a track that is not
    there at all is waited for too, every TRACK_RETRY_S."""
    from producer import STARVED_TRACK_MAX_S, TRACK_RETRY_S

    starved = {"streams": [{"width": 0, "height": 0}]}
    absent = {"streams": []}
    _probe_result(monkeypatch, *([starved] * 40 + [absent] * 3
                                 + [{"streams": [{"width": 720, "height": 1280}]}]))
    clock = _Clock()
    telemetry = Telemetry()
    decoder = Decoder("rtsp://127.0.0.1:8554/cam", telemetry,
                      clock=clock, sleep=clock.sleep, wait_for_track=True)
    start = clock.now
    assert decoder._await_probe() == (720, 1280)
    assert clock.now - start > STARVED_TRACK_MAX_S
    counters = telemetry.snapshot()["counters"]
    assert counters["starvedProbes"] == 40 and counters["trackWaits"] == 3
    assert clock.now - start == 40 * 1.0 + 3 * TRACK_RETRY_S


def test_a_stop_while_starved_lands(monkeypatch):
    """A /stop during the wait ends frames() quietly, as one during a
    reconnect does."""
    _probe_result(monkeypatch, {"streams": [{"width": 0, "height": 0}]})
    clock = _Clock()
    stopping = threading.Event()
    decoder = Decoder("rtsp://127.0.0.1:8554/ext", Telemetry(),
                      clock=clock, sleep=clock.sleep, stopping=stopping)
    start = clock.now
    original_sleep = clock.sleep

    def sleep(s):
        original_sleep(s)
        if clock.now - start > 5:
            stopping.set()

    decoder.sleep = sleep
    assert list(decoder.frames()) == []
    assert 5 < clock.now - start < 30


# -- stream-reader records: the grid by sender time ---------------------------

T0 = 1_757_500_000.0  # a sender's clock, unix seconds
RTP0 = 4_000_000_000  # an RTP timestamp near the top of its range


def _record(*, seq=0, ntp_s=None, rtp=RTP0, keyframe=False):
    """One record as stream-reader writes it, in the two reads the decoder
    makes of it: the header, then the frame. `ntp_s` None leaves the
    sender's time out (the flag clear)."""
    from producer import RECORD, RECORD_KEYFRAME, RECORD_MAGIC, RECORD_NTP_VALID
    flags = (RECORD_NTP_VALID if ntp_s is not None else 0) | (
        RECORD_KEYFRAME if keyframe else 0)
    header = RECORD.pack(RECORD_MAGIC, 1, flags, 1, 0, seq,
                         int(round((ntp_s or 0.0) * 1e9)), rtp & 0xFFFFFFFF,
                         WIDTH, HEIGHT, len(FRAME))
    return [header, FRAME]


def _frame_of(k):
    """A frame whose bytes say which record it came from."""
    return bytes([k] * (WIDTH * HEIGHT * 3 // 2))


def _reader_decoder(spawns, *, stopping=None, wall_start=T0 + 0.3):
    """A Decoder reading records through a stub stream-reader. The host's
    clock starts 300 ms after the sender's first frame: a real path's
    latency, well within CLOCK_SKEW_MAX_S."""
    clock = _Clock()
    wall = _Clock()
    wall.now = wall_start
    decoder = Decoder("rtsp://127.0.0.1:8554/cam", Telemetry(),
                      stopping=stopping, clock=clock, sleep=clock.sleep,
                      reader_bin=sys.executable, wall=wall)
    assert decoder.use_reader and decoder.timing == "ntp"
    decoder.size = (WIDTH, HEIGHT)
    queue = [list(reads) for reads in spawns]
    decoder.spawned = 0

    def spawn():
        decoder.spawned += 1
        decoder.proc = _Proc(queue.pop(0) if queue else [])

    decoder._spawn = spawn
    return decoder, wall


def _records(*specs):
    """Reads for a spawn: one record per (seq, ntp_s, rtp) spec, its frame
    bytes naming its seq."""
    reads = []
    for seq, ntp_s, rtp in specs:
        header, _frame = _record(seq=seq, ntp_s=ntp_s, rtp=rtp)
        reads += [header, _frame_of(seq)]
    return reads


def _collect(decoder, stopping, spawns_expected):
    """Every (index, at_s, frame tag) the decoder yields before the last
    scripted spawn's reads are gone."""
    out = []
    for index, at_s, yuv in decoder.frames():
        out.append((index, round(at_s, 4), int(yuv[0, 0])))
        if decoder.spawned >= spawns_expected and not decoder.proc.stdout.reads:
            stopping.set()
    return out


def test_reader_records_are_laid_on_the_grid_by_the_senders_time():
    stopping = threading.Event()
    decoder, _wall = _reader_decoder([_records(
        (0, T0, RTP0), (1, T0 + 1 / 30, RTP0 + 3000), (2, T0 + 2 / 30, RTP0 + 6000))],
        stopping=stopping)
    out = _collect(decoder, stopping, 1)
    assert out == [(0, 0.0, 0), (1, round(1 / 30, 4), 1), (2, round(2 / 30, 4), 2)]
    assert decoder.timing == "ntp"
    assert decoder.epoch == T0
    counters = decoder.telemetry.snapshot()["counters"]
    assert "frameEarly" not in counters and "frameGap" not in counters


def test_the_reader_is_asked_for_the_pinned_size():
    decoder, _wall = _reader_decoder([])
    argv = decoder._reader_argv()
    assert argv[0] == sys.executable
    assert argv[argv.index("-url") + 1] == "rtsp://127.0.0.1:8554/cam"
    assert argv[argv.index("-width") + 1] == str(WIDTH)
    assert argv[argv.index("-height") + 1] == str(HEIGHT)


def test_without_the_binary_a_live_stream_is_decoded_by_ffmpeg_as_before():
    decoder = Decoder("rtsp://127.0.0.1:8554/cam", Telemetry(),
                      reader_bin="/nonexistent/stream-reader")
    assert not decoder.use_reader and decoder.timing == "arrival"
    assert Decoder("/mnt/clips/session.mp4", Telemetry(),
                   reader_bin=sys.executable).timing == "file"


def test_a_slow_sender_repeats_the_last_frame_and_a_long_gap_jumps():
    stopping = threading.Event()
    decoder, _wall = _reader_decoder([_records(
        (0, T0, RTP0),
        (1, T0 + 2 / 30, RTP0 + 6000),        # one slot skipped: repeated
        (2, T0 + 2 / 30 + 1.0, RTP0 + 6000 + 90000))],  # a second: a jump
        stopping=stopping)
    out = _collect(decoder, stopping, 1)
    assert [(i, tag) for i, _at, tag in out] == [(0, 0), (1, 0), (2, 1), (32, 2)]
    assert out[3][1] == round(32 / 30, 4)
    counters = decoder.telemetry.snapshot()["counters"]
    assert counters["frameDup"] == 1 and counters["frameGap"] == 1


def test_a_frame_timed_at_or_before_the_last_slot_is_dropped():
    stopping = threading.Event()
    decoder, _wall = _reader_decoder([_records(
        (0, T0, RTP0),
        (1, T0 + 1 / 30, RTP0 + 3000),
        (2, T0 + 1 / 30 + 0.004, RTP0 + 3360),   # the same slot again
        (3, T0 + 0.5 / 30, RTP0 + 1500),         # before it
        (4, T0 + 2 / 30, RTP0 + 6000))],
        stopping=stopping)
    out = _collect(decoder, stopping, 1)
    assert [(i, tag) for i, _at, tag in out] == [(0, 0), (1, 1), (2, 4)]
    assert decoder.telemetry.snapshot()["counters"]["frameEarly"] == 2


def test_a_sender_clock_far_from_the_hosts_times_the_view_by_arrival():
    stopping = threading.Event()
    # The sender says it is 100 s ahead of this host: not believed. The
    # grid then follows the RTP timestamps from the arrival epoch.
    decoder, wall = _reader_decoder([_records(
        (0, T0 + 100, RTP0), (1, T0 + 100 + 1 / 30, RTP0 + 3000),
        (2, T0 + 100 + 3 / 30, RTP0 + 9000))],
        stopping=stopping)
    out = _collect(decoder, stopping, 1)
    assert [(i, tag) for i, _at, tag in out] == [(0, 0), (1, 1), (2, 1), (3, 2)]
    assert decoder.timing == "arrival"
    assert decoder.epoch == wall.now  # this host's clock at the first frame
    counters = decoder.telemetry.snapshot()["counters"]
    assert counters["clockSkew"] == 1 and counters["arrivalTimed"] == 1


def test_records_without_a_senders_time_run_on_their_rtp_timestamps():
    stopping = threading.Event()
    # No sender report: the flag is clear. The timestamps wrap around the
    # top of their range on the way.
    decoder, wall = _reader_decoder([_records(
        (0, None, 0xFFFFFFFF - 1500), (1, None, 1500), (2, None, 4500))],
        stopping=stopping)
    out = _collect(decoder, stopping, 1)
    assert [(i, tag) for i, _at, tag in out] == [(0, 0), (1, 1), (2, 2)]
    assert decoder.timing == "arrival"
    assert decoder.epoch == wall.now


def test_the_epoch_survives_a_reconnect_and_the_grid_keeps_its_place():
    stopping = threading.Event()
    decoder, _wall = _reader_decoder([
        _records((0, T0, RTP0), (1, T0 + 1 / 30, RTP0 + 3000)),
        _records((0, T0 + 2.0, 12345), (1, T0 + 2.0 + 1 / 30, 12345 + 3000)),
    ], stopping=stopping)
    out = _collect(decoder, stopping, 2)
    assert [(i, tag) for i, _at, tag in out] == [(0, 0), (1, 1), (60, 0), (61, 1)]
    assert decoder.epoch == T0
    counters = decoder.telemetry.snapshot()["counters"]
    assert counters["reconnects"] == 1 and counters["frameGap"] == 1


def test_a_record_out_of_step_restarts_the_reader():
    stopping = threading.Event()
    good = _records((0, T0, RTP0))
    bad_header, _frame = _record(seq=1, ntp_s=T0 + 1 / 30)
    bad_header = b"JUNK" + bad_header[4:]
    decoder, _wall = _reader_decoder([
        good + [bad_header, _frame_of(1)],
        _records((0, T0 + 1 / 30, RTP0 + 3000)),
    ], stopping=stopping)
    out = _collect(decoder, stopping, 2)
    assert [(i, tag) for i, _at, tag in out] == [(0, 0), (1, 0)]
    counters = decoder.telemetry.snapshot()["counters"]
    assert counters["readerDesync"] == 1 and counters["reconnects"] == 1
