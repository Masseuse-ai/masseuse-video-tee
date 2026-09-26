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
