"""The live stream (egress.py) and its HUD card (hud_card.py): what a
destination must look like, the ffmpeg command with and without the
microphone and the card, the supervisor's spawn, respawn and stop with a
scripted process, the status that names the host and never the address,
and the card's bounded state and picture."""

from __future__ import annotations

import socket
import threading
import time

import pytest

import egress as egress_module
from egress import Egress, EgressError, EgressPublisher, parse_destination, resolve_public
from hud_card import CARD_SIZE, HudCard, sanitize_state

DESTINATION = "rtmps://live.example.com/app/sk_live_secret_key_1234"


def resolver_for(address: str):
    def resolver(host, port, type=None):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]
    return resolver


# -- the destination ------------------------------------------------------------------


def test_a_destination_is_rtmps_with_a_path_and_no_more():
    url, host, port = parse_destination(f"  {DESTINATION}  ")
    assert (url, host, port) == (DESTINATION, "live.example.com", 443)
    assert parse_destination("rtmps://live.example.com:1935/app/key")[2] == 1935
    for bad, reason in (
        (None, "bad-url"), ("", "bad-url"), ("x" * 3000, "bad-url"),
        ("rtmps://live.example.com/app/k ey", "bad-url"),
        ("rtmp://live.example.com/app/key", "not-rtmps"),
        ("https://live.example.com/app/key", "not-rtmps"),
        ("rtmps:///app/key", "bad-url"),
        ("rtmps://live.example.com", "bad-url"),
        ("rtmps://live.example.com/", "bad-url"),
        ("rtmps://[::1/app/key", "bad-url"),
    ):
        with pytest.raises(EgressError) as raised:
            parse_destination(bad)
        assert raised.value.reason == reason, bad
        assert raised.value.status == 400
        assert raised.value.body()["status"] == "failed"


def test_the_host_must_resolve_to_a_public_address_that_is_not_our_own():
    assert resolve_public("live.example.com", 443, resolver=resolver_for("8.8.8.8")) == "8.8.8.8"
    for address in ("10.1.2.3", "192.168.0.7", "127.0.0.1", "169.254.169.254", "224.0.0.1", "0.0.0.0"):
        with pytest.raises(EgressError) as raised:
            resolve_public("live.example.com", 443, resolver=resolver_for(address))
        assert raised.value.reason == "private-address", address
    with pytest.raises(EgressError) as raised:
        resolve_public("live.example.com", 443, own_ip="8.8.8.8", resolver=resolver_for("8.8.8.8"))
    assert raised.value.reason == "private-address"

    def failing(host, port, type=None):
        raise socket.gaierror("no such host")
    with pytest.raises(EgressError) as raised:
        resolve_public("nowhere.example", 443, resolver=failing)
    assert raised.value.reason == "unreachable"
    assert raised.value.status == 502
    with pytest.raises(EgressError):
        resolve_public("empty.example", 443, resolver=lambda *a, **k: [])


# -- the command ------------------------------------------------------------------------


def test_the_command_copies_the_view_and_adds_the_microphone_and_the_card_when_asked():
    plain = EgressPublisher(DESTINATION, "rtsp://127.0.0.1:8554/overlay").argv()
    assert plain[:2] == ["ffmpeg", "-nostdin"]
    assert plain[plain.index("-i") + 1] == "rtsp://127.0.0.1:8554/overlay"
    assert "-c:v" in plain and plain[plain.index("-c:v") + 1] == "copy"
    assert "-an" in plain
    assert "-filter_complex" not in plain
    assert plain[-3:] == ["-f", "flv", DESTINATION]

    with_audio = EgressPublisher(DESTINATION, "rtsp://127.0.0.1:8554/overlay",
                                 audio_url="rtsp://127.0.0.1:8554/cam").argv()
    inputs = [with_audio[i + 1] for i, arg in enumerate(with_audio) if arg == "-i"]
    assert inputs == ["rtsp://127.0.0.1:8554/overlay", "rtsp://127.0.0.1:8554/cam"]
    assert "-an" not in with_audio
    assert with_audio[with_audio.index("-c:a") + 1] == "aac"
    assert with_audio[with_audio.index("-map", with_audio.index("-c:a") - 3) + 1] == "1:a"

    card = HudCard()
    with_card = EgressPublisher(DESTINATION, "rtsp://127.0.0.1:8554/overlay",
                                audio_url="rtsp://127.0.0.1:8554/cam", hud=card).argv()
    inputs = [with_card[i + 1] for i, arg in enumerate(with_card) if arg == "-i"]
    assert inputs == ["rtsp://127.0.0.1:8554/overlay", "rtsp://127.0.0.1:8554/cam", "pipe:0"]
    assert with_card[with_card.index("-video_size") + 1] == f"{CARD_SIZE[0]}x{CARD_SIZE[1]}"
    assert with_card[with_card.index("-pixel_format") + 1] == "rgba"
    graph = with_card[with_card.index("-filter_complex") + 1]
    assert graph.startswith("[0:v][2:v]overlay=")
    assert "main_h-overlay_h" in graph
    assert with_card[with_card.index("-c:v") + 1] == "libx264"
    assert "copy" not in with_card
    nvenc = EgressPublisher(DESTINATION, "rtsp://127.0.0.1:8554/overlay", hud=card, encoder="nvenc").argv()
    assert nvenc[nvenc.index("-c:v") + 1] == "h264_nvenc"
    assert nvenc[nvenc.index("-filter_complex") + 1].startswith("[0:v][1:v]overlay=")


# -- the process ------------------------------------------------------------------------


class FakeProc:
    """A process that runs until told to exit; `communicate` blocks on it."""

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.returncode = None
        self.stdin = self if kwargs.get("stdin") is not None else None
        self.written = 0
        self._done = threading.Event()
        self.stderr_text = b""

    def poll(self):
        return self.returncode

    def exit(self, code: int, stderr: bytes = b""):
        self.returncode = code
        self.stderr_text = stderr
        self._done.set()

    def communicate(self, timeout=None):
        self._done.wait(timeout)
        return b"", self.stderr_text

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self.returncode

    def terminate(self):
        self.exit(-15)

    def kill(self):
        self.exit(-9)

    def write(self, data):
        self.written += len(data)

    def flush(self):
        pass

    def close(self):
        pass


def wait_for(predicate, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)


def test_the_supervisor_spawns_respawns_with_backoff_and_stops_on_demand():
    procs: list[FakeProc] = []
    clock = {"now": 100.0}

    def popen(argv, **kwargs):
        proc = FakeProc(argv, **kwargs)
        procs.append(proc)
        return proc

    logged = []
    publisher = EgressPublisher(DESTINATION, "rtsp://127.0.0.1:8554/overlay", popen=popen,
                                clock=lambda: clock["now"], sleep=lambda s: clock.__setitem__("now", clock["now"] + s),
                                log=lambda *a, **k: logged.append(a[0]))
    publisher.start()
    wait_for(lambda: len(procs) == 1)
    assert publisher.alive
    assert publisher.snapshot()["spawns"] == 1
    # ffmpeg dies (the production restarted): respawned after the first backoff, the error kept host-free.
    procs[0].exit(1, b"rtmps://live.example.com/app/sk_live_secret_key_1234: Connection refused")
    wait_for(lambda: len(procs) == 2)
    assert publisher.failures == 1
    assert "live.example.com" not in publisher.last_error
    assert "secret" not in publisher.last_error
    assert publisher.last_error.startswith("ffmpeg exited 1")
    assert all("secret" not in line for line in logged)
    # Stopped: the process is terminated, no respawn follows.
    publisher.stop()
    assert procs[1].returncode == -15
    assert not publisher.alive
    clock["now"] += 100
    time.sleep(0.05)
    assert len(procs) == 2


def test_the_card_is_fed_to_the_process_while_it_runs():
    procs: list[FakeProc] = []

    def popen(argv, **kwargs):
        proc = FakeProc(argv, **kwargs)
        procs.append(proc)
        return proc

    card = HudCard()
    publisher = EgressPublisher(DESTINATION, "rtsp://127.0.0.1:8554/overlay", hud=card, popen=popen,
                                sleep=lambda s: time.sleep(0.001), log=lambda *a, **k: None)
    publisher.start()
    wait_for(lambda: len(procs) == 1 and procs[0].written >= CARD_SIZE[0] * CARD_SIZE[1] * 4 * 2)
    assert procs[0].kwargs["stdin"] is not None
    publisher.stop()


def test_egress_starts_replaces_and_clears_a_stream_and_says_only_the_host():
    procs: list[FakeProc] = []

    def popen(argv, **kwargs):
        proc = FakeProc(argv, **kwargs)
        procs.append(proc)
        return proc

    counts = {}

    class Telemetry:
        def count(self, name, n=1):
            counts[name] = counts.get(name, 0) + n

    logged = []
    stream = Egress("rtsp://127.0.0.1:8554/overlay", "rtsp://127.0.0.1:8554/cam", own_ip="34.1.2.3",
                    hud_card=HudCard(), telemetry=Telemetry(), resolver=resolver_for("8.8.8.8"),
                    popen=popen, clock=lambda: 1_700_000_000.0, log=lambda *a, **k: logged.append(a[0]))
    assert stream.active is False
    assert stream.status() == {"active": False, "host": None, "since": None, "audio": False, "hud": False,
                               "error": None, "restarts": 0, "alive": False, "card": {"held": False, "ageS": None}}

    with pytest.raises(EgressError) as raised:
        stream.connect({"url": "rtmp://live.example.com/app/key"}, "sess-1")
    assert raised.value.reason == "not-rtmps"
    assert stream.active is False

    answer = stream.connect({"url": DESTINATION, "audio": True}, "sess-1")
    wait_for(lambda: len(procs) == 1)
    assert answer["status"] == "connected"
    assert answer["active"] is True and answer["host"] == "live.example.com"
    assert answer["since"] == 1_700_000_000.0 and answer["audio"] is True and answer["hud"] is True
    assert "sk_live" not in str(answer)
    assert stream.session_id == "sess-1"
    assert "-c:a" in procs[0].argv and "pipe:0" in procs[0].argv
    assert all("sk_live" not in line for line in logged), logged
    assert counts["egressStarted"] == 1

    # A second destination replaces the first; the card may be left off.
    stream.connect({"url": "rtmps://other.example.net/live/key2", "hud": False}, "sess-1")
    wait_for(lambda: len(procs) == 2)
    assert procs[0].returncode == -15
    assert stream.status()["host"] == "other.example.net"
    assert stream.status()["hud"] is False and stream.status()["audio"] is False
    assert "pipe:0" not in procs[1].argv and procs[1].argv[procs[1].argv.index("-c:v") + 1] == "copy"

    # The card's state comes from the trainer, bounded.
    state = stream.set_hud_state({"tiles": [{"key": "clench", "label": "Clench rate", "value": "36", "unit": "/min"}]})
    assert state["tiles"][0]["value"] == "36"
    assert stream.status()["card"]["held"] is True

    assert stream.clear("the phone asked") is True
    assert procs[1].returncode == -15
    assert stream.active is False and stream.status()["host"] is None
    assert stream.clear("again") is False
    assert counts["egressStopped"] == 1
    with pytest.raises(ValueError):
        Egress("rtsp://127.0.0.1:8554/overlay", None).set_hud_state({})


# -- the card ---------------------------------------------------------------------------


def test_the_card_bounds_the_trainers_words_and_paints_them():
    state = sanitize_state({
        "tiles": [{"key": "clench", "label": "Clench rate", "value": "36", "unit": "/min", "state": "live"},
                  {"key": "vocal", "label": "Vocalizations", "value": "8", "unit": "/min", "detail": "+14 dB", "state": "live"},
                  {"key": "heart", "label": "Heart rate", "value": "\u2014", "state": "none"},
                  {"key": "posture", "label": "Posture", "value": "Hips lifting", "detail": "Bucking", "state": "live"},
                  {"key": "extra", "label": "x" * 100, "value": "y" * 100}],
        "unit": {"name": "MK-312BT", "detail": "Stroke", "tone": "live", "level": 35, "max": 70},
        "fans": {"watching": 12, "controlling": 3},
        "stale": False,
        "whatever": "ignored",
    })
    assert len(state["tiles"]) == 4, "at most four tiles"
    assert state["tiles"][1]["detail"] == "+14 dB"
    assert state["unit"] == {"name": "MK-312BT", "detail": "Stroke", "tone": "live", "level": 35, "max": 70}
    assert state["fans"] == {"watching": 12, "controlling": 3}
    long = sanitize_state({"tiles": [{"label": "x" * 100, "value": "y" * 100, "unit": "z" * 20}]})
    assert len(long["tiles"][0]["label"]) == 24 and len(long["tiles"][0]["value"]) == 40 and len(long["tiles"][0]["unit"]) == 8
    assert sanitize_state({"unit": {"level": True, "max": 0}})["unit"] == {"name": "", "detail": "", "tone": "neutral", "level": None, "max": None}
    assert sanitize_state({})["tiles"] == [] and sanitize_state({})["unit"] is None
    with pytest.raises(ValueError):
        sanitize_state(["not", "an", "object"])

    clock = {"now": 10.0}
    card = HudCard(clock=lambda: clock["now"])
    empty = card.render()
    assert len(empty) == CARD_SIZE[0] * CARD_SIZE[1] * 4
    card.set_state({
        "tiles": state["tiles"], "unit": state["unit"], "fans": state["fans"],
    })
    painted = card.render()
    assert len(painted) == len(empty)
    assert painted != empty, "the words are drawn"
    assert card.describe() == {"held": True, "ageS": 0.0, "tiles": 4}
    # A state the trainer stopped pushing goes back to the empty card.
    clock["now"] += egress_module.STOP_WAIT_S + 20
    assert card.render() == empty
    calibrating = HudCard()
    calibrating.set_state({"tiles": [{"key": "clench", "label": "Clench rate", "state": "calibrating"}], "stale": True})
    assert len(calibrating.render()) == len(empty)
