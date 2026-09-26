"""The graph wrapper's contract, without a GPU.

`Graphed` itself needs CUDA (its parity with the eager twin is checked at
every boot by `live_pose.GpuPose.check_parity`); what is pinned here is the
interface the tracker relies on - the eager twin, the fallback when capture
fails, and the knob - with the capture stubbed.
"""

from __future__ import annotations

import pytest

import gpu_graph
from telemetry import Telemetry


class _Buffer:
    """Stands in for a static device tensor: `copy_` is what replay uses."""

    def __init__(self, value=None):
        self.value = value

    def copy_(self, other):
        self.value = other


class _Captured:
    """A stand-in for `Graphed` that records the capture and replays by
    running the function over its static buffers, as the real one does."""

    graphed = True
    instances: list["_Captured"] = []

    def __init__(self, fn, static_inputs):
        self.fn = fn
        self.static_inputs = tuple(static_inputs)
        self.replays = 0
        self.static_outputs = fn(*self.static_inputs)
        _Captured.instances.append(self)

    def replay(self, *inputs):
        for static, value in zip(self.static_inputs, inputs, strict=True):
            static.copy_(value)
        self.replays += 1
        return self.static_outputs


class _Refuses:
    graphed = True

    def __init__(self, fn, static_inputs):
        raise RuntimeError("operation not permitted when stream is capturing")


def test_eager_runs_the_function_over_the_inputs_it_is_given():
    eager = gpu_graph.Eager(lambda a, b: a + b, (_Buffer(), _Buffer()))
    assert eager.graphed is False
    assert eager.replay(2, 3) == 5
    assert len(eager.static_inputs) == 2


def test_build_captures_on_cuda_and_replays_through_the_static_buffers(monkeypatch):
    monkeypatch.delenv("POSE_CUDA_GRAPHS", raising=False)
    _Captured.instances.clear()
    calls = []

    def fn(a, b):
        calls.append((a.value, b.value))
        return "out"

    a, b = _Buffer("a0"), _Buffer("b0")
    graph = gpu_graph.build(fn, (a, b), name="pose", device="cuda",
                            graphed_class=_Captured)
    assert graph.graphed and graph is _Captured.instances[0]
    assert calls == [("a0", "b0")]  # the capture ran the function once
    assert graph.replay("a1", "b1") == "out"
    assert (a.value, b.value) == ("a1", "b1")
    with pytest.raises(ValueError):
        graph.replay("only one")  # an input per static buffer, no more, no less


def test_a_failed_capture_falls_back_to_eager_and_is_counted(monkeypatch, capsys):
    monkeypatch.delenv("POSE_CUDA_GRAPHS", raising=False)
    telemetry = Telemetry()
    graph = gpu_graph.build(lambda x: x * 2, (_Buffer(),), name="detector",
                            device="cuda:0", telemetry=telemetry,
                            graphed_class=_Refuses)
    assert isinstance(graph, gpu_graph.Eager) and not graph.graphed
    assert graph.replay(21) == 42
    assert telemetry.snapshot()["counters"]["graphCaptureFailed"] == 1
    out = capsys.readouterr().out
    assert "detector: graph capture failed" in out
    assert "not permitted when stream is capturing" in out


def test_the_knob_and_a_cpu_device_run_eagerly_without_trying_to_capture(monkeypatch):
    monkeypatch.setenv("POSE_CUDA_GRAPHS", "0")
    graph = gpu_graph.build(lambda x: x, (_Buffer(),), name="pose",
                            device="cuda", graphed_class=_Refuses)
    assert isinstance(graph, gpu_graph.Eager)
    monkeypatch.delenv("POSE_CUDA_GRAPHS")
    assert gpu_graph.enabled("cuda") and gpu_graph.enabled("cuda:1")
    assert not gpu_graph.enabled("cpu") and not gpu_graph.enabled("mps")
    graph = gpu_graph.build(lambda x: x, (_Buffer(),), name="pose",
                            device="cpu", graphed_class=_Refuses)
    assert isinstance(graph, gpu_graph.Eager)
