"""CUDA graphs for the fixed-shape forwards, with an eager twin.

Both models the producer runs have fixed input shapes once the frame has
been preprocessed (a 640x640 detector input, a 1024x768 pose crop), so
each forward plus its post-processing can be captured once at boot and
replayed per frame as a single graph launch. In eager mode the same work
is a thousand-odd small kernel launches for Sapiens2-1B and several hundred
for RT-DETRv4-X, and on a Confidential Computing GPU each launch crosses
an encrypted boundary: the forwards were launch-bound, the GPU idle
between launches. The image ships no triton (workload/Dockerfile drops the
wheel on purpose), so `torch.compile` is not an option; the graphs are
captured by hand with `torch.cuda.CUDAGraph`.

Contract for a captured function: static shapes, no host syncs
(`.item()`, `.cpu()`, data-dependent Python branches), and no tensor built
from Python values inside it (that is a pageable host-to-device copy, which
capture rejects); constants are created before capture and closed over.
`Eager` runs the same function directly: the fallback when capture fails
or is disabled, and the CPU path the tests exercise.
"""

from __future__ import annotations

import os
import time


class Eager:
    """The replay interface, running the function directly."""

    graphed = False

    def __init__(self, fn, static_inputs=()):
        self.fn = fn
        self.static_inputs = tuple(static_inputs)

    def replay(self, *inputs):
        return self.fn(*inputs)


class Graphed:
    """`fn` captured once over `static_inputs`, replayed per call.

    `replay` copies each input into its static buffer and launches the
    graph on the current stream; the returned outputs are the graph's own
    buffers and are valid until the next replay, so a caller reads them
    (a `.cpu()` on the current stream orders after the replay) before
    calling again.
    """

    graphed = True

    def __init__(self, fn, static_inputs, warmup: int = 3):
        import torch

        self.fn = fn
        self.static_inputs = tuple(static_inputs)
        # Warm up on a side stream: cuBLAS/cuDNN pick algorithms and grow
        # their workspaces here rather than during capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn(*self.static_inputs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_outputs = fn(*self.static_inputs)
        torch.cuda.synchronize()

    def replay(self, *inputs):
        for static, value in zip(self.static_inputs, inputs, strict=True):
            static.copy_(value)
        self.graph.replay()
        return self.static_outputs


def enabled(device: str) -> bool:
    """Graphs run on CUDA unless POSE_CUDA_GRAPHS=0 (a local debugging knob;
    the enclave's launch policy does not let it be set there)."""
    if os.environ.get("POSE_CUDA_GRAPHS", "1") == "0":
        return False
    return str(device).startswith("cuda")


def build(fn, static_inputs, *, name: str, device: str, telemetry=None,
          graphed_class=Graphed):
    """Capture `fn`, or fall back to running it eagerly.

    A capture failure is printed and counted (`graphCaptureFailed`) and the
    eager twin returned: the slot keeps serving at the old speed rather
    than refusing to boot over an optimisation.
    """
    if not enabled(device):
        print(f"{name}: graph disabled, running eagerly", flush=True)
        return Eager(fn, static_inputs)
    started = time.monotonic()
    try:
        graphed = graphed_class(fn, static_inputs)
    except Exception as error:  # noqa: BLE001 - any capture failure means eager
        print(f"{name}: graph capture failed: {error!r}", flush=True)
        if telemetry is not None:
            telemetry.count("graphCaptureFailed")
        return Eager(fn, static_inputs)
    print(f"{name}: graph captured in "
          f"{(time.monotonic() - started) * 1000:.0f}ms", flush=True)
    return graphed
