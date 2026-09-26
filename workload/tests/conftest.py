"""Import paths for the workload's flat module layout.

The modules under test import each other by bare name (`import live_pose`,
`from producer import ...`), so the three source directories go on sys.path
- `workload/audio`, `workload/pixel` and `workload/producer`, in that order,
right after the tests directory itself (which pytest puts first for the
test-to-test imports such as `from test_tee_mode import request`).
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKLOAD = Path(__file__).resolve().parents[1]
TESTS = WORKLOAD / "tests"

for entry in (str(WORKLOAD / "producer"), str(WORKLOAD / "pixel"),
              str(WORKLOAD / "audio"), str(TESTS)):
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
