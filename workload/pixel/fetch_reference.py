"""Fetch production captures - the reference pins by default - for local work.

    python fetch_reference.py                 # every pin in tracks.REFERENCE_RUNS
    python fetch_reference.py session-file-2  # any named run
    python fetch_reference.py --force ...     # re-pull an already fetched run

Each run lands under /tmp/pose-workload/production/<run>/ with the files the
producer uploaded when the session ended (poses.jsonl, onsets.jsonl,
events.jsonl, payloads.jsonl, posts.jsonl, summary.json). A run that is
already fetched is left alone unless --force. gcloud does the copying, so
the caller's gcloud auth is the credential.
"""

from __future__ import annotations

import argparse
import json

import tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*",
                        help="run names under gs://<bucket>/runs/; default "
                             "the reference pins")
    parser.add_argument("--bucket", default=tracks.BUCKET)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runs = args.runs or sorted(tracks.REFERENCE_RUNS.values())
    if not runs:
        print("no reference runs pinned yet and none named")
        return 1
    for run in runs:
        target = tracks.fetch(run, bucket=args.bucket, force=args.force)
        rows = tracks.load_rows(run)
        posed = sum(1 for row in rows if row.get("keypoints"))
        summary = target / "summary.json"
        counters = {}
        if summary.exists():
            counters = json.loads(summary.read_text()).get("counters") or {}
        # The summary's counters are the instance's, not the session's: a
        # second session on the same instance inherits the first one's
        # totals. Absent means zero.
        print(f"{run}: {target}  rows={len(rows)} posed={posed} "
              f"poseDropped={counters.get('poseDropped', 0)} "
              f"descriptorsDropped={counters.get('descriptorsDropped', 0)} "
              f"(instance totals) cadence={tracks.cadence_of(rows):g}fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
