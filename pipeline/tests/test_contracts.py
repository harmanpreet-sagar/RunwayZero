"""Contract tests: every file the UI can load must match contracts/schema v1.0.

Run from the repo root:  python -m pytest -q
"""
import glob
import json
import os

import pytest

from pipeline.contracts import ROOT, WEB_DATA_DIR, kind_for, validate

DIRS = [os.path.join(ROOT, "contracts", "fixtures"), WEB_DATA_DIR]
FILES = [f for d in DIRS for f in sorted(glob.glob(os.path.join(d, "*.json")))]


def load(path):
    with open(path) as fh:
        return json.load(fh)


@pytest.mark.parametrize("path", FILES, ids=lambda p: os.path.relpath(p, ROOT))
def test_matches_schema(path):
    validate(kind_for(path), load(path))


@pytest.mark.parametrize("data_dir", [d for d in DIRS if os.path.exists(os.path.join(d, "scenarios.json"))])
def test_index_is_consistent(data_dir):
    index = load(os.path.join(data_dir, "scenarios.json"))
    scenario_ids = {s["id"] for s in index["scenarios"]}
    for entry in index["replays"]:
        path = os.path.join(data_dir, entry["file"])
        assert os.path.exists(path), f"scenarios.json lists missing file {entry['file']}"
        replay = load(path)
        assert entry["scenario_id"] in scenario_ids
        assert (replay["scenario_id"], replay["seed"], replay["dropout"]) == (entry["scenario_id"], entry["seed"], entry["dropout"])
        ids = {o["id"] for o in replay["objects"]}
        zones = {z["id"] for z in load(os.path.join(data_dir, "map.json"))["zones"]}
        prev_t = None
        for fr in replay["frames"]:
            assert {o["id"] for o in fr["objects"]} <= ids, f"t={fr['t']}: object not declared in replay.objects"
            assert all(a["detector"] in replay["detectors"] for a in fr["alerts"])
            assert all(a["zone_id"] in zones for a in fr["alerts"]), f"t={fr['t']}: alert for unknown zone"
            assert prev_t is None or fr["t"] > prev_t, "frames must be in increasing time order"
            prev_t = fr["t"]
