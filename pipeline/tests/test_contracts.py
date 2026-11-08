"""Contract tests: every file the UI can load must match contracts/schema v1.1.

Run from the repo root:  python -m pytest -q
"""
import glob
import json
import os

import pytest

from pipeline.contracts import (ROOT, WEB_DATA_DIR, brier, kind_for, p_safe_from, points, snap_confidence,
                                validate)

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


@pytest.mark.parametrize("data_dir", [d for d in DIRS if os.path.exists(os.path.join(d, "puzzles.json"))])
def test_puzzles_are_consistent(data_dir):
    index = load(os.path.join(data_dir, "puzzles.json"))
    zones = {z["id"] for z in load(os.path.join(data_dir, "map.json"))["zones"]}
    seen = set()
    for entry in index["puzzles"]:
        assert entry["id"] not in seen, f"duplicate puzzle id {entry['id']}"
        seen.add(entry["id"])
        path = os.path.join(data_dir, entry["file"])
        assert os.path.exists(path), f"puzzles.json lists missing file {entry['file']}"
        p = load(path)
        tag = p["id"]
        assert (p["id"], p["difficulty"]) == (entry["id"], entry["difficulty"]), tag
        assert p.get("is_mock", False) == index.get("is_mock", False), f"{tag}: mock and real files mixed"

        types = {o["id"]: o["type"] for o in p["objects"]}
        assert types.get(p["vehicle_id"]) == "vehicle", f"{tag}: vehicle_id is not a declared vehicle"
        assert types.get(p["aircraft_id"]) == "aircraft", f"{tag}: aircraft_id is not a declared aircraft"
        assert p["zone_id"] in zones, f"{tag}: zone {p['zone_id']} not in map.json"

        # pre ends exactly at the freeze; reveal is strictly after it; both in time order
        pre_t = [f["t"] for f in p["pre"]]
        rev_t = [f["t"] for f in p["reveal"]]
        assert pre_t == sorted(pre_t) and rev_t == sorted(rev_t), f"{tag}: frames out of order"
        assert pre_t[-1] == p["decision_t"], f"{tag}: last pre frame must be at decision_t"
        assert rev_t[0] > p["decision_t"], f"{tag}: reveal must start after decision_t"
        for f in p["pre"] + p["reveal"]:
            assert {o["id"] for o in f["objects"]} <= set(types), f"{tag} t={f['t']}: undeclared object"

        # The AI only forecasts the future, and its call matches its own numbers
        ai = p["ai"]
        assert p["aircraft_id"] in ai["paths"], f"{tag}: ai.paths must include the aircraft"
        for pts in ai["paths"].values():
            assert all(pt["t"] > p["decision_t"] for pt in pts), f"{tag}: ai path point at or before decision_t"
        for c in [ai] + list(ai.get("baselines", {}).values()):
            assert (c["call"], c["confidence"]) == snap_confidence(c["p_safe"]), f"{tag}: call/confidence don't match p_safe"

        o = p["outcome"]
        assert o["conflict"] == (o["min_sep_m"] < o["threshold_m"]), f"{tag}: conflict flag disagrees with min_sep_m"
        assert o["correct_call"] == ("HOLD" if o["conflict"] else "CLEAR"), tag
        assert o["threshold_m"] == index["conflict_threshold_m"], f"{tag}: threshold differs from puzzles.json"


def test_scoring_matches_scoring_ts():
    # Same examples as the comment block in contracts/scoring.ts
    assert points(brier(p_safe_from("CLEAR", 0.95), conflict=False)) == 100
    assert points(brier(p_safe_from("CLEAR", 0.95), conflict=True)) == 10
    assert points(brier(p_safe_from("HOLD", 0.5), conflict=True)) == 75
    assert points(brier(p_safe_from("HOLD", 0.75), conflict=True)) == 94
    assert snap_confidence(0.02) == ("HOLD", 0.95)
    assert snap_confidence(0.70) == ("CLEAR", 0.75)
    assert snap_confidence(0.55) == ("CLEAR", 0.5)
