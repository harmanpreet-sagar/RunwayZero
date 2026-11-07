"""Generate MOCK contract files so the front end can be built before real data exists.

Fictional airport: one east-west runway, crossing taxiway C, parallel taxiway A.
A landing aircraft (AC1) and a vehicle crossing at C (V1) meet in zone Z1 around t=27s.
Every file has is_mock=true. Numbers here are made up and must never be reported.

    python -m pipeline.make_fixtures     # writes contracts/fixtures/
"""
import math
import os

import numpy as np

from pipeline.contracts import ROOT, SCHEMA_VERSION, replay_filename, write

OUT = os.path.join(ROOT, "contracts", "fixtures")
DT, DURATION, HORIZON = 1.0, 60, 5.0
ZONE = {"id": "Z1", "name": "RWY 9/27 x TWY C", "polygon": [[-60, -60], [60, -60], [60, 60], [-60, 60]]}


def in_zone(x, y):
    return -60 <= x <= 60 and -60 <= y <= 60


def zone_dist(x, y):
    """Distance from a point to the Z1 square (0 inside)."""
    dx, dy = max(abs(x) - 60, 0), max(abs(y) - 60, 0)
    return math.hypot(dx, dy)


def true_state(oid, t):
    if oid == "AC1":   # landing roll along +x, decelerating
        v = max(70 - 1.2 * t, 8)
        x = -1500 + 70 * t - 0.6 * t * t if 70 - 1.2 * t > 8 else -1500 + 70 * 51.67 - 0.6 * 51.67 ** 2 + 8 * (t - 51.67)
        return x, 0.0, 90.0, v
    if oid == "V1":    # vehicle crossing north on taxiway C
        return 0.0, -224 + 8 * t, 0.0, 8.0
    if oid == "AC2":   # taxiing east on parallel taxiway A, never conflicts
        return -800 + 10 * t, -200.0, 90.0, 10.0
    raise KeyError(oid)


OBJECTS = [
    {"id": "AC1", "type": "aircraft", "label": "MOCK101 (landing)"},
    {"id": "V1", "type": "vehicle", "label": "Rescue 1 (mock)"},
    {"id": "AC2", "type": "aircraft", "label": "MOCK202 (taxi)"},
]


def build_replay(scenario_id, seed, dropout):
    rng = np.random.default_rng(seed)
    lost_window = (15, 27) if scenario_id == "tracking_dropout" else None
    delay = 3 if scenario_id == "delayed_data" else 0
    last = {}   # id -> (t_seen, x, y)
    hist = {}   # id -> list of (t, x, y) observed
    frames, truth_start, truth_end, first_alert = [], None, None, {}

    times = [i * DT for i in range(int(DURATION / DT) + 1)]
    for t in times:
        fobjs, conflict_ids = [], []
        for o in OBJECTS:
            oid = o["id"]
            x, y, hdg, spd = true_state(oid, t)
            if in_zone(x, y):
                conflict_ids.append(oid)
            dropped = (lost_window and oid == "V1" and lost_window[0] <= t < lost_window[1]) or rng.random() < dropout * 0.3
            if not dropped:
                ox, oy, *_ = true_state(oid, max(t - delay, 0))
                last[oid] = (t, ox + rng.normal(0, 3), oy + rng.normal(0, 3))
                hist.setdefault(oid, []).append(last[oid])
            ts, lx, ly = last[oid]
            age = t - ts + delay
            status = "ok" if age <= 1.5 else ("stale" if age <= 5 else "lost")
            h = hist[oid]
            if len(h) >= 2:
                (t0, x0, y0), (t1, x1, y1) = h[-2], h[-1]
                vx, vy = (x1 - x0) / max(t1 - t0, 1e-6), (y1 - y0) / max(t1 - t0, 1e-6)
            else:
                vx = vy = 0.0
            cv = {"x": lx + vx * (age + HORIZON), "y": ly + vy * (age + HORIZON), "r": 0.0}
            kf = {"x": cv["x"], "y": cv["y"], "r": 6 + 3 * age}
            fx, fy, *_ = true_state(oid, t + HORIZON)
            ml = {"x": fx + rng.normal(0, 4), "y": fy + rng.normal(0, 4), "r": 5 + 2 * age}
            fobjs.append({
                "id": oid,
                "true": {"x": round(x, 2), "y": round(y, 2), "heading": hdg, "speed": round(spd, 2)},
                "obs": {"x": round(lx, 2), "y": round(ly, 2), "age_s": round(age, 2), "status": status},
                "pred": {d: {k: round(v, 2) for k, v in p.items()} for d, p in (("cv", cv), ("kf", kf), ("ml", ml))},
            })

        if len(conflict_ids) >= 2:
            truth_start = truth_start if truth_start is not None else t
            truth_end = t
        truth = [{"zone_id": "Z1", "objects": conflict_ids}] if len(conflict_ids) >= 2 else []
        frames.append({"t": t, "objects": fobjs, "alerts": [], "truth_conflicts": truth})

    # Mock alerts: a detector alerts when both AC1 and V1 predictions (plus radius) reach Z1.
    for fr in frames:
        p = {o["id"]: o["pred"] for o in fr["objects"]}
        for d in ("cv", "kf", "ml"):
            hits = [oid for oid in ("AC1", "V1") if zone_dist(p[oid][d]["x"], p[oid][d]["y"]) <= p[oid][d]["r"]]
            if len(hits) == 2 and (truth_end is None or fr["t"] <= truth_end):
                alert = {"detector": d, "zone_id": "Z1", "p": 1.0 if d == "cv" else (0.62 if d == "kf" else 0.81), "objects": hits}
                if truth_start is not None and fr["t"] <= truth_start:
                    alert["lead_s"] = truth_start - fr["t"]
                fr["alerts"].append(alert)
                first_alert.setdefault(d, fr["t"])

    events = []
    if lost_window:
        events += [{"t": lost_window[0], "kind": "dropout_start", "objects": ["V1"], "note": "V1 tracking lost"},
                   {"t": lost_window[1], "kind": "dropout_end", "objects": ["V1"], "note": "V1 tracking restored"}]
    if scenario_id == "cleared_crossing":
        events.append({"t": 10, "kind": "clearance", "objects": ["V1"], "note": "V1 cleared to cross at C (mock)"})
    if truth_start is not None:
        events += [{"t": truth_start, "kind": "truth_conflict_start", "zone_id": "Z1", "objects": ["AC1", "V1"], "note": "AC1 and V1 both in Z1"},
                   {"t": truth_end, "kind": "truth_conflict_end", "zone_id": "Z1", "objects": ["AC1", "V1"], "note": "Z1 clear"}]
    for d, t in first_alert.items():
        events.append({"t": t, "kind": "first_alert", "detector": d, "zone_id": "Z1", "note": f"First {d} alert"})
    events.sort(key=lambda e: e["t"])

    return {"schema_version": SCHEMA_VERSION, "is_mock": True, "scenario_id": scenario_id, "seed": seed,
            "dropout": dropout, "dt_s": DT, "pred_horizon_s": HORIZON, "detectors": ["cv", "kf", "ml"],
            "objects": OBJECTS, "frames": frames, "events": events}


def main():
    airport_map = {
        "schema_version": SCHEMA_VERSION, "is_mock": True, "airport": "MOCK", "units": "m",
        "bounds": {"xmin": -1600, "xmax": 1600, "ymin": -500, "ymax": 500},
        "runways": [{"id": "9/27", "polygon": [[-1500, -25], [1500, -25], [1500, 25], [-1500, 25]]}],
        "taxiways": [{"id": "C", "polyline": [[0, -400], [0, 400]]},
                     {"id": "A", "polyline": [[-1500, -200], [1500, -200]]}],
        "zones": [ZONE],
    }
    runs = [("cleared_crossing", 1, 0.0), ("tracking_dropout", 1, 0.3), ("delayed_data", 1, 0.0)]
    scenarios = {
        "schema_version": SCHEMA_VERSION, "is_mock": True,
        "scenarios": [
            {"id": "cleared_crossing", "name": "Cleared crossing during landing roll", "description": "A vehicle is cleared across the runway while an aircraft is rolling out."},
            {"id": "tracking_dropout", "name": "Vehicle tracking dropout", "description": "The crossing vehicle's position updates stop for 12 s."},
            {"id": "delayed_data", "name": "Delayed surveillance data", "description": "All position reports arrive 3 s late."},
        ],
        "replays": [{"scenario_id": s, "seed": sd, "dropout": d, "file": replay_filename(s, sd, d)} for s, sd, d in runs],
    }
    det = ["cv", "kf", "ml"]
    metrics = {
        "schema_version": SCHEMA_VERSION, "is_mock": True, "notes": "MOCK NUMBERS, for UI layout only",
        "detectors": det,
        "summary": [{"detector": "cv", "recall": 0.55, "precision": 0.70, "false_alerts_per_hr": 4.0, "median_lead_s": 3, "pr_auc": 0.52},
                    {"detector": "kf", "recall": 0.78, "precision": 0.66, "false_alerts_per_hr": 5.5, "median_lead_s": 6, "pr_auc": 0.71},
                    {"detector": "ml", "recall": 0.86, "precision": 0.72, "false_alerts_per_hr": 4.8, "median_lead_s": 7, "pr_auc": 0.80}],
        "error_by_age": [{"detector": d, "age_bucket": b, "horizon_s": 5, "mean_err_m": round(e * m, 1)}
                         for d, m in zip(det, (1.6, 1.0, 0.75)) for b, e in zip(("0-1s", "1-3s", "3s+"), (6, 14, 38))],
        "coverage": [{"detector": d, "age_bucket": b, "target": 0.9, "empirical": c}
                     for d, cs in (("kf", (0.93, 0.88, 0.79)), ("ml", (0.91, 0.90, 0.87))) for b, c in zip(("0-1s", "1-3s", "3s+"), cs)],
        "pr_curve": [{"detector": d, "threshold": round(th, 1), "precision": round(min(1, 0.4 + 0.55 * th * q), 2), "recall": round(max(0, 1 - th * (1.1 - q * 0.3)), 2)}
                     for d, q in zip(det, (0.7, 0.85, 1.0)) for th in np.linspace(0.1, 0.9, 9)],
        "robustness": [{"detector": d, "dropout": dr, "recall": round(r0 - k * dr, 2)}
                       for d, r0, k in (("cv", 0.6, 0.6), ("kf", 0.8, 0.4), ("ml", 0.88, 0.3)) for dr in (0.0, 0.3, 0.6)],
        "per_scenario": [{"scenario_id": s, "detector": d, "recall": r, "false_alerts_per_hr": fa}
                         for s, rs in (("cleared_crossing", (0.8, 0.9, 0.93)), ("tracking_dropout", (0.3, 0.7, 0.8)), ("delayed_data", (0.5, 0.75, 0.84)))
                         for d, r, fa in zip(det, rs, (4.0, 5.5, 4.8))],
    }
    for obj, name in ((airport_map, "map.json"), (scenarios, "scenarios.json"), (metrics, "metrics.json")):
        print("wrote", write(obj, name, OUT))
    for s, sd, d in runs:
        print("wrote", write(build_replay(s, sd, d), replay_filename(s, sd, d), OUT))


if __name__ == "__main__":
    main()
