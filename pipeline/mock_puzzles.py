"""MOCK puzzles for the game loop (contract v1.1). Called by make_fixtures.

Same fictional airport as the mock replays: runway 9/27 along y=0, taxiway C at x=0,
crossing zone Z1 = the +/-60 m square. A vehicle (V1) holds short of the runway at
(0, -90); at decision_t it drives north across the runway at 8 m/s. A landing aircraft
(AC1) rolls out along +x. Whether they get close depends on AC1's start, speed and braking.

The AI here is FAKE (forecasts are hand-built so the UI has something to draw, including
one case where the AI is wrong). Every file has is_mock=true. Never report these numbers.
"""
import math

import numpy as np

from pipeline.contracts import SCHEMA_VERSION, puzzle_filename, snap_confidence

DT = 1.0
DECISION_T = 20
END_T = 50
THRESHOLD_M = 100.0
HOLD_SHORT_Y = -90.0
VEH_SPEED = 8.0
ZONE_HALF = 60.0


def in_zone(x, y):
    return abs(x) <= ZONE_HALF and abs(y) <= ZONE_HALF


def aircraft(t, x0, v0, decel):
    """Landing roll along +x: decelerate to 8 m/s taxi speed, then hold that speed."""
    t_floor = max((v0 - 8) / decel, 0)
    if t <= t_floor:
        return x0 + v0 * t - 0.5 * decel * t * t, v0 - decel * t
    x_floor = x0 + v0 * t_floor - 0.5 * decel * t_floor ** 2
    return x_floor + 8 * (t - t_floor), 8.0


def vehicle(t):
    """Approach the hold-short line, wait, then cross at decision_t (it really crossed)."""
    if t <= DECISION_T:
        return 0.0, max(-224 + 8 * t, HOLD_SHORT_Y), (8.0 if -224 + 8 * t < HOLD_SHORT_Y else 0.0)
    return 0.0, HOLD_SHORT_Y + VEH_SPEED * (t - DECISION_T), VEH_SPEED


def taxi_east(t, x0, y, v):
    return x0 + v * t, y, v


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def p_safe_from_path(ac_path, veh_path):
    """P(closest approach while the vehicle is in the zone >= threshold), Gaussian on the 90% radius."""
    best = None
    for a, v in zip(ac_path, veh_path):
        if not in_zone(v["x"], v["y"]):
            continue
        d = math.hypot(a["x"] - v["x"], a["y"] - v["y"])
        sigma = max(a["r"], 1.0) / 1.645
        p = phi((d - THRESHOLD_M) / sigma)
        if best is None or p < best[0]:
            best = (p, d, a["t"], a["r"])
    p, d, t, r = best
    return min(max(p, 0.02), 0.98), d, t, r


def build_puzzle(pid, difficulty, title, ac, extras=(), fog_lost_from=None, ai_mode="good", seed=0):
    rng = np.random.default_rng(seed)
    x0, v0, decel = ac
    objects = [{"id": "AC1", "type": "aircraft", "label": f"MOCK{100 + seed} (landing)"},
               {"id": "V1", "type": "vehicle", "label": "Rescue 1 (mock)"}]
    for e in extras:
        objects.append({"id": e[0], "type": e[1], "label": e[2]})

    def truth(oid, t):
        if oid == "AC1":
            x, s = aircraft(t, x0, v0, decel)
            return x, 0.0, 90.0, s
        if oid == "V1":
            x, y, s = vehicle(t)
            return x, y, 0.0, s
        e = next(e for e in extras if e[0] == oid)
        x, y, s = taxi_east(t, *e[3])
        return x, y, 90.0, s

    ids = [o["id"] for o in objects]
    times = [round(i * DT, 3) for i in range(int(END_T / DT) + 1)]

    # ---- pre: what the player sees (observed positions in fog mode) ----
    pre, last_seen = [], {}
    for t in times:
        if t > DECISION_T:
            break
        objs = []
        for oid in ids:
            x, y, h, s = truth(oid, t)
            lost = fog_lost_from is not None and oid == "AC1" and t >= fog_lost_from
            if not lost:
                noise = 6 if difficulty == "fog" else 0
                last_seen[oid] = (t, x + rng.normal(0, noise), y + rng.normal(0, noise), h, s)
            ts, lx, ly, lh, ls = last_seen[oid]
            age = round(t - ts, 2)
            status = "ok" if age <= 1.5 else ("stale" if age <= 5 else "lost")
            objs.append({"id": oid, "x": round(lx, 1), "y": round(ly, 1), "heading": lh, "speed": round(ls, 1),
                         "age_s": age, "status": status})
        pre.append({"t": t, "objects": objs})

    # ---- reveal: the truth after the freeze ----
    reveal = []
    for t in times:
        if t <= DECISION_T:
            continue
        objs = []
        for oid in ids:
            x, y, h, s = truth(oid, t)
            objs.append({"id": oid, "x": round(x, 1), "y": round(y, 1), "heading": h, "speed": round(s, 1)})
        reveal.append({"t": t, "objects": objs})

    # ---- outcome: closest approach while the vehicle is in the zone ----
    seps = []
    for fr in reveal:
        o = {ob["id"]: ob for ob in fr["objects"]}
        if in_zone(o["V1"]["x"], o["V1"]["y"]):
            seps.append((math.hypot(o["AC1"]["x"] - o["V1"]["x"], o["AC1"]["y"] - o["V1"]["y"]), fr["t"]))
    min_sep, min_t = min(seps)
    conflict = min_sep < THRESHOLD_M

    # ---- fake AI forecasts ----
    future = [t for t in times if t > DECISION_T]
    seen_t, sx, sy, _, sspeed = last_seen["AC1"]
    age0 = DECISION_T - seen_t

    def cv_path(r0, growth):
        return [{"t": t, "x": round(sx + sspeed * (t - seen_t), 1), "y": round(sy, 1),
                 "r": round(r0 + growth * (t - DECISION_T + age0), 1)} for t in future]

    def truth_path(noise, r0, growth):
        out = []
        for t in future:
            x, y, _, _ = truth("AC1", t)
            out.append({"t": t, "x": round(x + rng.normal(0, noise), 1), "y": round(y + rng.normal(0, noise / 2), 1),
                        "r": round(r0 + growth * (t - DECISION_T) + 2 * age0, 1)})
        return out

    veh_path = [{"t": t, "x": 0.0, "y": round(HOLD_SHORT_Y + VEH_SPEED * (t - DECISION_T), 1), "r": 4.0} for t in future]
    paths = {"cv": cv_path(5, 0.0), "kf": cv_path(8, 3.0),
             "ml": truth_path(4, 8, 1.5) if ai_mode == "good" else cv_path(10, 1.0)}

    calls = {}
    for det, path in paths.items():
        p, d, t_close, r = p_safe_from_path(path, veh_path)
        call, conf = snap_confidence(p)
        calls[det] = {"call": call, "p_safe": round(p, 3), "confidence": conf, "_d": d, "_t": t_close, "_r": r}

    ml = calls["ml"]
    eta = next((pt["t"] - DECISION_T for pt in paths["ml"] if abs(pt["x"]) <= ZONE_HALF), None)
    eta_txt = f"reach the crossing in about {eta:.0f} s" if eta is not None else "not reach the crossing in the next 30 s"
    reasoning = (f"MOCK: I predict MOCK{100 + seed} will {eta_txt}. Rescue 1 needs about "
                 f"{(2 * ZONE_HALF + 30) / VEH_SPEED:.0f} s to clear the runway. Predicted closest approach "
                 f"{ml['_d']:.0f} m ±{ml['_r']:.0f} m against a {THRESHOLD_M:.0f} m limit, so {ml['call']} "
                 f"({ml['confidence'] * 100:.0f}% sure).")

    puzzle = {
        "schema_version": SCHEMA_VERSION, "is_mock": True, "id": pid, "difficulty": difficulty,
        "source": {"airport": "MOCK", "recording": "mock", "t0_s": 0},
        "dt_s": DT, "decision_t": DECISION_T, "zone_id": "Z1", "vehicle_id": "V1", "aircraft_id": "AC1",
        "question": "Rescue 1 is holding short of RWY 9/27 at taxiway C. Clear it to cross now?",
        "objects": objects, "pre": pre,
        "ai": {"detector": "ml", "call": ml["call"], "p_safe": ml["p_safe"], "confidence": ml["confidence"],
               "paths": {"AC1": paths["ml"], "V1": veh_path}, "reasoning": reasoning,
               "coach": f"MOCK coach: closest approach was {min_sep:.0f} m at t={min_t:.0f} s.",
               "baselines": {d: {k: c[k] for k in ("call", "p_safe", "confidence")} for d, c in calls.items()}},
        "reveal": reveal,
        "outcome": {"conflict": conflict, "threshold_m": THRESHOLD_M, "min_sep_m": round(min_sep, 1),
                    "min_sep_t": min_t, "correct_call": "HOLD" if conflict else "CLEAR",
                    "note": f"Closest approach {min_sep:.0f} m at t={min_t:.0f} s "
                            f"({'inside' if conflict else 'outside'} the {THRESHOLD_M:.0f} m limit)."},
    }
    return puzzle, title


BUSY = (("AC2", "aircraft", "MOCK202 (taxi)", (-900, -200.0, 10.0)),
        ("V2", "vehicle", "Ops 4 (mock)", (300, -200.0, 6.0)),
        ("AC3", "aircraft", "MOCK303 (taxi)", (-1400, -200.0, 7.0)))

SPECS = [
    # id,     difficulty, title,                 (x0, v0, decel),   extras, fog_lost_from, ai_mode, seed
    # Intended mix (checked by tests): 3 conflicts / 3 safe, two 75% AI calls, and the AI
    # is confidently WRONG on mock_06 (stale fog data). On mock_02 the Kalman baseline is wrong.
    ("mock_01", "clear", "Landing traffic, taxiway C", (-1800, 70, 1.0), (), None, "good", 1),
    ("mock_02", "clear", "Slow arrival, taxiway C", (-1800, 60, 0.8), (), None, "good", 2),
    ("mock_03", "busy", "Rush hour at C", (-2000, 70, 1.0), BUSY, None, "good", 3),
    ("mock_04", "busy", "After the arrival, taxiway C", (-900, 70, 1.0), BUSY, None, "good", 4),
    ("mock_05", "fog", "Fog: arrival lost on radar", (-1800, 70, 1.0), (), 14, "good", 5),
    ("mock_06", "fog", "Fog: stale arrival", (-2500, 75, 1.0), (), 12, "fooled", 6),
]

DIFFICULTIES = [
    {"id": "clear", "name": "Clear day", "description": "Light traffic, perfect surveillance."},
    {"id": "busy", "name": "Busy hour", "description": "More traffic on the screen. Same rules."},
    {"id": "fog", "name": "Fog mode", "description": "Surveillance is stale or missing. You only see where things were."},
]


def build_all():
    """Returns (index, [(filename, puzzle), ...])."""
    built = [build_puzzle(*s) for s in SPECS]
    index = {
        "schema_version": SCHEMA_VERSION, "is_mock": True, "conflict_threshold_m": THRESHOLD_M,
        "difficulties": DIFFICULTIES,
        "puzzles": [{"id": p["id"], "difficulty": p["difficulty"], "title": title, "file": puzzle_filename(p["id"])}
                    for p, title in built],
    }
    return index, [(puzzle_filename(p["id"]), p) for p, _ in built]
