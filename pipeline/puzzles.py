"""Puzzle generator: cut real runway-crossing moments out of the recording and turn
each one into a round of the game (contract v1.1: puzzles.json + puzzle_<id>.json).

    python -m pipeline.truth      # first: makes data/derived/ (tracks + crossings)
    python -m pipeline.puzzles    # then:  writes puzzles into web/public/data

What a puzzle is
----------------
A vehicle is at (or approaching) a runway. We freeze the recording at the decision moment
and ask: "clear it to cross now?". After the player answers, the next ~30 s play out
and we measure how close the vehicle got to an aircraft while it was on the runway.
Closer than CONFLICT_THRESHOLD_M means crossing then was a conflict, so HOLD was right.

Two kinds of puzzle
-------------------
* REAL: the moment just before a vehicle really crossed. The reveal is the recording.
* WHAT IF: in this hour every real crossing was safe (the closest any aircraft came was
  150 m), so real crossings alone would make "always CLEAR" unbeatable. But some
  vehicles WAITED at the hold line while jets used the runway. For moments during those
  waits we ask "what if it had gone now?". The vehicle replays ITS OWN REAL CROSSING,
  started at the decision moment, and every aircraft moves EXACTLY as recorded. These
  puzzles carry source.vehicle_time_shift_s so the UI can label them "WHAT IF".

The AI player here is the constant-velocity baseline (the Kalman and ML players come
later and only replace `forecast_aircraft()`).
"""
import argparse
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline import truth
from pipeline.contracts import (
    ROOT,
    SCHEMA_VERSION,
    WEB_DATA_DIR,
    puzzle_filename,
    snap_confidence,
    write,
)

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

CONFLICT_THRESHOLD_M = 100.0   # closest approach below this while on the runway = conflict.
                               # 100 m gives a usable number of HOLD moments in this hour
                               # (50 m gives 13 out of 163 candidates, 100 m gives 20).

SECONDS_BEFORE_DECISION = 20   # how much traffic the player watches before the freeze
SECONDS_AFTER_DECISION = 30    # how much of the future the reveal shows

# A vehicle counts as "waiting at the hold line" when it is stopped this far from the
# runway centreline (just outside the runway, around the hold-short line).
WAITING_SPEED_MS = 1.0
WAITING_DISTANCE_M = (55, 160)
LOOK_BACK_FOR_WAIT_S = 150     # how far before the crossing we look for a wait
MIN_WAIT_FOR_WHAT_IF_S = 10    # shorter waits don't get what-if puzzles

DECISION_SPACING_S = 10        # two puzzles from the same wait must be at least this far apart
WHAT_IF_STEP_S = 5             # try a decision moment every 5 s during a wait

VIEW_RADIUS_M = 1500           # traffic within this distance of the crossing is shown
MAX_PUZZLES = 40
TARGET_HOLD_SHARE = 0.35       # aim for about a third of puzzles where HOLD is right

# Constant-velocity AI player
AI_HORIZON_S = SECONDS_AFTER_DECISION
AI_BASE_RADIUS_M = 10.0        # 90% uncertainty radius right away...
AI_RADIUS_GROWTH_M_PER_S = 3.0  # ...growing by this much per second of look-ahead
AI_ASSUMED_VEHICLE_SPEED_MS = 6.0  # typical crossing speed in this hour (~150 m in ~25 s)

PUZZLE_DIR_DEFAULT = WEB_DATA_DIR
DERIVED = truth.DERIVED
RECORDING = os.path.splitext(os.path.basename(truth.TRACKS_CSV))[0]
RECORDING_START_UTC = int(RECORDING.rsplit("_", 1)[-1])   # file name ends with the UTC epoch


# --------------------------------------------------------------------------------------
# Small data holders
# --------------------------------------------------------------------------------------

@dataclass
class Candidate:
    """One possible puzzle: a vehicle, a runway, and the moment we freeze."""
    kind: str                    # "real" or "what_if"
    runway: str
    zone_id: str
    vehicle_id: str
    decision_time: float         # seconds into the recording
    go_time: float               # when the vehicle REALLY started its crossing
    crossing_end_time: float     # when it REALLY left the runway
    was_waiting: bool            # stopped at the hold line at the decision moment?

    @property
    def time_shift(self):
        """How far the vehicle's crossing is moved in time (0 for real puzzles)."""
        return self.decision_time - self.go_time if self.kind == "what_if" else 0.0


@dataclass
class Outcome:
    """What happened after the decision moment."""
    conflict: bool
    min_separation_m: float
    min_separation_time: float   # seconds into the recording
    aircraft_id: str             # the aircraft that came closest


@dataclass
class Scene:
    """Everything we need to look up positions quickly."""
    tracks: pd.DataFrame
    runways: dict                                       # name -> truth.Runway
    zones: dict                                         # zone id -> polygon (list of [x, y])
    track_by_id: dict = field(default_factory=dict)     # id -> DataFrame indexed by t
    kind_by_id: dict = field(default_factory=dict)      # id -> "aircraft" / "vehicle" / ...


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def load_scene():
    """Read the truth layer's outputs and index every track by agent id."""
    tracks_path = os.path.join(DERIVED, "tracks.csv.gz")
    if not os.path.exists(tracks_path):
        raise SystemExit("data/derived/tracks.csv.gz is missing: run `python -m pipeline.truth` first.")

    tracks = pd.read_csv(tracks_path, dtype={"id": str})
    runways = {r.name: r for r in truth.build_runways(truth.load_map_segments(), tracks)}

    # Rebuild the zone polygons the same way truth.py did (same crossings in -> same zones out),
    # so zone ids here always match the ones in map.json.
    crossings = pd.read_csv(os.path.join(DERIVED, "crossings.csv"), dtype={"vehicle_id": str})
    zone_list, crossings = truth.build_zones(crossings.drop(columns=["zone_id"]), list(runways.values()))
    zones = {zone["id"]: zone["polygon"] for zone in zone_list}

    scene = Scene(tracks=tracks, runways=runways, zones=zones)
    for agent_id, track in tracks.groupby("id"):
        scene.track_by_id[agent_id] = track.set_index("t")
        scene.kind_by_id[agent_id] = track["kind"].iloc[0]
    return scene, crossings


# --------------------------------------------------------------------------------------
# Positions (real and what-if)
# --------------------------------------------------------------------------------------

def real_position(scene, agent_id, time):
    """Recorded (x, y, heading, speed) of an agent at a whole second, or None if not seen then."""
    track = scene.track_by_id[agent_id]
    if time not in track.index:
        return None
    row = track.loc[time]
    return float(row["x"]), float(row["y"]), float(row["heading"]), float(row["speed"])


def vehicle_position(scene, candidate, time):
    """Where the puzzle's vehicle is at `time`.

    Before the decision moment it is always the recording. After it, a what-if vehicle
    replays its real crossing shifted in time: at decision_time + k it is wherever it
    really was at go_time + k. Once that replay is over it stays where it stopped."""
    if candidate.kind == "real" or time <= candidate.decision_time:
        return real_position(scene, candidate.vehicle_id, time)

    replay_time = min(time - candidate.time_shift, candidate.crossing_end_time + 5)
    return real_position(scene, candidate.vehicle_id, replay_time)


def position_of(scene, candidate, agent_id, time):
    """Position of any agent in this puzzle at `time` (the vehicle may be a what-if replay)."""
    if agent_id == candidate.vehicle_id:
        return vehicle_position(scene, candidate, time)
    return real_position(scene, agent_id, time)


# --------------------------------------------------------------------------------------
# Finding candidates
# --------------------------------------------------------------------------------------

def find_wait_before_crossing(scene, crossing):
    """Seconds (recording time) the vehicle spent stopped at the hold line right before it crossed.

    Returns (wait_times, go_time): go_time is the last second it was still waiting, i.e. when it
    really set off. If it never stopped, go_time is a few seconds before it entered the runway."""
    runway = scene.runways[crossing.runway]
    track = scene.track_by_id[crossing.vehicle_id]
    before = track.loc[crossing.t_enter - LOOK_BACK_FOR_WAIT_S: crossing.t_enter - 1]

    _, distance_from_centreline = runway.to_runway_coords(before["x"].values, before["y"].values)
    distance_from_centreline = np.abs(distance_from_centreline)
    stopped_at_hold_line = (
        (before["speed"].values < WAITING_SPEED_MS)
        & (distance_from_centreline >= WAITING_DISTANCE_M[0])
        & (distance_from_centreline <= WAITING_DISTANCE_M[1])
    )
    wait_times = before.index.values[stopped_at_hold_line]

    if len(wait_times):
        go_time = float(wait_times.max())
    else:
        go_time = float(crossing.t_enter - 8)     # it rolled straight through; freeze 8 s before
    return wait_times, go_time


def make_candidates(scene, crossings):
    """A REAL candidate for every crossing on an active runway, plus WHAT-IF candidates
    every few seconds while the vehicle was waiting at the hold line."""
    active_runways = {name for name, rw in scene.runways.items() if rw.active}
    candidates = []

    for crossing in crossings.itertuples():
        if crossing.runway not in active_runways:
            continue

        wait_times, go_time = find_wait_before_crossing(scene, crossing)
        common = dict(runway=crossing.runway, zone_id=crossing.zone_id, vehicle_id=crossing.vehicle_id,
                      go_time=go_time, crossing_end_time=float(crossing.t_exit))

        candidates.append(Candidate(kind="real", decision_time=go_time,
                                    was_waiting=len(wait_times) > 0, **common))

        if len(wait_times) < MIN_WAIT_FOR_WHAT_IF_S:
            continue
        first_decision = wait_times.min() + 5            # give the vehicle a moment to settle
        last_decision = go_time - DECISION_SPACING_S     # leave room for the real puzzle
        for decision_time in np.arange(first_decision, last_decision + 1, WHAT_IF_STEP_S):
            candidates.append(Candidate(kind="what_if", decision_time=float(decision_time),
                                        was_waiting=True, **common))
    return candidates


# --------------------------------------------------------------------------------------
# Outcome: how close did it get?
# --------------------------------------------------------------------------------------

def vehicle_is_on_runway(runway, x, y):
    """True when (x, y) is on the runway surface used for crossings (same rule as truth.py)."""
    along, perp = runway.to_runway_coords(x, y)
    return (abs(perp) <= truth.CROSSING_HALF_WIDTH_M) and (runway.along_min <= along <= runway.along_max)


def interpolate(p0, p1, fraction):
    """Point a fraction of the way from p0 to p1 (positions are (x, y, ...))."""
    return p0[0] + (p1[0] - p0[0]) * fraction, p0[1] + (p1[1] - p0[1]) * fraction


def measure_outcome(scene, candidate):
    """Closest approach between the vehicle and ANY aircraft while the vehicle is on the runway,
    over the reveal window. Positions are compared every 0.25 s (linearly interpolated between
    the 1 s reports) so a fast jet can't 'jump over' the vehicle between two samples."""
    runway = scene.runways[candidate.runway]
    start, end = candidate.decision_time, candidate.decision_time + SECONDS_AFTER_DECISION
    aircraft_ids = [a for a, k in scene.kind_by_id.items() if k == "aircraft"]

    best = Outcome(conflict=False, min_separation_m=math.inf, min_separation_time=start, aircraft_id="")
    for second in np.arange(start, end):
        vehicle_now = vehicle_position(scene, candidate, second)
        vehicle_next = vehicle_position(scene, candidate, second + 1)
        if vehicle_now is None or vehicle_next is None:
            continue

        for aircraft_id in aircraft_ids:
            aircraft_now = real_position(scene, aircraft_id, second)
            aircraft_next = real_position(scene, aircraft_id, second + 1)
            if aircraft_now is None or aircraft_next is None:
                continue

            for fraction in (0.0, 0.25, 0.5, 0.75):
                vx, vy = interpolate(vehicle_now, vehicle_next, fraction)
                if not vehicle_is_on_runway(runway, vx, vy):
                    continue
                ax, ay = interpolate(aircraft_now, aircraft_next, fraction)
                distance = math.hypot(ax - vx, ay - vy)
                if distance < best.min_separation_m:
                    best = Outcome(conflict=distance < CONFLICT_THRESHOLD_M, min_separation_m=distance,
                                   min_separation_time=second + fraction, aircraft_id=aircraft_id)
    return best


def closest_moving_aircraft(scene, candidate):
    """Fallback 'question aircraft' when the vehicle never got onto the runway in the reveal:
    the aircraft that came closest to the crossing zone during the whole puzzle."""
    zone_center = np.mean(scene.zones[candidate.zone_id], axis=0)
    start = candidate.decision_time - SECONDS_BEFORE_DECISION
    end = candidate.decision_time + SECONDS_AFTER_DECISION
    window = scene.tracks[(scene.tracks.kind == "aircraft") & scene.tracks.t.between(start, end)]
    if window.empty:
        return None
    distance = np.hypot(window.x - zone_center[0], window.y - zone_center[1])
    return window.loc[distance.idxmin(), "id"]


# --------------------------------------------------------------------------------------
# Choosing which candidates become puzzles
# --------------------------------------------------------------------------------------

def space_out(candidates):
    """Drop candidates closer than DECISION_SPACING_S to an already-kept one from the same
    vehicle and runway (5 s apart they'd be near-duplicate puzzles). HOLD candidates are
    kept first so we don't throw the rare ones away."""
    kept = []
    ordered = sorted(candidates, key=lambda c: (not c.outcome.conflict, c.decision_time))
    for candidate in ordered:
        too_close = any(
            other.vehicle_id == candidate.vehicle_id
            and other.runway == candidate.runway
            and abs(other.decision_time - candidate.decision_time) < DECISION_SPACING_S
            for other in kept
        )
        if not too_close:
            kept.append(candidate)
    return kept


def spread_pick(pool, how_many, prefer_runways):
    """Pick `how_many` candidates from `pool`, spread from the closest calls to the easiest
    ones (so it's a mix of hard and easy), taking runways in `prefer_runways` first."""
    if how_many <= 0 or not pool:
        return []
    preferred = sorted((c for c in pool if c.runway in prefer_runways), key=lambda c: c.outcome.min_separation_m)
    others = sorted((c for c in pool if c.runway not in prefer_runways), key=lambda c: c.outcome.min_separation_m)

    picked = []
    for group in (preferred, others):
        still_needed = how_many - len(picked)
        if still_needed <= 0 or not group:
            continue
        positions = np.linspace(0, len(group) - 1, min(still_needed, len(group)))
        picked += [group[int(round(p))] for p in positions]
    return picked


def choose_puzzles(candidates):
    """Every HOLD puzzle, plus enough CLEAR puzzles to make HOLD about TARGET_HOLD_SHARE.

    The CLEAR puzzles are chosen so that nothing on screen gives the answer away:
      * the WHAT IF tag: as many what-if CLEARs as what-if HOLDs (a tag means 50/50),
      * the runway name: CLEARs come from the runways that have HOLDs first.
    Within each group we spread picks from close calls to easy ones."""
    holds = [c for c in candidates if c.outcome.conflict]
    clears = [c for c in candidates if not c.outcome.conflict]

    wanted_clears = round(len(holds) * (1 - TARGET_HOLD_SHARE) / TARGET_HOLD_SHARE)
    wanted_clears = max(0, min(wanted_clears, MAX_PUZZLES - len(holds)))

    hold_runways = {c.runway for c in holds}
    what_if_holds = sum(c.kind == "what_if" for c in holds)

    what_if_clears = spread_pick([c for c in clears if c.kind == "what_if"],
                                 min(what_if_holds, wanted_clears), hold_runways)
    real_clears = spread_pick([c for c in clears if c.kind == "real"],
                              wanted_clears - len(what_if_clears), hold_runways)

    return sorted(holds + what_if_clears + real_clears, key=lambda c: c.decision_time)


# --------------------------------------------------------------------------------------
# The constant-velocity AI player
# --------------------------------------------------------------------------------------

def standard_normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def last_seen(pre_frames, agent_id, max_age_s=3):
    """The agent's most recent position in the pre frames, if seen in the last few seconds."""
    decision_t = pre_frames[-1]["t"]
    for frame in reversed(pre_frames):
        if decision_t - frame["t"] > max_age_s:
            return None
        for obj in frame["objects"]:
            if obj["id"] == agent_id:
                return obj
    return None


def recent_velocity(frames, agent_id, seconds=3):
    """Average velocity (m/s, as vx, vy) over the last few seen positions in the pre frames."""
    seen = [(f["t"], o["x"], o["y"]) for f in frames for o in f["objects"] if o["id"] == agent_id]
    seen = seen[-(seconds + 1):]
    if len(seen) < 2:
        return 0.0, 0.0
    (t0, x0, y0), (t1, x1, y1) = seen[0], seen[-1]
    return (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)


def forecast_aircraft(pre_frames, aircraft_id, decision_t):
    """Constant-velocity forecast: keep going at the recent speed and direction.
    Returns [{t, x, y, r}] for every second after the decision, r = 90% radius."""
    last = last_seen(pre_frames, aircraft_id)
    vx, vy = recent_velocity(pre_frames, aircraft_id)
    path = []
    for step in range(1, AI_HORIZON_S + 1):
        path.append({
            "t": decision_t + step,
            "x": round(last["x"] + vx * step, 1),
            "y": round(last["y"] + vy * step, 1),
            "r": round(AI_BASE_RADIUS_M + AI_RADIUS_GROWTH_M_PER_S * step, 1),
        })
    return path


def assumed_vehicle_path(pre_frames, vehicle_id, runway, decision_t):
    """What the AI assumes the vehicle will do if cleared: drive straight across the runway,
    at a typical speed, to the same distance on the far side. (It does NOT know the real path.)"""
    last = last_seen(pre_frames, vehicle_id)
    along, perp = runway.to_runway_coords(last["x"], last["y"])
    far_side = -np.sign(perp) * max(abs(perp), truth.CROSSING_HALF_WIDTH_M + 10)
    total_distance = abs(far_side - perp)

    path = []
    for step in range(1, AI_HORIZON_S + 1):
        travelled = min(AI_ASSUMED_VEHICLE_SPEED_MS * step, total_distance)
        new_perp = perp + np.sign(far_side - perp) * travelled
        x, y = runway.to_world(along, new_perp)
        path.append({"t": decision_t + step, "x": round(float(x), 1), "y": round(float(y), 1), "r": 5.0})
    return path


def ai_decision(pre_frames, objects, vehicle_id, runway, decision_t):
    """The AI's call. For every aircraft on screen, forecast its path; while the (assumed)
    vehicle is on the runway, how likely is that aircraft to be within the threshold?
    p_safe = 1 - the worst of those probabilities.

    Returns (p_safe, paths, reasoning, worst_aircraft_id)."""
    vehicle_path = assumed_vehicle_path(pre_frames, vehicle_id, runway, decision_t)
    on_runway = [vehicle_is_on_runway(runway, p["x"], p["y"]) for p in vehicle_path]
    aircraft_on_screen = [o["id"] for o in objects if o["type"] == "aircraft"]

    worst = {"p_conflict": 0.0, "aircraft": None, "distance": math.inf, "radius": 0.0, "path": None, "time": None}
    for aircraft_id in aircraft_on_screen:
        if last_seen(pre_frames, aircraft_id) is None:
            continue                                  # not visible around the freeze: the AI can't see it
        path = forecast_aircraft(pre_frames, aircraft_id, decision_t)
        for vehicle_point, aircraft_point, vehicle_on_runway in zip(vehicle_path, path, on_runway):
            if not vehicle_on_runway:
                continue
            distance = math.hypot(aircraft_point["x"] - vehicle_point["x"], aircraft_point["y"] - vehicle_point["y"])
            sigma = aircraft_point["r"] / 1.645                      # 90% radius -> standard deviation
            p_conflict = standard_normal_cdf((CONFLICT_THRESHOLD_M - distance) / sigma)
            if p_conflict > worst["p_conflict"] or worst["aircraft"] is None:
                worst = {"p_conflict": p_conflict, "aircraft": aircraft_id, "distance": distance,
                         "radius": aircraft_point["r"], "path": path, "time": aircraft_point["t"]}

    p_safe = min(max(1 - worst["p_conflict"], 0.02), 0.98)
    call, confidence = snap_confidence(p_safe)

    paths = {vehicle_id: vehicle_path}
    if worst["aircraft"]:
        paths[worst["aircraft"]] = worst["path"]

    seconds_on_runway = sum(on_runway)
    if worst["aircraft"] is None:
        reasoning = (f"No aircraft in view. Crossing takes about {seconds_on_runway} s, "
                     f"so {call} ({confidence * 100:.0f}% sure).")
    else:
        label = object_label(worst["aircraft"], "aircraft")
        reasoning = (f"Keeping {label} at its current speed and heading (constant-velocity forecast), "
                     f"the closest it gets while the vehicle is on the runway is about "
                     f"{worst['distance']:.0f} m ±{worst['radius']:.0f} m, "
                     f"{worst['time'] - decision_t:.0f} s from now. The limit is {CONFLICT_THRESHOLD_M:.0f} m, "
                     f"so {call} ({confidence * 100:.0f}% sure).")
    return p_safe, paths, reasoning, worst["aircraft"]


# --------------------------------------------------------------------------------------
# Building the puzzle file
# --------------------------------------------------------------------------------------

def object_label(agent_id, kind):
    """Short, readable label. The data has no call signs, so we use the last 4 digits of the id."""
    prefix = {"aircraft": "Aircraft", "vehicle": "Vehicle"}.get(kind, "Object")
    return f"{prefix} {agent_id[-4:]}"


def agents_in_view(scene, candidate, aircraft_id):
    """Aircraft and vehicles within VIEW_RADIUS_M of the crossing at any point in the puzzle.
    The puzzle's own vehicle and aircraft are always included."""
    zone_center = np.mean(scene.zones[candidate.zone_id], axis=0)
    start = candidate.decision_time - SECONDS_BEFORE_DECISION
    end = candidate.decision_time + SECONDS_AFTER_DECISION
    window = scene.tracks[
        scene.tracks.t.between(start, end) & scene.tracks.kind.isin(["aircraft", "vehicle"])
    ]
    distance = np.hypot(window.x - zone_center[0], window.y - zone_center[1])
    ids = set(window.loc[distance <= VIEW_RADIUS_M, "id"])
    ids |= {candidate.vehicle_id, aircraft_id}

    # puzzle vehicle and aircraft first, then the rest in a stable order
    ordered = [candidate.vehicle_id, aircraft_id] + sorted(ids - {candidate.vehicle_id, aircraft_id})
    return [{"id": a, "type": scene.kind_by_id[a], "label": object_label(a, scene.kind_by_id[a])} for a in ordered]


def frame_objects(scene, candidate, objects, recording_time, with_status):
    """Positions of every puzzle object at one recording second (skipping ones not seen then)."""
    out = []
    for obj in objects:
        position = position_of(scene, candidate, obj["id"], recording_time)
        if position is None:
            continue
        x, y, heading, speed = position
        entry = {"id": obj["id"], "x": round(x, 1), "y": round(y, 1),
                 "heading": round(heading, 1), "speed": round(speed, 1)}
        if with_status:
            entry.update(age_s=0, status="ok")      # clear / busy: perfect surveillance
        out.append(entry)
    return out


def puzzle_id_for(candidate):
    runway_tag = candidate.runway.replace("/", "").lower()
    return f"r{runway_tag}_v{candidate.vehicle_id[-4:]}_t{int(candidate.decision_time)}"


def local_clock(recording_time):
    """Recording seconds -> local Boston clock time (EST), e.g. '17:31'."""
    utc = datetime.fromtimestamp(RECORDING_START_UTC + recording_time, tz=timezone.utc)
    return f"{(utc.hour - 5) % 24:02d}:{utc.minute:02d}"


def build_puzzle(scene, candidate):
    """Turn a chosen candidate into a contract-shaped puzzle dict."""
    outcome = candidate.outcome
    runway = scene.runways[candidate.runway]
    decision_recording_time = candidate.decision_time
    start = decision_recording_time - SECONDS_BEFORE_DECISION

    aircraft_id = outcome.aircraft_id or closest_moving_aircraft(scene, candidate)
    objects = agents_in_view(scene, candidate, aircraft_id)

    # Puzzle time starts at 0 = SECONDS_BEFORE_DECISION before the freeze.
    pre_frames, reveal_frames = [], []
    for step in range(0, SECONDS_BEFORE_DECISION + SECONDS_AFTER_DECISION + 1):
        recording_time = start + step
        is_pre = recording_time <= decision_recording_time
        objs = frame_objects(scene, candidate, objects, recording_time, with_status=is_pre)
        (pre_frames if is_pre else reveal_frames).append({"t": float(step), "objects": objs})

    decision_t = float(SECONDS_BEFORE_DECISION)
    p_safe, paths, reasoning, _ = ai_decision(pre_frames, objects, candidate.vehicle_id, runway, decision_t)
    ai_call, ai_confidence = snap_confidence(p_safe)
    if aircraft_id not in paths:                    # contract: the question aircraft must have a path
        if last_seen(pre_frames, aircraft_id) is not None:
            paths[aircraft_id] = forecast_aircraft(pre_frames, aircraft_id, decision_t)
        else:
            paths[aircraft_id] = [{"t": decision_t + 1, "x": 0.0, "y": 0.0, "r": 0.0}]

    vehicle_label = object_label(candidate.vehicle_id, "vehicle")
    aircraft_label = object_label(aircraft_id, "aircraft")
    where = "is holding short of" if candidate.was_waiting else "is approaching"
    question = f"{vehicle_label} {where} RWY {candidate.runway}. Clear it to cross now?"

    sep = outcome.min_separation_m
    if math.isinf(sep):
        sep, note = 9999.0, f"{vehicle_label} never got onto the runway in the next {SECONDS_AFTER_DECISION} s."
    else:
        closest_t = outcome.min_separation_time - start
        note = (f"Closest approach: {sep:.0f} m to {aircraft_label}, {closest_t - decision_t:.0f} s after the "
                f"decision ({'inside' if outcome.conflict else 'outside'} the {CONFLICT_THRESHOLD_M:.0f} m limit).")
    if candidate.kind == "what_if":
        note += (f" What if: {vehicle_label} really waited another {-candidate.time_shift:.0f} s; here it "
                 f"crosses now, on its own real route. The aircraft are exactly as recorded.")

    source = {"airport": "KBOS", "recording": RECORDING, "t0_s": start}
    if candidate.kind == "what_if":
        source["vehicle_time_shift_s"] = candidate.time_shift

    return {
        "schema_version": SCHEMA_VERSION,
        "id": puzzle_id_for(candidate),
        "difficulty": "clear",                      # set properly in label_difficulty()
        "source": source,
        "dt_s": 1.0,
        "decision_t": decision_t,
        "zone_id": candidate.zone_id,
        "vehicle_id": candidate.vehicle_id,
        "aircraft_id": aircraft_id,
        "question": question,
        "objects": objects,
        "pre": pre_frames,
        "ai": {
            "detector": "cv",
            "call": ai_call,
            "p_safe": round(p_safe, 3),
            "confidence": ai_confidence,
            "paths": paths,
            "reasoning": reasoning,
            "baselines": {"cv": {"call": ai_call, "p_safe": round(p_safe, 3), "confidence": ai_confidence}},
        },
        "reveal": reveal_frames,
        "outcome": {
            "conflict": outcome.conflict,
            "threshold_m": CONFLICT_THRESHOLD_M,
            "min_sep_m": round(sep, 1),
            "min_sep_t": round(outcome.min_separation_time - start, 2),
            "correct_call": "HOLD" if outcome.conflict else "CLEAR",
            "note": note,
        },
    }


def label_difficulty(puzzles):
    """'busy' when more traffic is on screen than in a typical puzzle (above the median), else 'clear'.
    (Fog mode is added later from these same moments with degraded surveillance.)"""
    counts = [len(p["objects"]) for p in puzzles]
    median = float(np.median(counts)) if counts else 0
    for puzzle, count in zip(puzzles, counts):
        puzzle["difficulty"] = "busy" if count > median else "clear"
    return median


# --------------------------------------------------------------------------------------
# Check picture
# --------------------------------------------------------------------------------------

def plot_check(scene, puzzles, out_png, how_many=8):
    """Small multiples: pre paths (thin), reveal paths (thick), zone (gold), closest approach (x).
    Lets a human confirm that HOLD puzzles really show a jet passing the crossing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [p for p in puzzles if p["outcome"]["conflict"]][: how_many // 2]
    picks += [p for p in puzzles if not p["outcome"]["conflict"]][: how_many - len(picks)]
    columns = 4
    rows = math.ceil(len(picks) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False)

    for ax, puzzle in zip(axes.flat, picks):
        zone = np.array(scene.zones[puzzle["zone_id"]] + [scene.zones[puzzle["zone_id"]][0]])
        ax.fill(zone[:, 0], zone[:, 1], color="gold", alpha=0.5)
        for obj in puzzle["objects"][:2]:                          # vehicle + question aircraft
            color = "tab:red" if obj["type"] == "vehicle" else "tab:blue"
            for frames, width in ((puzzle["pre"], 1), (puzzle["reveal"], 3)):
                pts = np.array([(o["x"], o["y"]) for f in frames for o in f["objects"] if o["id"] == obj["id"]])
                if len(pts):
                    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=width)
        ai_path = np.array([(p["x"], p["y"]) for p in puzzle["ai"]["paths"].get(puzzle["aircraft_id"], [])])
        if len(ai_path):
            ax.plot(ai_path[:, 0], ai_path[:, 1], "b:", lw=1)
        center = zone[:-1].mean(axis=0)
        ax.set_xlim(center[0] - 700, center[0] + 700)
        ax.set_ylim(center[1] - 700, center[1] + 700)
        ax.set_aspect("equal")
        tag = "WHAT IF" if "vehicle_time_shift_s" in puzzle["source"] else "REAL"
        ax.set_title(f"{puzzle['id']} [{tag}]\n{puzzle['outcome']['correct_call']} · sep "
                     f"{puzzle['outcome']['min_sep_m']:.0f} m · AI {puzzle['ai']['call']} "
                     f"{puzzle['ai']['confidence'] * 100:.0f}%", fontsize=9)
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=70)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

DIFFICULTIES = [
    {"id": "clear", "name": "Clear day", "description": "Light traffic, perfect surveillance."},
    {"id": "busy", "name": "Busy hour", "description": "More traffic on the screen. Same rules."},
]


def main(out_dir=PUZZLE_DIR_DEFAULT):
    scene, crossings = load_scene()

    # 1. every possible decision moment, with what really happened next
    candidates = make_candidates(scene, crossings)
    candidates = [c for c in candidates                       # the player must see the vehicle at the freeze
                  if vehicle_position(scene, c, c.decision_time) is not None]
    for candidate in candidates:
        candidate.outcome = measure_outcome(scene, candidate)
    print(f"{len(candidates)} candidates, {sum(c.outcome.conflict for c in candidates)} where HOLD was right")

    # 2. thin out near-duplicates, then balance HOLD vs CLEAR
    chosen = choose_puzzles(space_out(candidates))

    # 3. build, label difficulty, write (every file is validated against the contract)
    puzzles = [build_puzzle(scene, c) for c in chosen]
    busy_cutoff = label_difficulty(puzzles)

    index = {
        "schema_version": SCHEMA_VERSION,
        "conflict_threshold_m": CONFLICT_THRESHOLD_M,
        "difficulties": DIFFICULTIES,
        "puzzles": [
            {"id": p["id"], "difficulty": p["difficulty"], "file": puzzle_filename(p["id"]),
             "title": f"RWY {c.runway}, {local_clock(c.decision_time)}"}
            for p, c in zip(puzzles, chosen)
        ],
    }
    for puzzle in puzzles:
        write(puzzle, puzzle_filename(puzzle["id"]), out_dir)
    print("wrote", write(index, "puzzles.json", out_dir))

    # 4. bookkeeping: which time windows are used by puzzles (the ML model must NOT train on
    #    these), a summary for the README, and a check picture
    windows = pd.DataFrame([{
        "puzzle_id": p["id"],
        "start_s": p["source"]["t0_s"],
        "end_s": p["source"]["t0_s"] + SECONDS_BEFORE_DECISION + SECONDS_AFTER_DECISION,
    } for p in puzzles])
    windows.to_csv(os.path.join(DERIVED, "puzzle_windows.csv"), index=False)

    summary = {
        "candidates": len(candidates),
        "candidates_hold": sum(c.outcome.conflict for c in candidates),
        "puzzles": len(puzzles),
        "hold": sum(p["outcome"]["conflict"] for p in puzzles),
        "clear": sum(not p["outcome"]["conflict"] for p in puzzles),
        "real": sum("vehicle_time_shift_s" not in p["source"] for p in puzzles),
        "what_if": sum("vehicle_time_shift_s" in p["source"] for p in puzzles),
        "by_difficulty": {d: sum(p["difficulty"] == d for p in puzzles) for d in ("clear", "busy")},
        "busy_if_more_objects_than": busy_cutoff,
        "by_runway": {r: sum(c.runway == r for c in chosen) for r in sorted({c.runway for c in chosen})},
        "cv_ai_accuracy": round(np.mean([p["ai"]["call"] == p["outcome"]["correct_call"] for p in puzzles]), 3),
        "threshold_m": CONFLICT_THRESHOLD_M,
    }
    with open(os.path.join(DERIVED, "puzzles_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    plot_check(scene, puzzles, os.path.join(ROOT, "data", "puzzles_check.png"))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut game puzzles from the recording.")
    parser.add_argument("--out", default=PUZZLE_DIR_DEFAULT,
                        help="where puzzles.json + puzzle_*.json go (default web/public/data)")
    main(parser.parse_args().out)
