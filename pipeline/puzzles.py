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

Difficulty
----------
Each real moment is used ONCE, in one difficulty: about a third become FOG MODE (the player
and the AI only see degraded surveillance, see faults.py); the rest are BUSY HOUR when more
traffic is on screen than usual, else CLEAR DAY.

The AI players (constant velocity, Kalman filter, ML) live in ai_player.py; the human plays
against the ML player.
"""
import argparse
import json
import math
import os
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline import ai_player, faults, kalman, ml_model, truth
from pipeline.contracts import (
    ROOT,
    SCHEMA_VERSION,
    WEB_DATA_DIR,
    brier,
    p_safe_from,
    puzzle_filename,
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

FOG_SHARE = 1 / 3              # about a third of the moments become Fog mode (HOLD/CLEAR balanced)

# AI players (see ai_player.py). The puzzle's AI is MAIN_AI; every player's call is kept
# in ai.baselines for the results page.
AI_PLAYERS = ("cv", "kf", "ml")   # constant velocity, Kalman filter, learned correction (LightGBM)
MAIN_AI = "ml"

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
    difficulty: str = ""         # "fog" is decided before building; clear/busy after
    runway_traffic: set = field(default_factory=set)   # aircraft that used this runway during the puzzle

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
    aircraft_id: str             # the runway aircraft that came closest ("" if none was on the runway then)
    vehicle_on_runway: bool = False   # did the vehicle get onto the runway during the reveal at all?


@dataclass
class Scene:
    """Everything we need to look up positions quickly."""
    tracks: pd.DataFrame
    runways: dict                                       # name -> truth.Runway
    zones: dict                                         # zone id -> polygon (list of [x, y])
    bounds: dict = field(default_factory=dict)          # the map area the player sees (map.json bounds)
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

    # the map area the player sees = the background image's extent (same as map.json "bounds")
    background = truth.make_background(truth.load_reference(), out_dir=os.path.join(DERIVED, "tmp_bg"))
    bounds = background["bounds"] if background else {"xmin": -3300, "xmax": 3300, "ymin": -3100, "ymax": 3100}
    scene = Scene(tracks=tracks, runways=runways, zones=zones, bounds=bounds)
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


def runway_users_at(scene, runway, time):
    """Ids of aircraft on, or lined up with, `runway` at one recording second (see ai_player.is_runway_user)."""
    users = []
    for aircraft_id, kind in scene.kind_by_id.items():
        if kind != "aircraft":
            continue
        position = real_position(scene, aircraft_id, time)
        x, y, heading, speed = position if position is not None else (0, 0, 0, 0)
        if position is not None and ai_player.is_runway_user(runway, x, y, heading, speed):
            users.append(aircraft_id)
    return users


VISIBLE_AT_FREEZE_S = 3        # "visible at the freeze" = seen in the last 3 s before the decision


def inside_polygon(x, y, polygon):
    """Ray-casting test: is (x, y) inside the polygon (list of [x, y] corners)?"""
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def aircraft_in_zone_at_freeze(scene, candidate):
    """True if a runway aircraft is INSIDE the crossing zone at the freeze.

    The question is "can the vehicle cross now?"; asking it while a jet is sitting in (or
    rolling through) the crossing square is confusing and inconsistent with the other
    puzzles, which all freeze with the traffic before or clear of the square. Skip those."""
    polygon = scene.zones[candidate.zone_id]
    for aircraft_id in candidate.runway_traffic:
        position = real_position(scene, aircraft_id, candidate.decision_time)
        if position is not None and inside_polygon(position[0], position[1], polygon):
            return True
    return False


def can_matter_this_round(scene, candidate, aircraft_id):
    """Is this aircraft relevant to THIS round? At the freeze it must be inside the map area the
    player sees, and either already on the runway or close enough to reach the crossing within the
    reveal (SECONDS_AFTER_DECISION) at its current speed. An arrival 50 s out on a 3 km final is
    real traffic, but it can't affect a 30 s round and isn't even on the screen."""
    position = real_position(scene, aircraft_id, candidate.decision_time)
    if position is None:
        return False
    x, y, _, speed = position
    if not (scene.bounds["xmin"] <= x <= scene.bounds["xmax"] and scene.bounds["ymin"] <= y <= scene.bounds["ymax"]):
        return False
    runway = scene.runways[candidate.runway]
    along, _ = runway.to_runway_coords(x, y)
    if runway.along_min <= along <= runway.along_max:
        return True                                   # already on the runway
    zone_center = np.mean(scene.zones[candidate.zone_id], axis=0)
    distance = math.hypot(x - zone_center[0], y - zone_center[1])
    return distance <= speed * SECONDS_AFTER_DECISION


def runway_traffic_at_freeze(scene, candidate):
    """Aircraft on (or lined up with) the crossed runway that the player can SEE at the freeze
    and that can matter within this round. If there are none, the moment is dropped."""
    runway = scene.runways[candidate.runway]
    users = set()
    for second in np.arange(candidate.decision_time - VISIBLE_AT_FREEZE_S, candidate.decision_time + 1):
        users.update(runway_users_at(scene, runway, second))
    return {a for a in users if can_matter_this_round(scene, candidate, a)}


def measure_outcome(scene, candidate):
    """Closest approach between the vehicle and any aircraft ON OR LINED UP WITH the crossed runway,
    while the vehicle is on the runway, over the reveal window. (A plane parked on another runway or
    taxiing past doesn't count.) Positions are compared every 0.25 s, linearly interpolated between
    the 1 s reports, so a fast jet can't 'jump over' the vehicle between two samples."""
    runway = scene.runways[candidate.runway]
    start, end = candidate.decision_time, candidate.decision_time + SECONDS_AFTER_DECISION

    best = Outcome(conflict=False, min_separation_m=math.inf, min_separation_time=start, aircraft_id="")
    entered_runway = False
    for second in np.arange(start, end):
        vehicle_now = vehicle_position(scene, candidate, second)
        vehicle_next = vehicle_position(scene, candidate, second + 1)
        if vehicle_now is None or vehicle_next is None:
            continue
        if vehicle_is_on_runway(runway, vehicle_now[0], vehicle_now[1]):
            entered_runway = True

        for aircraft_id in runway_users_at(scene, runway, second):
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
    best.vehicle_on_runway = entered_runway
    return best


def closest_runway_user(scene, candidate):
    """Of the runway aircraft visible at the freeze, the one closest to the crossing at the freeze.
    Used as the 'question aircraft' when none of them got close during the vehicle's crossing."""
    zone_center = np.mean(scene.zones[candidate.zone_id], axis=0)
    best = (math.inf, None)
    for aircraft_id in sorted(candidate.runway_traffic):
        track = scene.track_by_id[aircraft_id]
        seen = track.loc[candidate.decision_time - VISIBLE_AT_FREEZE_S: candidate.decision_time]
        distance = float(np.hypot(seen.x - zone_center[0], seen.y - zone_center[1]).iloc[-1])
        best = min(best, (distance, aircraft_id))
    return best[1]


def closest_approach_to(scene, candidate, aircraft_id):
    """Closest the vehicle got to one aircraft during the reveal, wherever either of them was.
    Returns (distance_m, recording_time)."""
    best = (math.inf, candidate.decision_time)
    for second in np.arange(candidate.decision_time + 1, candidate.decision_time + SECONDS_AFTER_DECISION + 1):
        vehicle = vehicle_position(scene, candidate, second)
        aircraft = real_position(scene, aircraft_id, second)
        if vehicle is not None and aircraft is not None:
            best = min(best, (math.hypot(aircraft[0] - vehicle[0], aircraft[1] - vehicle[1]), second))
    return best


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

    # The question aircraft: the runway aircraft that came closest during the crossing, if the player
    # could see it at the freeze; otherwise the visible runway aircraft closest to the crossing.
    if outcome.aircraft_id in candidate.runway_traffic:
        aircraft_id = outcome.aircraft_id
    else:
        aircraft_id = closest_runway_user(scene, candidate)
    objects = agents_in_view(scene, candidate, aircraft_id)

    # Puzzle time starts at 0 = SECONDS_BEFORE_DECISION before the freeze.
    pre_frames, reveal_frames = [], []
    for step in range(0, SECONDS_BEFORE_DECISION + SECONDS_AFTER_DECISION + 1):
        recording_time = start + step
        is_pre = recording_time <= decision_recording_time
        objs = frame_objects(scene, candidate, objects, recording_time, with_status=is_pre)
        (pre_frames if is_pre else reveal_frames).append({"t": float(step), "objects": objs})

    decision_t = float(SECONDS_BEFORE_DECISION)
    puzzle_id = puzzle_id_for(candidate)
    seed = zlib.crc32(puzzle_id.encode())            # same puzzle -> same fog, same AI samples

    # Fog mode: the player AND the AI only get degraded surveillance before the freeze.
    if candidate.difficulty == "fog":
        pre_frames = faults.apply_fog(pre_frames, aircraft_id, seed, bounds=scene.bounds)

    # Every AI player answers from the same frames; MAIN_AI is the one the human plays against.
    label_for = lambda agent_id: object_label(agent_id, scene.kind_by_id[agent_id])
    answers = {
        name: ai_player.decide(name, pre_frames, objects, candidate.vehicle_id, runway,
                               CONFLICT_THRESHOLD_M, label_for, seed)
        for name in AI_PLAYERS
    }
    main_answer = answers[MAIN_AI]

    paths = dict(main_answer.paths)
    if aircraft_id not in paths:                      # contract: the question aircraft must have a path
        observations = faults.observations_from_frames(pre_frames, aircraft_id)
        forecast = ai_player.FORECASTERS[MAIN_AI](observations, decision_t, ai_player.HORIZON_S, seed)
        paths[aircraft_id] = (forecast.as_path() if forecast is not None
                              else [{"t": decision_t + 1, "x": 0.0, "y": 0.0, "r": 0.0}])

    vehicle_label = object_label(candidate.vehicle_id, "vehicle")
    aircraft_label = object_label(aircraft_id, "aircraft")
    where = "is holding short of" if candidate.was_waiting else "is approaching"
    question = f"{vehicle_label} {where} RWY {candidate.runway}. Clear it to cross now?"

    if outcome.aircraft_id == aircraft_id:
        # the usual case: separation to the question aircraft while the vehicle was on the runway
        sep, closest_time = outcome.min_separation_m, outcome.min_separation_time
        note = (f"Closest approach: {sep:.0f} m to {aircraft_label}, {closest_time - candidate.decision_time:.0f} s "
                f"after the decision ({'inside' if outcome.conflict else 'outside'} the "
                f"{CONFLICT_THRESHOLD_M:.0f} m limit).")
    else:
        # no runway aircraft was near while the vehicle was on the runway: report how close the
        # question aircraft got anyway, and say why it was safe
        sep, closest_time = closest_approach_to(scene, candidate, aircraft_id)
        if sep < CONFLICT_THRESHOLD_M:
            return None        # it got close OFF the runway: too ambiguous to score as CLEAR, skip it
        if outcome.vehicle_on_runway:
            why = f"{aircraft_label} was off the runway (or clear of the crossing) while {vehicle_label} crossed."
        else:
            why = f"{vehicle_label} didn't reach the runway in the next {SECONDS_AFTER_DECISION} s."
        note = f"Safe: {why} Closest it got to {aircraft_label}: {sep:.0f} m."
    if candidate.kind == "what_if":
        note += (f" What if: {vehicle_label} really waited another {-candidate.time_shift:.0f} s; here it "
                 f"crosses now, on its own real route. The aircraft are exactly as recorded.")

    source = {"airport": "KBOS", "recording": RECORDING, "t0_s": start}
    if candidate.kind == "what_if":
        source["vehicle_time_shift_s"] = candidate.time_shift

    return {
        "schema_version": SCHEMA_VERSION,
        "id": puzzle_id,
        "difficulty": candidate.difficulty or "clear",   # clear/busy set in label_difficulty()
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
            "detector": MAIN_AI,
            "call": main_answer.call,
            "p_safe": main_answer.p_safe,
            "confidence": main_answer.confidence,
            "paths": paths,
            "reasoning": main_answer.reasoning,
            "baselines": {
                name: {"call": a.call, "p_safe": a.p_safe, "confidence": a.confidence}
                for name, a in answers.items()
            },
        },
        "reveal": reveal_frames,
        "outcome": {
            "conflict": outcome.conflict,
            "threshold_m": CONFLICT_THRESHOLD_M,
            "min_sep_m": round(sep, 1),
            "min_sep_t": round(closest_time - start, 2),
            "correct_call": "HOLD" if outcome.conflict else "CLEAR",
            "note": note,
        },
    }


def assign_fog(chosen):
    """Mark about FOG_SHARE of the moments as Fog mode: every 3rd HOLD and every 3rd CLEAR
    (in time order), so Fog has both answers and each moment is used only once."""
    for answer in (True, False):
        same_answer = [c for c in chosen if c.outcome.conflict == answer]
        step = round(1 / FOG_SHARE)
        for candidate in same_answer[1::step]:
            candidate.difficulty = "fog"


def label_difficulty(puzzles):
    """Among the non-fog puzzles: the busier half (most objects on screen) is 'busy', the rest
    'clear'. Split by rank, not by "above the median", because many puzzles have the same
    count and a median cut would put almost all of them in one level.
    Returns the smallest object count labelled busy."""
    normal = sorted((p for p in puzzles if p["difficulty"] != "fog"),
                    key=lambda p: (len(p["objects"]), p["id"]))
    half = len(normal) // 2
    for rank, puzzle in enumerate(normal):
        puzzle["difficulty"] = "busy" if rank >= len(normal) - half else "clear"
    busy = [len(p["objects"]) for p in normal if p["difficulty"] == "busy"]
    return min(busy) if busy else 0


# --------------------------------------------------------------------------------------
# Check picture
# --------------------------------------------------------------------------------------

def plot_check(scene, puzzles, out_png, how_many=8):
    """Small multiples: pre paths (thin), reveal paths (thick), zone (gold), closest approach (x).
    Lets a human confirm that HOLD puzzles really show a jet passing the crossing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # a mix: HOLD and CLEAR, with fog puzzles included
    fog = [p for p in puzzles if p["difficulty"] == "fog"]
    picks = [p for p in fog if p["outcome"]["conflict"]][:2] + [p for p in fog if not p["outcome"]["conflict"]][:1]
    picks += [p for p in puzzles if p["outcome"]["conflict"] and p not in picks][: how_many // 2 - 1]
    picks += [p for p in puzzles if not p["outcome"]["conflict"] and p not in picks][: how_many - len(picks)]
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
        # the AI's forecast for the question aircraft: dotted mean path + cone circles every 5 s
        ai_path = puzzle["ai"]["paths"].get(puzzle["aircraft_id"], [])
        if ai_path:
            xy = np.array([(p["x"], p["y"]) for p in ai_path])
            ax.plot(xy[:, 0], xy[:, 1], "b:", lw=1)
            for p in ai_path[4::5]:
                ax.add_patch(plt.Circle((p["x"], p["y"]), p["r"], color="tab:blue", fill=False, lw=0.6))

        # fog: where the player last SAW the aircraft (the ghost)
        last_seen = [o for o in puzzle["pre"][-1]["objects"] if o["id"] == puzzle["aircraft_id"]]
        if last_seen and last_seen[0]["status"] != "ok":
            ax.plot(last_seen[0]["x"], last_seen[0]["y"], "o", mfc="none", mec="k", ms=9)
        center = zone[:-1].mean(axis=0)
        ax.set_xlim(center[0] - 700, center[0] + 700)
        ax.set_ylim(center[1] - 700, center[1] + 700)
        ax.set_aspect("equal")
        tag = "WHAT IF" if "vehicle_time_shift_s" in puzzle["source"] else "REAL"
        ax.set_title(f"{puzzle['id']} [{tag}, {puzzle['difficulty']}]\n{puzzle['outcome']['correct_call']} · sep "
                     f"{puzzle['outcome']['min_sep_m']:.0f} m · AI ({puzzle['ai']['detector']}) {puzzle['ai']['call']} "
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
    {"id": "fog", "name": "Fog mode",
     "description": "Surveillance is stale or missing. You only see where things were."},
]


def ai_scorecard(puzzles):
    """How each AI player did, per difficulty: accuracy and Brier score (snapped to 50/75/95,
    the same rules the human plays by). Lower Brier is better."""
    card = {}
    for name in AI_PLAYERS:
        card[name] = {}
        for difficulty in ("clear", "busy", "fog", "all"):
            group = [p for p in puzzles if difficulty in ("all", p["difficulty"])]
            if not group:
                continue
            calls = [(p["ai"]["baselines"][name], p["outcome"]["conflict"]) for p in group]
            card[name][difficulty] = {
                "n": len(group),
                "accuracy": round(float(np.mean([(c["call"] == "HOLD") == conflict for c, conflict in calls])), 3),
                "brier": round(float(np.mean([brier(p_safe_from(c["call"], c["confidence"]), conflict)
                                              for c, conflict in calls])), 4),
            }
    return card


def main(out_dir=PUZZLE_DIR_DEFAULT, retrain=True):
    scene, crossings = load_scene()

    # 1. every possible decision moment, with what really happened next
    candidates = make_candidates(scene, crossings)
    candidates = [c for c in candidates                       # the player must see the vehicle at the freeze
                  if vehicle_position(scene, c, c.decision_time) is not None]
    # no aircraft on (or lined up with) the runway visible at the freeze = not a real decision: drop it
    for candidate in candidates:
        candidate.runway_traffic = runway_traffic_at_freeze(scene, candidate)
    empty_runway = [c for c in candidates if not c.runway_traffic]
    candidates = [c for c in candidates if c.runway_traffic]

    # an aircraft already inside the crossing square at the freeze = confusing question: drop it
    occupied_zone = [c for c in candidates if aircraft_in_zone_at_freeze(scene, c)]
    candidates = [c for c in candidates if c not in occupied_zone]

    for candidate in candidates:
        candidate.outcome = measure_outcome(scene, candidate)

    # a HOLD is only fair if the player could see the aircraft that caused it
    unfair_hold = [c for c in candidates if c.outcome.conflict and c.outcome.aircraft_id not in c.runway_traffic]
    candidates = [c for c in candidates if c not in unfair_hold]
    print(f"{len(candidates)} candidates with runway traffic at the freeze "
          f"({len(empty_runway)} empty-runway moments dropped, {len(occupied_zone)} with an aircraft already in the "
          f"crossing square dropped, {len(unfair_hold)} HOLDs with an unseen aircraft dropped), "
          f"{sum(c.outcome.conflict for c in candidates)} where HOLD was right")

    # 2. thin out near-duplicates, balance HOLD vs CLEAR, pick the Fog moments
    chosen = choose_puzzles(space_out(candidates))
    assign_fog(chosen)

    # 3. the time windows used by puzzles. The Kalman filter and the ML model are tuned/trained
    #    on every OTHER part of the recording, never on the puzzles' own answers.
    windows = pd.DataFrame([{
        "puzzle_id": puzzle_id_for(c),
        "start_s": c.decision_time - SECONDS_BEFORE_DECISION,
        "end_s": c.decision_time + SECONDS_AFTER_DECISION,
    } for c in chosen])
    windows.to_csv(os.path.join(DERIVED, "puzzle_windows.csv"), index=False)
    excluded = windows[["start_s", "end_s"]].values.tolist()
    if retrain or not os.path.exists(kalman.PARAMS_PATH):
        kalman.tune(scene.tracks, excluded)
    if retrain or ml_model.load_model() is None:
        print("training the ML model (~2 min)...")
        ml_model.train_and_save(scene.tracks, excluded)

    # 4. build, label difficulty, write (every file is validated against the contract)
    built = [(build_puzzle(scene, c), c) for c in chosen]
    skipped_ambiguous = sum(p is None for p, _ in built)
    chosen = [c for p, c in built if p is not None]
    puzzles = [p for p, _ in built if p is not None]
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
    # remove puzzle files from earlier runs that aren't in this set any more (the UI only
    # loads what puzzles.json lists, but stale files would confuse people and the tests)
    keep = {puzzle_filename(p["id"]) for p in puzzles}
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name.startswith("puzzle_") and name.endswith(".json") and name not in keep:
                os.remove(os.path.join(out_dir, name))
    for puzzle in puzzles:
        write(puzzle, puzzle_filename(puzzle["id"]), out_dir)
    print("wrote", write(index, "puzzles.json", out_dir))

    # 5. summary for the README / results page, and a check picture
    summary = {
        "candidates": len(candidates),
        "dropped_empty_runway": len(empty_runway),
        "dropped_aircraft_in_zone": len(occupied_zone),
        "dropped_unfair_hold": len(unfair_hold),
        "dropped_close_off_runway": skipped_ambiguous,
        "candidates_hold": sum(c.outcome.conflict for c in candidates),
        "puzzles": len(puzzles),
        "hold": sum(p["outcome"]["conflict"] for p in puzzles),
        "clear": sum(not p["outcome"]["conflict"] for p in puzzles),
        "real": sum("vehicle_time_shift_s" not in p["source"] for p in puzzles),
        "what_if": sum("vehicle_time_shift_s" in p["source"] for p in puzzles),
        "by_difficulty": {
            d: {"puzzles": sum(p["difficulty"] == d for p in puzzles),
                "hold": sum(p["difficulty"] == d and p["outcome"]["conflict"] for p in puzzles)}
            for d in ("clear", "busy", "fog")
        },
        "busy_if_more_objects_than": busy_cutoff,
        "by_runway": {r: sum(c.runway == r for c in chosen) for r in sorted({c.runway for c in chosen})},
        "threshold_m": CONFLICT_THRESHOLD_M,
        "main_ai": MAIN_AI,
        "ai_scorecard": ai_scorecard(puzzles),
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
    parser.add_argument("--skip-training", action="store_true",
                        help="reuse the saved Kalman settings and ML model instead of re-training (~2 min)")
    args = parser.parse_args()
    main(args.out, retrain=not args.skip_training)
