"""Unit tests for the puzzle generator's logic, on tiny hand-made inputs (no Amelia data needed).

Run from the repo root:  python -m pytest -q
"""
import numpy as np
import pandas as pd

from pipeline.puzzles import (DECISION_SPACING_S, Candidate, Outcome, Scene, choose_puzzles, seconds_to_reach,
                              space_out, vehicle_position)


def make_candidate(kind="what_if", decision_time=100.0, conflict=False, separation=300.0,
                   vehicle_id="V1", runway="15R/33L"):
    candidate = Candidate(kind=kind, runway=runway, zone_id="Z1", vehicle_id=vehicle_id,
                          decision_time=decision_time, go_time=150.0, crossing_end_time=180.0,
                          was_waiting=True)
    candidate.outcome = Outcome(conflict=conflict, min_separation_m=separation,
                                min_separation_time=decision_time, aircraft_id="A1")
    return candidate


def scene_with_one_vehicle():
    """Vehicle V1 waits at y = -100 until t = 150, then drives north at 5 m/s."""
    times = np.arange(0, 200, dtype=float)
    ys = np.where(times <= 150, -100.0, -100.0 + 5.0 * (times - 150))
    track = pd.DataFrame({"t": times, "id": "V1", "kind": "vehicle", "x": 0.0, "y": ys,
                          "speed": 0.0, "heading": 0.0}).set_index("t")
    return Scene(tracks=track.reset_index(), runways={}, zones={},
                 track_by_id={"V1": track}, kind_by_id={"V1": "vehicle"})


def test_real_puzzle_uses_the_recording():
    scene = scene_with_one_vehicle()
    real = make_candidate(kind="real", decision_time=150.0)
    assert vehicle_position(scene, real, 160.0)[1] == -100.0 + 5.0 * 10


def test_what_if_vehicle_replays_its_own_crossing_from_the_decision():
    scene = scene_with_one_vehicle()
    what_if = make_candidate(kind="what_if", decision_time=100.0)       # 50 s earlier than it really went
    assert what_if.time_shift == -50.0
    # before the decision: exactly the recording (still waiting)
    assert vehicle_position(scene, what_if, 90.0)[1] == -100.0
    # 10 s after the decision it is where it really was 10 s after it set off
    assert vehicle_position(scene, what_if, 110.0)[1] == -100.0 + 5.0 * 10


def test_space_out_drops_near_duplicates_but_keeps_hold_first():
    a = make_candidate(decision_time=100.0, conflict=False)
    b = make_candidate(decision_time=100.0 + DECISION_SPACING_S / 2, conflict=True)   # too close, rare HOLD
    c = make_candidate(decision_time=100.0 + 3 * DECISION_SPACING_S, conflict=False)
    kept = space_out([a, b, c])
    assert b in kept and a not in kept and c in kept


def test_choose_puzzles_keeps_every_hold_and_balances_what_if_tags():
    holds = [make_candidate(decision_time=t, conflict=True, separation=30.0) for t in (100, 120, 140)]
    what_if_clears = [make_candidate(decision_time=t, separation=200.0 + t) for t in range(200, 400, 20)]
    real_clears = [make_candidate(kind="real", decision_time=t, separation=400.0) for t in range(500, 700, 20)]
    chosen = choose_puzzles(holds + what_if_clears + real_clears)

    assert all(h in chosen for h in holds)
    what_if_chosen = [c for c in chosen if c.kind == "what_if"]
    # as many what-if CLEARs as what-if HOLDs, so the WHAT IF tag doesn't predict the answer
    assert sum(not c.outcome.conflict for c in what_if_chosen) == sum(c.outcome.conflict for c in what_if_chosen)


# ---- which aircraft count for a crossing: only those on / lined up with the crossed runway ----

from pipeline.ai_player import is_runway_user          # noqa: E402
from pipeline.truth import Runway                      # noqa: E402


def east_west_runway():
    """2 km runway along the x axis (heading 90)."""
    return Runway(name="9/27", center=np.array([0.0, 0.0]), u=np.array([1.0, 0.0]),
                  along_min=-1000, along_max=1000)


def test_aircraft_rolling_down_the_runway_counts_either_direction():
    rw = east_west_runway()
    assert is_runway_user(rw, 300.0, 5.0, heading=90.0, speed=60.0)
    assert is_runway_user(rw, 300.0, 5.0, heading=270.0, speed=60.0)


def test_aircraft_holding_in_position_for_takeoff_counts():
    assert is_runway_user(east_west_runway(), -950.0, 2.0, heading=90.0, speed=0.0)


def test_aircraft_on_final_approach_counts_but_a_slow_one_beyond_the_end_does_not():
    rw = east_west_runway()
    assert is_runway_user(rw, -3500.0, 20.0, heading=90.0, speed=70.0)       # landing
    assert not is_runway_user(rw, -1500.0, 20.0, heading=90.0, speed=8.0)    # taxiing past the end


def test_aircraft_turning_off_the_runway_does_not_count():
    # 45 m from the centreline at taxi speed: leaving onto a taxiway (puzzle t2035 at the freeze)
    assert not is_runway_user(east_west_runway(), 300.0, 45.0, heading=100.0, speed=7.0)


def test_aircraft_crossing_the_runway_on_a_taxiway_does_not_count():
    assert not is_runway_user(east_west_runway(), 300.0, 10.0, heading=0.0, speed=8.0)


def test_aircraft_on_a_parallel_taxiway_or_other_runway_does_not_count():
    rw = east_west_runway()
    assert not is_runway_user(rw, 300.0, 80.0, heading=90.0, speed=10.0)    # parallel taxiway, 80 m away
    assert not is_runway_user(rw, 300.0, -500.0, heading=90.0, speed=60.0)  # a different parallel runway


# ---- no freezes with an aircraft already inside the crossing square ----

from pipeline.puzzles import inside_polygon                         # noqa: E402


def test_inside_polygon():
    square = [[-60, -60], [60, -60], [60, 60], [-60, 60]]
    assert inside_polygon(0, 0, square)
    assert inside_polygon(37, -10, square)          # e.g. a jet rolling through the crossing
    assert not inside_polygon(120, 0, square)       # past the square
    assert not inside_polygon(0, -90, square)       # vehicle at the hold line


# ---- a freeze is only a puzzle if some aircraft can reach the crossing within the round ----

def test_jet_starting_its_takeoff_roll_can_reach_the_crossing():
    # barely moving (8 m/s) but on the runway 900 m away: speeding up at 2 m/s^2 gets there in ~26 s,
    # far sooner than ~112 s at its current speed
    assert seconds_to_reach(900.0, 8.0, on_runway=True) < 30
    assert seconds_to_reach(900.0, 8.0, on_runway=False) > 100


def test_slow_taxiing_aircraft_far_away_cannot():
    # 6 kt, 3 km down the runway (the old t2035 case): no way it arrives within 30 s
    assert seconds_to_reach(3000.0, 3.0, on_runway=True) > 30


def test_choose_puzzles_caps_holds_evenly_in_time_not_by_who_gets_them_right():
    holds = [make_candidate(decision_time=t, conflict=True, separation=30.0) for t in range(100, 400, 20)]   # 15
    clears = [make_candidate(kind="real", decision_time=t, separation=400.0) for t in range(500, 600, 20)]  # 5
    chosen = choose_puzzles(holds + clears)
    kept_holds = [c for c in chosen if c.outcome.conflict]
    assert len(kept_holds) == 5 and sum(not c.outcome.conflict for c in chosen) == 5      # 50/50
    times = [c.decision_time for c in kept_holds]
    assert times[0] == 100 and times[-1] == 380                                            # spread across the window
