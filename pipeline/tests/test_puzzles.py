"""Unit tests for the puzzle generator's logic, on tiny hand-made inputs (no Amelia data needed).

Run from the repo root:  python -m pytest -q
"""
import numpy as np
import pandas as pd

from pipeline.puzzles import Candidate, Outcome, Scene, choose_puzzles, space_out, vehicle_position


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
    b = make_candidate(decision_time=105.0, conflict=True)               # 5 s later, rare HOLD
    c = make_candidate(decision_time=120.0, conflict=False)
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
    assert is_runway_user(rw, 300.0, 5.0, heading=90.0)
    assert is_runway_user(rw, 300.0, 5.0, heading=270.0)


def test_aircraft_on_final_approach_counts():
    assert is_runway_user(east_west_runway(), -3500.0, 20.0, heading=90.0)


def test_aircraft_crossing_the_runway_on_a_taxiway_does_not_count():
    assert not is_runway_user(east_west_runway(), 300.0, 10.0, heading=0.0)


def test_aircraft_on_a_parallel_taxiway_or_other_runway_does_not_count():
    rw = east_west_runway()
    assert not is_runway_user(rw, 300.0, 180.0, heading=90.0)     # parallel taxiway, 180 m away
    assert not is_runway_user(rw, 300.0, -500.0, heading=90.0)    # a different parallel runway
