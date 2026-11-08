"""Tests for the Kalman forecaster and Fog-mode degradation, on hand-made tracks.

Run from the repo root:  python -m pytest -q
"""
import numpy as np

from pipeline import ai_player, kalman
from pipeline.faults import apply_fog, observations_from_frames

PARAMS = kalman.KalmanParams()     # defaults, so the tests don't depend on tuned settings


def straight_line(speed=20.0, seconds=20):
    """An aircraft moving east at a constant speed, one report per second."""
    return [(float(t), speed * t, 0.0) for t in range(seconds + 1)]


def speeding_up(accel=2.0, seconds=20, start_speed=5.0):
    """An aircraft starting a takeoff roll: constant acceleration along x."""
    return [(float(t), start_speed * t + 0.5 * accel * t * t, 0.0) for t in range(seconds + 1)]


def test_straight_line_is_predicted_almost_exactly():
    f = kalman.forecast(straight_line(), decision_time=20.0, steps=10, params=PARAMS)
    expected_x = 20.0 * 30                                        # 10 s after the decision
    assert abs(f.mean[-1, 0] - expected_x) < 15
    assert abs(f.mean[-1, 1]) < 5


def test_kalman_beats_constant_velocity_on_a_takeoff_roll():
    observations = speeding_up()
    truth_x = 5.0 * 30 + 0.5 * 2.0 * 30 * 30                      # where it really is at t = 30
    kf = kalman.forecast(observations, 20.0, steps=10, params=PARAMS)
    cv = ai_player.forecast_constant_velocity(observations, 20.0, steps=10, seed=0)
    assert abs(kf.mean[-1, 0] - truth_x) < abs(cv.mean[-1, 0] - truth_x)
    assert kf.accel_along_track > 0.5                             # it noticed the acceleration


def test_missing_reports_widen_the_cone():
    full = straight_line()
    with_gap = [o for o in full if o[0] <= 10]                    # last 10 s of reports missing
    f_full = kalman.forecast(full, 20.0, steps=10, params=PARAMS)
    f_gap = kalman.forecast(with_gap, 20.0, steps=10, params=PARAMS)
    assert f_gap.radius[0] > 2 * f_full.radius[0]


def test_never_reverses_or_exceeds_max_speed():
    braking = [(float(t), 30 * t - 1.5 * t * t, 0.0) for t in range(11)]   # nearly stopped at t = 10
    f = kalman.forecast(braking, 10.0, steps=30, params=PARAMS)
    assert np.all(np.diff(f.mean[:, 0]) > -1.0)                   # doesn't roll backwards
    fast = kalman.forecast(straight_line(speed=120.0), 20.0, steps=10, params=PARAMS)
    assert fast.speed_now <= kalman.MAX_SPEED_MS + 1e-6


def frames_for(objects_by_time):
    """Pre frames from {t: [(id, x, y), ...]}."""
    return [{"t": float(t), "objects": [{"id": i, "x": x, "y": y, "heading": 90.0, "speed": 10.0,
                                         "age_s": 0, "status": "ok"} for i, x, y in objs]}
            for t, objs in sorted(objects_by_time.items())]


def test_fog_freezes_the_question_aircraft_and_delays_the_rest():
    frames = frames_for({t: [("A1", 10.0 * t, 0.0), ("V1", 0.0, -100.0 + t)] for t in range(21)})
    fog = apply_fog(frames, "A1", seed=1)

    last = {o["id"]: o for o in fog[-1]["objects"]}
    assert last["A1"]["status"] == "lost" and last["A1"]["age_s"] >= 6
    assert 2 <= last["V1"]["age_s"] <= 3 and last["V1"]["status"] == "stale"
    # the ghost doesn't move while the aircraft is lost
    ghost = [(o["x"], o["y"]) for f in fog[-5:] for o in f["objects"] if o["id"] == "A1"]
    assert len(set(ghost)) == 1


def test_fog_is_reproducible_and_leaves_the_original_alone():
    frames = frames_for({t: [("A1", 10.0 * t, 0.0)] for t in range(21)})
    assert apply_fog(frames, "A1", seed=7) == apply_fog(frames, "A1", seed=7)
    assert frames[-1]["objects"][0]["status"] == "ok"


def test_observations_collapse_repeated_stale_reports():
    frames = frames_for({t: [("A1", 10.0 * t, 0.0)] for t in range(21)})
    fog = apply_fog(frames, "A1", seed=3)
    observations = observations_from_frames(fog, "A1")
    times = [t for t, _, _ in observations]
    assert times == sorted(set(times))                            # one observation per real report
    assert times[-1] <= 20 - 6                                    # nothing after the dropout began
