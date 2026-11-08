"""Tests for the ML forecaster's building blocks (no trained model or Amelia data needed).

Run from the repo root:  python -m pytest -q
"""
import numpy as np
import pandas as pd

from pipeline import ml_model


def test_travel_frame_points_along_motion():
    forward, left = ml_model.travel_frame(np.array([0.0, 10.0]))      # moving north
    assert np.allclose(forward, [0, 1]) and np.allclose(left, [-1, 0])  # left of north = west


def test_parked_aircraft_gets_a_fixed_frame():
    forward, _ = ml_model.travel_frame(np.array([0.1, 0.0]))
    assert np.allclose(forward, [1, 0])


def test_correction_targets_are_in_the_aircraft_frame():
    state = np.array([0, 0, 0, 20.0, 0, 0])                            # moving north at 20 m/s
    steps = max(ml_model.HORIZONS_S)
    kalman_mean = np.zeros((steps, 2))
    true_future = np.tile([-5.0, 30.0], (steps, 1))                    # 30 m further ahead, 5 m to the left
    targets = ml_model.correction_targets(state, kalman_mean, true_future)
    assert targets["fwd_10"] == 30.0 and targets["left_10"] == 5.0


def test_interpolate_by_horizon_is_linear_from_zero():
    values = {h: float(h) for h in ml_model.HORIZONS_S}                # correction = 1 m per second
    per_second = ml_model.interpolate_by_horizon(values, 30, at_zero=0.0)
    assert np.allclose(per_second, np.arange(1, 31))


def test_purge_drops_training_slices_that_overlap_the_test_block():
    data = pd.DataFrame({"decision_time": [100.0, 160.0, 200.0, 400.0]})
    test = data.iloc[[2]]                                              # the slice at 200 s
    train = ml_model.purged_training_rows(data, test)
    # a slice covers 20 s before + 30 s after its decision, so slices < 50 s apart overlap:
    # 160 (40 s away) and 200 itself go; 100 (100 s away) and 400 stay
    assert sorted(train.decision_time) == [100.0, 400.0]


def test_no_model_means_no_ml_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_model, "REPORT_PATH", str(tmp_path / "missing.json"))
    ml_model.load_model.cache_clear()
    assert ml_model.forecast_ml([(0.0, 0.0, 0.0), (1.0, 10.0, 0.0)], 1.0, 30) is None
    ml_model.load_model.cache_clear()
