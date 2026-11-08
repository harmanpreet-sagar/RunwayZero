"""Unit tests for the truth layer's geometry, on tiny hand-made inputs (no Amelia data needed).

Run from the repo root:  python -m pytest -q
"""
import numpy as np
import pandas as pd

from pipeline.truth import Runway, chain_segments, find_crossings, to_contract_xy


def east_west_runway():
    """A 2 km runway along the x axis (heading 90), centred on the origin."""
    return Runway(name="9/27", center=np.array([0.0, 0.0]), u=np.array([1.0, 0.0]),
                  along_min=-1000, along_max=1000)


def vehicle_track(vid, xs, ys, t0=0.0):
    """Build a tracks DataFrame for one vehicle, one point per second."""
    n = len(xs)
    return pd.DataFrame({"t": t0 + np.arange(n, dtype=float), "id": vid, "kind": "vehicle",
                         "x": xs, "y": ys, "speed": 8.0, "heading": 0.0, "alt_ft": 12.5, "on_ground": True})


def test_amelia_axes_are_swapped_and_scaled():
    # Amelia: x = north (km), y = east (km)  ->  contract: x = east (m), y = north (m)
    x, y = to_contract_xy(1.0, 2.0)
    assert (x, y) == (2000.0, 1000.0)


def test_runway_coords_round_trip():
    rw = Runway(name="t", center=np.array([100.0, 50.0]), u=np.array([0.6, 0.8]))
    a, p = rw.to_runway_coords(400.0, 0.0)
    x, y = rw.to_world(a, p)
    assert np.allclose([x, y], [400.0, 0.0])


def test_driving_straight_across_is_a_crossing():
    # south -> north across the runway at x = 300, 8 m/s
    ys = np.arange(-150, 151, 8.0)
    tracks = vehicle_track("V1", np.full(len(ys), 300.0), ys)
    c = find_crossings(tracks, [east_west_runway()])
    assert len(c) == 1
    row = c.iloc[0]
    assert row.runway == "9/27" and row.from_side == "right"      # came from the south = right of heading 90
    assert abs(row.along_m - 300) < 1
    assert row.duration_s < 20


def test_turning_back_is_not_a_crossing():
    # drives onto the runway and returns to the same side
    ys = np.r_[np.arange(-150, 0, 8.0), np.arange(0, -151, -8.0)]
    tracks = vehicle_track("V1", np.full(len(ys), 0.0), ys)
    assert find_crossings(tracks, [east_west_runway()]).empty


def test_driving_along_the_runway_is_not_a_crossing():
    # enters from the south, drives 1.5 km along the runway (way over MAX_CROSSING_S), exits north
    ys = np.r_[np.arange(-150, 0, 8.0), np.zeros(200), np.arange(0, 151, 8.0)]
    xs = np.r_[np.full(19, -800.0), np.linspace(-800, 700, 200), np.full(19, 700.0)]
    tracks = vehicle_track("V1", xs, ys)
    assert find_crossings(tracks, [east_west_runway()]).empty


def test_crossing_beyond_the_runway_end_is_ignored():
    # crosses the extended centreline 500 m past the runway's end: not a runway crossing
    ys = np.arange(-150, 151, 8.0)
    tracks = vehicle_track("V1", np.full(len(ys), 1500.0), ys)
    assert find_crossings(tracks, [east_west_runway()]).empty


def test_chain_segments_joins_a_straight_line():
    a = np.array([[0, 0], [10, 0], [20, 0]], dtype=float)
    b = np.array([[10, 0], [20, 0], [30, 0]], dtype=float)
    chains = chain_segments(a, b)
    assert len(chains) == 1 and len(chains[0]) == 4
