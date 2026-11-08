"""Kalman-filter forecaster for aircraft: the AI player's second brain (after constant velocity).

What it does
------------
Given the positions we've SEEN of an aircraft (possibly with gaps, possibly stale), it
estimates where the aircraft is, how fast it's going and how hard it's speeding up or
slowing down, and then forecasts the next ~30 s as a cloud of possible futures.

Why constant ACCELERATION
-------------------------
The constant-velocity AI's worst misses are jets starting a takeoff roll (slow at the
freeze, fast 10 s later) and jets braking after landing. A filter that also tracks
acceleration sees both coming. To stop it from extrapolating forever, the acceleration
fades away over a few seconds (time constant ACCEL_FADE_S) and speeds are kept physical.

Uncertainty
-----------
The filter's covariance grows with every second it predicts without a new report, so
stale or missing data (Fog mode) automatically widens the cone. We forecast by sampling
SAMPLES possible futures; the cone radius at each second is the 90th percentile of how
far those futures spread around their mean. A single scale factor (cone_scale) is fitted
on real tracks so that the cone really contains the truth ~90% of the time.

    python -m pipeline.kalman     # tune on tracks outside the puzzle windows, print a report
"""
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from pipeline import truth

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

DT = 1.0                       # one step = one second (the recording's report rate)
MAX_SPEED_MS = 90.0            # ~175 kt: no aircraft on or near the runway goes faster
SAMPLES = 400                  # possible futures drawn per aircraft forecast
CONE_PERCENTILE = 90           # the cone radius covers this share of the possible futures

PARAMS_PATH = os.path.join(truth.DERIVED, "kalman_params.json")


@dataclass
class KalmanParams:
    """Knobs of the filter. Defaults are sensible; `tune()` picks better ones from data."""
    jerk_noise: float = 0.5        # how quickly acceleration is allowed to change (m/s^3)
    accel_fade_s: float = 6.0      # acceleration fades with this time constant (seconds)
    measurement_noise_m: float = 5.0   # how noisy a single position report is
    cone_scale: float = 1.0        # multiplies the forecast spread so the cone is calibrated


# --------------------------------------------------------------------------------------
# Matrices
# --------------------------------------------------------------------------------------
# State vector, per aircraft: [x, y, vx, vy, ax, ay]  (metres, m/s, m/s^2)

def transition_matrix(params, dt=DT):
    """How the state moves forward one step: position += v*dt + a*dt^2/2, v += a*dt, a fades."""
    fade = math.exp(-dt / params.accel_fade_s)
    one_axis = np.array([
        [1.0, dt, 0.5 * dt * dt],
        [0.0, 1.0, dt],
        [0.0, 0.0, fade],
    ])
    return _interleave(one_axis)


def process_noise(params, dt=DT):
    """Uncertainty added each step: random 'jerk' (change of acceleration), same on both axes."""
    q = params.jerk_noise ** 2
    one_axis = q * np.array([
        [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
        [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
        [dt ** 3 / 6, dt ** 2 / 2, dt],
    ])
    return _interleave(one_axis)


def _interleave(one_axis):
    """Turn a 3x3 per-axis matrix (p, v, a) into the 6x6 matrix for state [x, y, vx, vy, ax, ay]."""
    full = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            full[2 * i, 2 * j] = one_axis[i, j]          # x-axis block
            full[2 * i + 1, 2 * j + 1] = one_axis[i, j]  # y-axis block
    return full


MEASURE = np.array([[1, 0, 0, 0, 0, 0],                   # we only ever observe position
                    [0, 1, 0, 0, 0, 0]], dtype=float)


# --------------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------------

def step_forward(states, F, noise=None, travel_direction=None, stopped=None):
    """Move one or many states (shape (6,) or (N, 6)) one second ahead, keeping them physical.

    noise             random jerk to add this step (sampled futures only)
    travel_direction  unit vector of the aircraft's direction of travel at the decision moment
    stopped           boolean mask of sampled futures that have already come to a stop

    Rules, applied AFTER the noise so randomness can't sneak past them:
      * aircraft don't roll backwards: if the velocity ends up pointing against the direction
        of travel (or flips relative to the previous step), the aircraft has STOPPED,
      * a stopped aircraft stays stopped (otherwise random jitter would ratchet it forward),
      * no faster than MAX_SPEED_MS.
    Returns (new_states, stopped)."""
    old_velocity = states[..., 2:4].copy()
    new = states @ F.T
    if noise is not None:
        new = new + noise

    velocity = new[..., 2:4]                    # views into `new`: editing them edits `new`
    accel = new[..., 4:6]

    reference = old_velocity if travel_direction is None else travel_direction
    going_backwards = (velocity * reference).sum(axis=-1, keepdims=True) < 0
    stopped = going_backwards if stopped is None else (stopped | going_backwards)
    velocity[:] = np.where(stopped, 0.0, velocity)
    accel[:] = np.where(stopped, 0.0, accel)

    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    too_fast = speed > MAX_SPEED_MS
    velocity[:] = np.where(too_fast, velocity * MAX_SPEED_MS / np.maximum(speed, 1e-9), velocity)
    return new, stopped


def run_filter(observations, until_time, params):
    """Filter a list of observations [(time, x, y), ...] (sorted, possibly with gaps) and
    predict forward to `until_time`. Returns (state, covariance) at `until_time`, or None
    if there are fewer than two observations to work with."""
    if len(observations) < 2:
        return None

    (t0, x0, y0), (t1, x1, y1) = observations[0], observations[1]
    gap = max(t1 - t0, DT)
    state = np.array([x0, y0, (x1 - x0) / gap, (y1 - y0) / gap, 0.0, 0.0])
    covariance = np.diag([25.0, 25.0, 25.0, 25.0, 4.0, 4.0])      # unsure about speed/accel at first

    F, Q = transition_matrix(params), process_noise(params)
    R = np.eye(2) * params.measurement_noise_m ** 2
    reports = {round(t): (x, y) for t, x, y in observations}

    time = t0
    while time < until_time:
        # predict one second ahead
        state, _ = step_forward(state, F)
        covariance = F @ covariance @ F.T + Q
        time += DT

        # correct with a report, if one exists for this second
        if round(time) in reports:
            measured = np.array(reports[round(time)])
            innovation = measured - MEASURE @ state
            innovation_cov = MEASURE @ covariance @ MEASURE.T + R
            gain = covariance @ MEASURE.T @ np.linalg.inv(innovation_cov)
            state = state + gain @ innovation
            covariance = (np.eye(6) - gain @ MEASURE) @ covariance

    state, _ = step_forward(state, np.eye(6))           # a correction can overshoot the speed cap: re-apply it
    return state, covariance


@dataclass
class Forecast:
    """A forecast for one aircraft: mean path + cone, and the sampled possible futures."""
    times: np.ndarray            # (steps,) seconds, puzzle time
    mean: np.ndarray             # (steps, 2) mean position
    radius: np.ndarray           # (steps,) 90% cone radius
    samples: np.ndarray          # (SAMPLES, steps, 2) possible future positions
    speed_now: float             # estimated speed at the decision moment (m/s)
    accel_along_track: float     # estimated acceleration along the direction of travel (m/s^2)

    def as_path(self):
        """Contract format: [{t, x, y, r}, ...]."""
        path = []
        for time, (x, y), radius in zip(self.times, self.mean, self.radius):
            path.append({
                "t": float(time),
                "x": round(float(x), 1),
                "y": round(float(y), 1),
                "r": round(float(radius), 1),
            })
        return path


def forecast(observations, decision_time, steps, params, seed=0):
    """Forecast `steps` seconds after `decision_time` from observations [(time, x, y), ...].
    Returns a Forecast, or None if the aircraft hasn't been seen enough."""
    filtered = run_filter(observations, decision_time, params)
    if filtered is None:
        return None
    state, covariance = filtered

    rng = np.random.default_rng(seed)
    F, Q = transition_matrix(params), process_noise(params)

    # direction of travel at the decision moment (None if it's (almost) parked: then it may go anywhere)
    speed_now = float(np.linalg.norm(state[2:4]))
    direction = state[2:4] / speed_now if speed_now > 1.0 else None

    # draw possible current states, then push each one forward with random jerk
    current = rng.multivariate_normal(state, _symmetric(covariance), size=SAMPLES)
    stopped = np.zeros((SAMPLES, 1), dtype=bool)
    if direction is not None:                       # a sampled state can't start out going backwards either
        current, stopped = step_forward(current, np.eye(6), travel_direction=direction)
    jitter_chol = np.linalg.cholesky(Q + np.eye(6) * 1e-9)
    future = np.empty((SAMPLES, steps, 2))
    for step in range(steps):
        noise = rng.standard_normal((SAMPLES, 6)) @ jitter_chol.T
        current, stopped = step_forward(current, F, noise=noise, travel_direction=direction, stopped=stopped)
        future[:, step, :] = current[:, 0:2]

    mean = future.mean(axis=0)
    # spread the samples around their mean by cone_scale, so the 90% cone is calibrated
    future = mean + params.cone_scale * (future - mean)
    radius = np.percentile(np.linalg.norm(future - mean, axis=-1), CONE_PERCENTILE, axis=0)

    velocity, accel = state[2:4], state[4:6]
    speed = float(np.linalg.norm(velocity))
    along = float(accel @ velocity / speed) if speed > 0.5 else 0.0
    times = decision_time + DT * np.arange(1, steps + 1)
    return Forecast(times=times, mean=mean, radius=radius, samples=future, speed_now=speed, accel_along_track=along)


def _symmetric(matrix):
    """Numerical safety: force a covariance matrix to be exactly symmetric."""
    return (matrix + matrix.T) / 2


# --------------------------------------------------------------------------------------
# Tuning on real tracks (outside the puzzle windows)
# --------------------------------------------------------------------------------------

GAP_SECONDS = (6, 12)          # like Fog mode: the last reports before the decision are missing


def make_test_cases(tracks, excluded_windows, history_s=20, horizon_s=20, every_s=10, seed=0):
    """Slices of real aircraft tracks: `history_s` seconds seen, then `horizon_s` seconds to predict.
    Slices that overlap a puzzle window are skipped (the game must not be tuned on its own answers).

    Every slice is used twice: once with all reports (like Clear day) and once with the last
    6-12 s of reports missing (like Fog mode), so the cone is calibrated for BOTH situations.
    Each case is (observations, decision_time, true_future, has_gap)."""
    rng = np.random.default_rng(seed)
    aircraft = tracks[(tracks.kind == "aircraft") & (tracks.alt_ft <= 1500)]
    cases = []
    for _, track in aircraft.groupby("id"):
        track = track.set_index("t")
        times = track.index.values
        for start in np.arange(times.min(), times.max() - history_s - horizon_s, every_s):
            end = start + history_s + horizon_s
            if any(a < end and start < b for a, b in excluded_windows):
                continue
            history = track.loc[start:start + history_s]
            future = track.loc[start + history_s + 1:end]
            if len(history) < history_s or len(future) < horizon_s or history.speed.max() < 3:
                continue                              # skip parked aircraft: nothing to learn
            observations = [(t, r.x, r.y) for t, r in history.iterrows()]
            decision_time = start + history_s
            true_future = future[["x", "y"]].values[:horizon_s]
            cases.append((observations, decision_time, true_future, False))

            gap = int(rng.integers(GAP_SECONDS[0], GAP_SECONDS[1] + 1))
            with_gap = [o for o in observations if o[0] <= decision_time - gap]
            cases.append((with_gap, decision_time, true_future, True))
    return cases


def evaluate(cases, params, horizons=(5, 10, 20)):
    """Mean forecast error at a few horizons, and how often the truth was inside the 90% cone,
    reported separately for clean histories and gappy (fog-like) ones.
    Also returns error / cone-radius at every horizon (used to calibrate the cone)."""
    errors = {(h, gap): [] for h in horizons for gap in (False, True)}
    ratios = []                                       # error / cone radius
    covered = {False: [], True: []}
    for i, (observations, decision_time, truth_future, has_gap) in enumerate(cases):
        f = forecast(observations, decision_time, steps=max(horizons), params=params, seed=i)
        if f is None:
            continue
        for h in horizons:
            error = float(np.linalg.norm(f.mean[h - 1] - truth_future[h - 1]))
            errors[(h, has_gap)].append(error)
            ratios.append(error / max(f.radius[h - 1], 1e-6))
            covered[has_gap].append(error <= f.radius[h - 1])

    report = {}
    for gap, name in ((False, "clean"), (True, "gap")):
        for h in horizons:
            report[f"{name}_error_{h}s_m"] = round(float(np.mean(errors[(h, gap)])), 1)
        report[f"{name}_coverage"] = round(float(np.mean(covered[gap])), 3)
    return report, np.array(ratios)


def tune(tracks, excluded_windows, verbose=True):
    """Pick the filter settings with the lowest 10 s error on real tracks (clean + gappy), then fit
    cone_scale so the cone contains the truth 90% of the time. Saves data/derived/kalman_params.json."""
    cases = make_test_cases(tracks, excluded_windows)
    rng = np.random.default_rng(0)
    if len(cases) > 300:                              # keep tuning quick
        cases = [cases[i] for i in sorted(rng.choice(len(cases), 300, replace=False))]

    best = None
    for jerk in (0.2, 0.5, 1.0, 2.0):
        for fade in (3.0, 6.0, 12.0):
            for noise in (3.0, 8.0):
                params = KalmanParams(jerk_noise=jerk, accel_fade_s=fade, measurement_noise_m=noise)
                report, _ = evaluate(cases, params)
                score = report["clean_error_10s_m"] + report["gap_error_10s_m"]
                if best is None or score < best[1]:
                    best = (params, score)
    params, _ = best

    # calibrate: scale the spread so 90% of real errors (clean AND gappy, all horizons) fall inside
    _, ratios = evaluate(cases, params)
    params.cone_scale = float(np.quantile(ratios, 0.9))
    report, _ = evaluate(cases, params)

    os.makedirs(truth.DERIVED, exist_ok=True)
    with open(PARAMS_PATH, "w") as fh:
        json.dump({"params": asdict(params), "report": report, "test_cases": len(cases)}, fh, indent=2)
    load_params.cache_clear()                         # make the next forecast use the new settings
    if verbose:
        print(f"Kalman tuned on {len(cases)} real track slices: {asdict(params)}")
        print(f"  {report}")
    return params, report


@lru_cache(maxsize=1)
def load_params():
    """Tuned settings if `tune()` has run, else the defaults. (Cached: read the file once.)"""
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as fh:
            return KalmanParams(**json.load(fh)["params"])
    return KalmanParams()


if __name__ == "__main__":
    tracks = pd.read_csv(os.path.join(truth.DERIVED, "tracks.csv.gz"), dtype={"id": str})
    windows_path = os.path.join(truth.DERIVED, "puzzle_windows.csv")
    windows = pd.read_csv(windows_path)[["start_s", "end_s"]].values.tolist() if os.path.exists(windows_path) else []
    tune(tracks, windows)
