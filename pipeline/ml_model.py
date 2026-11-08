"""The ML player's forecaster: a LightGBM model that learns to CORRECT the Kalman filter.

    python -m pipeline.ml_model      # build the dataset, cross-validate, train, save, report

Idea
----
The Kalman filter (kalman.py) is good at smoothing noisy reports and at knowing how unsure
it is, but it only knows physics: "keep doing what you were doing, a bit less every second".
It can't know that a slow jet lined up at the end of an active runway is about to start
its takeoff roll, or that a landing jet is about to brake and turn off.
Those are PATTERNS, and patterns are what a learned model can pick up from data.

So the model predicts the Kalman filter's ERROR: for an aircraft at the decision moment,
"how far ahead / behind and how far left / right of the Kalman forecast will it really be,
5, 10, ... 30 s from now?". ML forecast = Kalman forecast + predicted correction.

The correction is expressed in the aircraft's own frame (along its direction of travel and
sideways), so the model learns "it'll be 40 m further ahead than Kalman thinks" without
caring which way the runway points.

Data and honesty
----------------
* Training slices come from every aircraft track OUTSIDE the puzzle windows
  (data/derived/puzzle_windows.csv), so the game never tests the model on what it trained on.
* Like the Kalman tuning, every slice is used twice: complete data, and with the last
  6-12 s of reports missing (Fog-like), so the model also learns what stale data does.
* Evaluation is 5-fold cross-validation over 10-minute BLOCKS OF TIME, never random rows:
  neighbouring seconds of the same aircraft are nearly identical, so a random split would
  leak the answer and make the model look better than it is.

Uncertainty (split-conformal)
-----------------------------
The cone radius at each horizon is the 90th percentile of the model's own errors on data
it did NOT train on (out-of-fold errors), separately for complete and gappy data. So "90%"
is measured, not assumed.
"""
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from pipeline import kalman, truth

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

HORIZONS_S = (5, 10, 15, 20, 25, 30)   # the model predicts the correction at these look-aheads
HISTORY_S = 20                         # seconds of reports the forecaster sees before the decision
SLICE_EVERY_S = 3                      # one training slice every 3 s along each track
GAP_SECONDS = kalman.GAP_SECONDS       # Fog-like copies: last 6-12 s of reports missing
TIME_BLOCK_S = 600                     # cross-validation groups: 10-minute blocks of the recording
FOLDS = 5
STALE_AFTER_S = 1.5                    # data older than this counts as "gappy" for the cone

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,           # small trees: we only have one hour of data
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)

MODEL_DIR = os.path.join(truth.DERIVED, "ml")
REPORT_PATH = os.path.join(truth.DERIVED, "ml_report.json")

FEATURES = [
    "speed",               # Kalman speed estimate at the decision (m/s)
    "accel_along",         # Kalman acceleration along the direction of travel (m/s^2)
    "accel_cross",         # ...and sideways (turning)
    "data_age_s",          # how old the last report is
    "reports_last_10s",    # how many reports arrived in the last 10 s
    "speed_change_10s",    # measured speed now minus 10 s ago (from the reports)
    "turn_rate",           # change of direction of travel over the last ~10 s (deg/s)
    "on_runway_axis",      # 1 if on / lined up with an ACTIVE runway (see ai_player.is_runway_user)
    "runway_left_m",       # metres of that runway left ahead in the direction of travel
    "runway_perp_m",       # sideways distance from that runway's centreline
    "heading_off_runway",  # degrees between travel direction and the runway (0 = aligned)
]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def active_runways():
    """The active runways from the truth layer (geometry + activity), built once."""
    tracks = pd.read_csv(os.path.join(truth.DERIVED, "tracks.csv.gz"), dtype={"id": str})
    return [r for r in truth.build_runways(truth.load_map_segments(), tracks) if r.active]


def travel_frame(velocity):
    """Unit vectors (forward, left) for a velocity. Parked aircraft: just use east/north."""
    speed = float(np.linalg.norm(velocity))
    forward = velocity / speed if speed > 1.0 else np.array([1.0, 0.0])
    left = np.array([-forward[1], forward[0]])
    return forward, left


def measured_speed(observations, at_time, window=3):
    """Speed from the reports themselves (not the filter) around `at_time`, or nan."""
    near = [o for o in observations if at_time - window <= o[0] <= at_time]
    if len(near) < 2:
        return math.nan
    (t0, x0, y0), (t1, x1, y1) = near[0], near[-1]
    return math.hypot(x1 - x0, y1 - y0) / max(t1 - t0, 1e-6)


def direction_deg(observations, start, end):
    """Direction of travel (deg) between the reports nearest `start` and `end`, or nan."""
    a = [o for o in observations if start - 2 <= o[0] <= start + 2]
    b = [o for o in observations if end - 2 <= o[0] <= end]
    if not a or not b or b[-1][0] - a[0][0] < 2:
        return math.nan
    dx, dy = b[-1][1] - a[0][1], b[-1][2] - a[0][2]
    if math.hypot(dx, dy) < 5:
        return math.nan
    return math.degrees(math.atan2(dx, dy))


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------

def make_features(observations, decision_time, params=None):
    """Feature dict for one aircraft at the decision moment, plus the Kalman filter's state.
    Returns (features, kalman_state) or None if the aircraft hasn't been seen enough."""
    from pipeline.ai_player import is_runway_user   # local import: ai_player imports this module

    params = params or kalman.load_params()
    filtered = kalman.run_filter(observations, decision_time, params)
    if filtered is None:
        return None
    state, _ = filtered

    velocity, accel = state[2:4], state[4:6]
    forward, left = travel_frame(velocity)
    last_report = observations[-1][0]

    speed_then = measured_speed(observations, last_report - 10)
    speed_now = measured_speed(observations, last_report)
    heading_then = direction_deg(observations, last_report - 12, last_report - 6)
    heading_now = direction_deg(observations, last_report - 6, last_report)
    turn = (heading_now - heading_then + 180) % 360 - 180 if not (math.isnan(heading_then) or math.isnan(heading_now)) else 0.0

    features = {
        "speed": float(np.linalg.norm(velocity)),
        "accel_along": float(accel @ forward),
        "accel_cross": float(accel @ left),
        "data_age_s": float(decision_time - last_report),
        "reports_last_10s": float(sum(1 for o in observations if o[0] > decision_time - 10)),
        "speed_change_10s": float(speed_now - speed_then) if not (math.isnan(speed_now) or math.isnan(speed_then)) else 0.0,
        "turn_rate": float(turn / 6.0),
        "on_runway_axis": 0.0,
        "runway_left_m": -1.0,          # -1 = not on a runway axis
        "runway_perp_m": 999.0,
        "heading_off_runway": 90.0,
    }

    # Relationship to the closest active runway (by sideways distance)
    x, y = state[0], state[1]
    heading = math.degrees(math.atan2(forward[0], forward[1])) % 360
    best = None
    for rw in active_runways():
        along, perp = rw.to_runway_coords(x, y)
        if best is None or abs(perp) < abs(best[2]):
            best = (rw, along, perp)
    rw, along, perp = best
    rw_heading = math.degrees(math.atan2(rw.u[0], rw.u[1])) % 180
    features["runway_perp_m"] = float(abs(perp))
    features["heading_off_runway"] = float(abs((heading - rw_heading + 90) % 180 - 90))
    if is_runway_user(rw, x, y, heading, features["speed"]):
        features["on_runway_axis"] = 1.0
        going_up = float(forward @ rw.u) >= 0              # travelling towards along_max?
        features["runway_left_m"] = float(rw.along_max - along if going_up else along - rw.along_min)
    return features, state


def correction_targets(state, kalman_mean, true_future):
    """Targets: where the aircraft REALLY was minus where Kalman said, at each horizon,
    in the aircraft's own frame (forward, left)."""
    forward, left = travel_frame(state[2:4])
    targets = {}
    for h in HORIZONS_S:
        miss = true_future[h - 1] - kalman_mean[h - 1]
        targets[f"fwd_{h}"] = float(miss @ forward)
        targets[f"left_{h}"] = float(miss @ left)
    return targets


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

def build_dataset(tracks, excluded_windows, seed=0):
    """One row per (aircraft, decision moment, clean/gappy): features, targets, and the
    Kalman and constant-velocity errors (for comparison). Skips puzzle windows and parked aircraft."""
    from pipeline.ai_player import forecast_constant_velocity

    params = kalman.load_params()
    rng = np.random.default_rng(seed)
    horizon = max(HORIZONS_S)
    rows = []

    aircraft = tracks[(tracks.kind == "aircraft") & (tracks.alt_ft <= 1500)]
    for aircraft_id, track in aircraft.groupby("id"):
        track = track.set_index("t")
        times = track.index.values
        for start in np.arange(times.min(), times.max() - HISTORY_S - horizon + 1, SLICE_EVERY_S):
            decision_time = start + HISTORY_S
            end = decision_time + horizon
            if any(a < end and start < b for a, b in excluded_windows):
                continue                                     # never train on a puzzle's own seconds
            history = track.loc[start:decision_time]
            future = track.loc[decision_time + 1:end]
            if len(history) < HISTORY_S or len(future) < horizon:
                continue
            if history.speed.max() < 3 and future.speed.max() < 3:
                continue                                     # parked the whole time: nothing to learn

            all_reports = [(t, r.x, r.y) for t, r in history.iterrows()]
            true_future = future[["x", "y"]].values[:horizon]
            gap = int(rng.integers(GAP_SECONDS[0], GAP_SECONDS[1] + 1))
            variants = (("clean", all_reports),
                        ("gap", [o for o in all_reports if o[0] <= decision_time - gap]))

            for variant, observations in variants:
                made = make_features(observations, decision_time, params)
                kf = kalman.forecast(observations, decision_time, horizon, params, seed=len(rows))
                cv = forecast_constant_velocity(observations, decision_time, horizon, seed=len(rows))
                if made is None or kf is None or cv is None:
                    continue
                features, state = made
                row = {"aircraft_id": aircraft_id, "decision_time": decision_time, "variant": variant,
                       "time_block": int(decision_time // TIME_BLOCK_S), **features,
                       **correction_targets(state, kf.mean, true_future)}
                for h in HORIZONS_S:
                    row[f"kf_err_{h}"] = float(np.linalg.norm(kf.mean[h - 1] - true_future[h - 1]))
                    row[f"cv_err_{h}"] = float(np.linalg.norm(cv.mean[h - 1] - true_future[h - 1]))
                    row[f"kf_r_{h}"] = float(kf.radius[h - 1])
                    row[f"cv_r_{h}"] = float(cv.radius[h - 1])
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------------------

def fit_models(train):
    """One small LightGBM regressor per (horizon, direction): 6 horizons x 2 = 12 models."""
    import lightgbm as lgb

    models = {}
    for h in HORIZONS_S:
        for axis in ("fwd", "left"):
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(train[FEATURES], train[f"{axis}_{h}"])
            models[(axis, h)] = model
    return models


def predicted_errors(models, rows):
    """Distance between the ML forecast and the truth at each horizon: |true correction - predicted correction|."""
    errors = {}
    for h in HORIZONS_S:
        d_fwd = rows[f"fwd_{h}"].values - models[("fwd", h)].predict(rows[FEATURES])
        d_left = rows[f"left_{h}"].values - models[("left", h)].predict(rows[FEATURES])
        errors[h] = np.hypot(d_fwd, d_left)
    return errors


def purged_training_rows(data, test):
    """Training rows that can't leak into `test`: a slice uses HISTORY_S seconds before its decision
    and up to 30 s after it, so any training slice within that reach of a test slice's time is dropped.
    (Without this, an aircraft near a block boundary would have almost identical slices on both sides.)"""
    reach = HISTORY_S + max(HORIZONS_S)
    test_times = np.sort(test.decision_time.unique())
    nearest = np.searchsorted(test_times, data.decision_time.values)
    left = np.abs(data.decision_time.values - test_times[np.clip(nearest - 1, 0, len(test_times) - 1)])
    right = np.abs(data.decision_time.values - test_times[np.clip(nearest, 0, len(test_times) - 1)])
    too_close = np.minimum(left, right) <= reach
    return data[~too_close & ~data.index.isin(test.index)]


def cross_validate(data):
    """5-fold CV over 10-minute time blocks, with a purge gap around each test block. Returns the
    dataset with out-of-fold ML errors added (ml_err_<h>): errors on data each model never saw."""
    blocks = np.array(sorted(data.time_block.unique()))
    fold_of_block = {b: i % FOLDS for i, b in enumerate(blocks)}
    data = data.assign(fold=data.time_block.map(fold_of_block))

    for h in HORIZONS_S:
        data[f"ml_err_{h}"] = np.nan
    for fold in range(FOLDS):
        test = data[data.fold == fold]
        train = purged_training_rows(data, test)
        if test.empty or train.empty:
            continue
        errors = predicted_errors(fit_models(train), test)
        for h in HORIZONS_S:
            data.loc[test.index, f"ml_err_{h}"] = errors[h]
    return data


def conformal_radii(errors):
    """90th percentile of out-of-fold errors per horizon: the ML cone radius."""
    return {h: float(np.quantile(errors[f"ml_err_{h}"].dropna(), 0.9)) for h in HORIZONS_S}


def cross_conformal_coverage(data):
    """Honest coverage check: for each fold, the radius comes from the OTHER folds' errors."""
    hits = []
    for fold in sorted(data.fold.unique()):
        test, rest = data[data.fold == fold], data[data.fold != fold]
        for variant in ("clean", "gap"):
            radii = conformal_radii(rest[rest.variant == variant])
            part = test[test.variant == variant]
            for h in HORIZONS_S:
                hits.extend((part[f"ml_err_{h}"] <= radii[h]).tolist())
    return float(np.mean(hits))


def report(data):
    """Mean error and cone coverage for cv / kf / ml, overall and by situation."""
    situations = {
        "all": data,
        "clean": data[data.variant == "clean"],
        "gap (fog-like)": data[data.variant == "gap"],
        "speeding up (> 0.5 m/s^2)": data[data.speed_change_10s > 5],
        "slowing down": data[data.speed_change_10s < -5],
        "on an active runway axis": data[data.on_runway_axis == 1],
    }
    out = {}
    for name, part in situations.items():
        if part.empty:
            continue
        out[name] = {"rows": int(len(part))}
        for player in ("cv", "kf", "ml"):
            out[name][player] = {f"err_{h}s_m": round(float(part[f"{player}_err_{h}"].mean()), 1) for h in (10, 20, 30)}
    for player in ("cv", "kf"):
        out["all"][player]["coverage"] = round(float(np.mean(
            [(data[f"{player}_err_{h}"] <= data[f"{player}_r_{h}"]).mean() for h in HORIZONS_S])), 3)
    out["all"]["ml"]["coverage"] = round(cross_conformal_coverage(data), 3)
    return out


def train_and_save(tracks, excluded_windows, verbose=True):
    """Build the dataset, cross-validate (report), train the final models on everything, save."""
    data = build_dataset(tracks, excluded_windows)
    data = cross_validate(data)
    results = report(data)

    models = fit_models(data)
    radii = {variant: conformal_radii(data[data.variant == variant]) for variant in ("clean", "gap")}

    os.makedirs(MODEL_DIR, exist_ok=True)
    for (axis, h), model in models.items():
        model.booster_.save_model(os.path.join(MODEL_DIR, f"{axis}_{h}.txt"))
    importance = pd.Series(
        sum(m.booster_.feature_importance("gain") for m in models.values()), index=FEATURES
    ).sort_values(ascending=False)
    summary = {
        "rows": int(len(data)),
        "aircraft": int(data.aircraft_id.nunique()),
        "time_blocks": int(data.time_block.nunique()),
        "horizons_s": list(HORIZONS_S),
        "conformal_radius_m": radii,
        "feature_importance_gain_share": (importance / importance.sum()).round(3).to_dict(),
        "results": results,
    }
    with open(REPORT_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)
    load_model.cache_clear()
    if verbose:
        print(json.dumps({k: summary[k] for k in ("rows", "aircraft", "time_blocks")}))
        for name, r in results.items():
            print(f"  {name:28s} rows {r['rows']:5d} | " + " | ".join(
                f"{p} " + " ".join(f"{v:6.1f}" for k, v in r[p].items() if k.startswith("err")) for p in ("cv", "kf", "ml")))
        print("  coverage:", {p: results["all"][p]["coverage"] for p in ("cv", "kf", "ml")})
        print("  top features:", list(importance.index[:5]))
    return summary


# --------------------------------------------------------------------------------------
# Using the saved model (the AI player calls this)
# --------------------------------------------------------------------------------------

@dataclass
class SavedModel:
    boosters: dict      # (axis, horizon) -> lightgbm.Booster
    radii: dict         # "clean"/"gap" -> {horizon: radius}


@lru_cache(maxsize=1)
def load_model():
    """The trained model from data/derived/ml/, or None if `train_and_save` hasn't run."""
    if not os.path.exists(REPORT_PATH):
        return None
    import lightgbm as lgb

    with open(REPORT_PATH) as fh:
        radii = {variant: {int(h): r for h, r in by_h.items()}
                 for variant, by_h in json.load(fh)["conformal_radius_m"].items()}
    boosters = {(axis, h): lgb.Booster(model_file=os.path.join(MODEL_DIR, f"{axis}_{h}.txt"))
                for h in HORIZONS_S for axis in ("fwd", "left")}
    return SavedModel(boosters=boosters, radii=radii)


def interpolate_by_horizon(values_at_horizons, steps, at_zero):
    """Values given at HORIZONS_S -> one value per second 1..steps (linear, `at_zero` at 0 s)."""
    xs = [0] + list(HORIZONS_S)
    ys = [at_zero] + [values_at_horizons[h] for h in HORIZONS_S]
    return np.interp(np.arange(1, steps + 1), xs, ys)


def forecast_ml(observations, decision_time, steps, seed=0):
    """ML forecast = Kalman forecast + learned correction, with the conformal cone.

    The sampled futures keep the Kalman filter's SHAPE of uncertainty (which way it's
    stretched), shifted by the correction and rescaled so the 90% radius equals the
    conformal radius measured on held-out data."""
    model = load_model()
    if model is None:
        return None
    made = make_features(observations, decision_time)
    kf = kalman.forecast(observations, decision_time, steps, kalman.load_params(), seed=seed)
    if made is None or kf is None:
        return None
    features, state = made

    row = pd.DataFrame([features])[FEATURES]
    forward, left = travel_frame(state[2:4])
    fwd = {h: float(model.boosters[("fwd", h)].predict(row)[0]) for h in HORIZONS_S}
    side = {h: float(model.boosters[("left", h)].predict(row)[0]) for h in HORIZONS_S}
    fwd_per_step = interpolate_by_horizon(fwd, steps, 0.0)
    side_per_step = interpolate_by_horizon(side, steps, 0.0)
    correction = np.outer(fwd_per_step, forward) + np.outer(side_per_step, left)     # (steps, 2)

    variant = "gap" if features["data_age_s"] > STALE_AFTER_S else "clean"
    radius = interpolate_by_horizon(model.radii[variant], steps, at_zero=model.radii[variant][HORIZONS_S[0]] / 2)

    mean = kf.mean + correction
    spread = kf.samples - kf.mean                                   # Kalman's shape of uncertainty
    scale = radius / np.maximum(kf.radius, 1e-6)
    samples = mean + spread * scale[None, :, None]

    # speed_now stays the filter's estimate of the CURRENT speed (what the player can see);
    # the learned part is about what happens next, and lives in `mean`.
    return kalman.Forecast(times=kf.times, mean=mean, radius=radius, samples=samples,
                           speed_now=kf.speed_now, accel_along_track=kf.accel_along_track)


if __name__ == "__main__":
    tracks = pd.read_csv(os.path.join(truth.DERIVED, "tracks.csv.gz"), dtype={"id": str})
    windows = pd.read_csv(os.path.join(truth.DERIVED, "puzzle_windows.csv"))[["start_s", "end_s"]].values.tolist()
    train_and_save(tracks, windows)
