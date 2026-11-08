"""The AI player: looks at the same frames the human saw and makes the CLEAR / HOLD call.

How it decides
--------------
1. Forecast every aircraft on (or lined up with) the runway for the next 30 s as a cloud of
   possible futures (the "forecaster": constant velocity, Kalman filter, or the ML model).
2. Assume the vehicle, if cleared, drives straight across the runway at a typical speed.
3. For each possible future, check: is any aircraft within the conflict distance while
   the vehicle is on the runway?  p_conflict = share of futures where that happens.
4. p_safe = 1 - p_conflict, snapped to 50 / 75 / 95% exactly like the human's buttons.

Each player is one entry in FORECASTERS.
"""
import math
from dataclasses import dataclass

import numpy as np

from pipeline import kalman, ml_model, truth
from pipeline.contracts import snap_confidence
from pipeline.faults import observations_from_frames

HORIZON_S = 30                     # seconds forecast after the decision (= the reveal length)
ASSUMED_VEHICLE_SPEED_MS = 6.0     # typical crossing speed in this hour (~150 m in ~25 s)
P_SAFE_LIMITS = (0.02, 0.98)       # never claim certainty

# Which aircraft matter for a crossing: those ON or LINED UP WITH the runway being crossed.
# Same rule for the scoring (puzzles.py) and the AI, so they agree on what counts.
RUNWAY_USER_MAX_PERP_M = 100        # within this distance of the runway centreline (extended)...
RUNWAY_USER_BEYOND_ENDS_M = 4000    # ...between the runway ends or up to this far beyond them (approach / climb-out)...
RUNWAY_USER_MAX_HEADING_DEG = 25    # ...and pointing along the runway (either direction)

# Constant-velocity forecaster settings (the naive baseline)
CV_BASE_RADIUS_M = 10.0            # 90% radius right away...
CV_RADIUS_GROWTH_M_PER_S = 3.0     # ...growing this much per second of look-ahead (and of data age)
RADIUS90_TO_SIGMA = 1 / 2.146      # for a round 2D Gaussian, the 90% radius is 2.146 standard deviations


# --------------------------------------------------------------------------------------
# Forecasters: observations -> kalman.Forecast (mean path, cone, sampled futures)
# --------------------------------------------------------------------------------------

def forecast_constant_velocity(observations, decision_t, steps, seed):
    """Naive baseline: keep the last measured speed and direction forever.
    Velocity comes from the last ~3 s of reports; the cone grows with look-ahead AND data age."""
    if len(observations) < 2:
        return None
    recent = observations[-4:]
    (t0, x0, y0), (t1, x1, y1) = recent[0], recent[-1]
    vx, vy = (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)
    data_age = decision_t - t1

    times = decision_t + np.arange(1, steps + 1)
    ahead = times - t1                                         # seconds since the last report
    mean = np.column_stack([x1 + vx * ahead, y1 + vy * ahead])
    radius = CV_BASE_RADIUS_M + CV_RADIUS_GROWTH_M_PER_S * ahead

    rng = np.random.default_rng(seed)
    sigma = radius * RADIUS90_TO_SIGMA
    samples = mean + rng.standard_normal((kalman.SAMPLES, steps, 2)) * sigma[None, :, None]
    speed = math.hypot(vx, vy)
    return kalman.Forecast(times=times, mean=mean, radius=radius, samples=samples,
                           speed_now=speed, accel_along_track=0.0) if data_age >= 0 else None


def forecast_kalman(observations, decision_t, steps, seed):
    """Constant-acceleration Kalman filter with settings tuned on real tracks (see kalman.py)."""
    return kalman.forecast(observations, decision_t, steps, kalman.load_params(), seed=seed)


def forecast_learned(observations, decision_t, steps, seed):
    """Kalman forecast + LightGBM correction learned from past traffic, conformal cone (see ml_model.py)."""
    return ml_model.forecast_ml(observations, decision_t, steps, seed=seed)


FORECASTERS = {
    "cv": forecast_constant_velocity,
    "kf": forecast_kalman,
    "ml": forecast_learned,
}


# --------------------------------------------------------------------------------------
# The vehicle and the runway
# --------------------------------------------------------------------------------------

def is_on_runway(runway, x, y):
    """True when (x, y) is on the runway surface used for crossings (same rule as truth.py)."""
    along, perp = runway.to_runway_coords(x, y)
    return (abs(perp) <= truth.CROSSING_HALF_WIDTH_M) and (runway.along_min <= along <= runway.along_max)


def is_runway_user(runway, x, y, heading):
    """True for an aircraft on, or lined up with, `runway`: on the runway, holding in position
    for takeoff, on final approach or climbing out. False for aircraft crossing it on a taxiway,
    on a parallel taxiway, parked, or on another runway."""
    along, perp = runway.to_runway_coords(x, y)
    runway_heading = math.degrees(math.atan2(runway.u[0], runway.u[1])) % 180
    heading_off = abs((heading - runway_heading + 90) % 180 - 90)      # 0 = aligned, either direction
    return (
        abs(perp) <= RUNWAY_USER_MAX_PERP_M
        and runway.along_min - RUNWAY_USER_BEYOND_ENDS_M <= along <= runway.along_max + RUNWAY_USER_BEYOND_ENDS_M
        and heading_off <= RUNWAY_USER_MAX_HEADING_DEG
    )


def seen_using_runway(pre_frames, aircraft_id, runway):
    """Did the player/AI see this aircraft on or lined up with the runway before the freeze?"""
    return any(
        is_runway_user(runway, obj["x"], obj["y"], obj.get("heading", 0.0))
        for frame in pre_frames for obj in frame["objects"] if obj["id"] == aircraft_id
    )


def assumed_vehicle_path(vehicle_observations, runway, decision_t, steps):
    """If cleared, the AI assumes the vehicle drives straight across the runway at a typical
    speed to the same distance on the far side. (It does NOT know the vehicle's real path.)"""
    _, x, y = vehicle_observations[-1]
    along, perp = runway.to_runway_coords(x, y)
    far_side = -np.sign(perp) * max(abs(perp), truth.CROSSING_HALF_WIDTH_M + 10)
    total_distance = abs(far_side - perp)
    direction = np.sign(far_side - perp)

    path = []
    for step in range(1, steps + 1):
        travelled = min(ASSUMED_VEHICLE_SPEED_MS * step, total_distance)
        px, py = runway.to_world(along, perp + direction * travelled)
        path.append({"t": decision_t + step, "x": round(float(px), 1), "y": round(float(py), 1), "r": 5.0})
    return path


# --------------------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------------------

@dataclass
class Answer:
    detector: str
    p_safe: float
    call: str                  # "CLEAR" / "HOLD"
    confidence: float          # 0.5 / 0.75 / 0.95
    paths: dict                # object id -> [{t, x, y, r}] (vehicle + most dangerous aircraft)
    reasoning: str
    worst_aircraft: str        # id of the aircraft that drove the decision (or None)


def decide(detector, pre_frames, objects, vehicle_id, runway, threshold_m, label_for, seed=0):
    """Make the AI's call using `detector`'s forecaster on the frames the player saw."""
    forecaster = FORECASTERS[detector]
    decision_t = pre_frames[-1]["t"]

    vehicle_obs = observations_from_frames(pre_frames, vehicle_id)
    vehicle_path = assumed_vehicle_path(vehicle_obs, runway, decision_t, HORIZON_S)
    vehicle_xy = np.array([(p["x"], p["y"]) for p in vehicle_path])
    vehicle_on_runway = np.array([is_on_runway(runway, x, y) for x, y in vehicle_xy])

    conflict_in_any = np.zeros(kalman.SAMPLES, dtype=bool)    # per possible future, across all aircraft
    worst = None                                               # (p_conflict, aircraft id, forecast, closest)

    for index, obj in enumerate(o for o in objects if o["type"] == "aircraft"):
        if not seen_using_runway(pre_frames, obj["id"], runway):
            continue                                           # taxiing / parked elsewhere: not a threat to this crossing
        observations = observations_from_frames(pre_frames, obj["id"])
        forecast = forecaster(observations, decision_t, HORIZON_S, seed=seed * 1000 + index)
        if forecast is None:
            continue                                           # never seen enough: can't forecast it

        # distance from every possible aircraft position to where the vehicle would be, each second
        distance = np.linalg.norm(forecast.samples - vehicle_xy[None, :, :], axis=-1)   # (SAMPLES, steps)
        close_while_on_runway = (distance < threshold_m) & vehicle_on_runway[None, :]
        conflict = close_while_on_runway.any(axis=1)
        conflict_in_any |= conflict

        p_conflict = float(conflict.mean())
        mean_distance = np.linalg.norm(forecast.mean - vehicle_xy, axis=-1)
        mean_distance = np.where(vehicle_on_runway, mean_distance, np.inf)
        closest_step = int(np.argmin(mean_distance))
        closest = (float(mean_distance[closest_step]), closest_step + 1, float(forecast.radius[closest_step]))
        if worst is None or (p_conflict, -closest[0]) > (worst[0], -worst[3][0]):
            worst = (p_conflict, obj["id"], forecast, closest, observations)

    p_safe = 1.0 - float(conflict_in_any.mean())
    p_safe = min(max(p_safe, P_SAFE_LIMITS[0]), P_SAFE_LIMITS[1])
    call, confidence = snap_confidence(p_safe)

    paths = {vehicle_id: vehicle_path}
    worst_id = None
    if worst is not None:
        worst_id = worst[1]
        paths[worst_id] = worst[2].as_path()

    reasoning = explain(detector, worst, decision_t, threshold_m, call, confidence, label_for)
    return Answer(detector=detector, p_safe=round(p_safe, 3), call=call, confidence=confidence,
                  paths=paths, reasoning=reasoning, worst_aircraft=worst_id)


def explain(detector, worst, decision_t, threshold_m, call, confidence, label_for):
    """One or two plain-English sentences behind the call."""
    sure = f"{call} ({confidence * 100:.0f}% sure)."
    if worst is None:
        return f"No aircraft on or lined up with this runway, so {sure}"

    p_conflict, aircraft_id, forecast, (closest_m, seconds_ahead, radius), observations = worst
    label = label_for(aircraft_id)
    data_age = decision_t - observations[-1][0]
    stale = f" Its last report is {data_age:.0f} s old, so I'm less sure where it is." if data_age > 1.5 else ""

    if detector == "cv":
        motion = f"{label} is doing {forecast.speed_now:.0f} m/s; I assume it keeps that speed and heading."
    elif detector == "ml":
        ahead = float(np.linalg.norm(forecast.mean[min(9, len(forecast.mean) - 1)] - observations[-1][1:]))
        motion = (f"{label} is doing {forecast.speed_now:.0f} m/s. From similar moments in past traffic, "
                  f"I expect it to cover about {ahead:.0f} m in the next 10 s.")
    else:
        accel = forecast.accel_along_track
        if accel > 0.3:
            trend = f"speeding up at {accel:.1f} m/s²"
        elif accel < -0.3:
            trend = f"slowing at {-accel:.1f} m/s²"
        else:
            trend = "holding its speed"
        motion = f"{label} is doing {forecast.speed_now:.0f} m/s and {trend}."

    if math.isinf(closest_m):
        where = "The vehicle wouldn't be on the runway long enough to matter."
    else:
        where = (f"Closest predicted approach while the vehicle is on the runway: about {closest_m:.0f} m "
                 f"±{radius:.0f} m, {seconds_ahead} s from now; within {threshold_m:.0f} m in "
                 f"{p_conflict * 100:.0f}% of possible futures.")
    return f"{motion}{stale} {where} So {sure}"
