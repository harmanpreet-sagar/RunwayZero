"""Fog mode: degrade what the player (and the AI) sees before the freeze.

Real surveillance fails in a few typical ways; Fog mode applies all three, seeded per
puzzle so the same puzzle always looks the same:

  * DROPOUT  the question aircraft's reports stop for the last few seconds before the
             freeze. It is shown at its LAST KNOWN position, marked "stale", then "lost".
  * DELAY    every other object's report is a couple of seconds old ("stale").
  * NOISE    every report is off by a few metres.

The reveal is never degraded: it always shows the truth. The AI only gets these same
degraded frames, so it plays by the same rules as the human.
"""
import numpy as np

DROPOUT_SECONDS = (6, 12)      # the question aircraft goes silent this long before the freeze
DELAY_SECONDS = (2, 3)         # other objects' reports are this old
NOISE_M = 8.0                  # standard deviation of position error per report
MIN_REPORTS_BEFORE_DROPOUT = 3  # the question aircraft is always seen for at least this many seconds

# Same age thresholds as the contract's status field
STALE_AFTER_S = 1.5
LOST_AFTER_S = 5.0


def status_for_age(age_s):
    if age_s <= STALE_AFTER_S:
        return "ok"
    return "stale" if age_s <= LOST_AFTER_S else "lost"


def apply_fog(pre_frames, question_aircraft_id, seed):
    """Return degraded copies of `pre_frames` (the originals are untouched).

    Each frame's objects get x/y replaced by what the sensors would report, plus age_s
    (how old that report is) and status (ok / stale / lost)."""
    rng = np.random.default_rng(seed)
    decision_t = pre_frames[-1]["t"]
    dropout_s = int(rng.integers(DROPOUT_SECONDS[0], DROPOUT_SECONDS[1] + 1))
    delay_s = int(rng.integers(DELAY_SECONDS[0], DELAY_SECONDS[1] + 1))
    silent_from = decision_t - dropout_s            # question aircraft: no reports after this

    # If the question aircraft only came into view shortly before the freeze, a long dropout would
    # hide it COMPLETELY, which isn't fog, it's an unfair puzzle. Always leave it at least
    # MIN_REPORTS_BEFORE_DROPOUT seconds of reports, so the player sees where it was (a ghost).
    first_seen = min((f["t"] for f in pre_frames for o in f["objects"] if o["id"] == question_aircraft_id),
                     default=decision_t)
    silent_from = max(silent_from, min(first_seen + MIN_REPORTS_BEFORE_DROPOUT - 1, decision_t))

    # true positions by (object id, time), so a delayed report can look up an older position
    truth = {(o["id"], f["t"]): o for f in pre_frames for o in f["objects"]}

    # one noisy report per object per second, drawn once so a "frozen" ghost doesn't jitter
    noise = {}

    def reported(object_id, time):
        """The noisy report the sensors produced for this object at `time` (None if none)."""
        if (object_id, time) not in truth:
            return None
        if (object_id, time) not in noise:
            noise[(object_id, time)] = rng.normal(0.0, NOISE_M, size=2)
        true_obj = truth[(object_id, time)]
        dx, dy = noise[(object_id, time)]
        return {**true_obj, "x": round(true_obj["x"] + dx, 1), "y": round(true_obj["y"] + dy, 1)}

    degraded = []
    for frame in pre_frames:
        t = frame["t"]
        objects = []
        for true_obj in frame["objects"]:
            object_id = true_obj["id"]

            if object_id == question_aircraft_id and t > silent_from:
                # dropout: keep showing the last report from before the silence
                report_time = silent_from
            elif object_id == question_aircraft_id:
                report_time = t                       # before the dropout it's reported on time
            else:
                report_time = t - delay_s             # everything else arrives late

            # walk back to the latest second that actually has a report
            report = None
            while report is None and report_time >= pre_frames[0]["t"]:
                report = reported(object_id, report_time)
                if report is None:
                    report_time -= 1
            if report is None:
                continue                              # nothing reported yet: not on screen

            age = float(t - report_time)
            objects.append({**report, "age_s": age, "status": status_for_age(age)})
        degraded.append({"t": t, "objects": objects})
    return degraded


def observations_from_frames(pre_frames, object_id):
    """The distinct reports of one object as [(report_time, x, y), ...], oldest first.

    A frame shows a report that is age_s old, so it was taken at frame.t - age_s. Stale and
    lost frames repeat an old report; those repeats are collapsed into one observation.
    Works for clear/busy frames too (age_s = 0 everywhere)."""
    seen = {}
    for frame in pre_frames:
        for obj in frame["objects"]:
            if obj["id"] == object_id:
                report_time = frame["t"] - obj.get("age_s", 0.0)
                seen[report_time] = (report_time, obj["x"], obj["y"])
    return [seen[t] for t in sorted(seen)]
