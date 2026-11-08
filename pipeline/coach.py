"""Gemini debriefs: fill each puzzle's `ai.coach` with a short explanation of what happened
and what to learn from it. Precomputed once (no live API calls during the game), then
reviewed by a person before committing.

    python -m pipeline.coach            # draft debriefs -> data/derived/coach_drafts.json (resumes; skips done ones)
    python -m pipeline.coach --apply    # after reviewing the drafts: write them into web/public/data/puzzle_*.json
    python -m pipeline.coach --only r15r33l_v3475_t1870 --redo

Needs `pip install google-genai` and GEMINI_API_KEY in the environment.
Run it AFTER `python -m pipeline.puzzles` (which rewrites the puzzle files).

How it stays honest
-------------------
The facts are computed here, in Python, with the same rules the game shows (knots, feet,
the present-speed margin from the info display, each model's call). Gemini only gets those
facts and is told not to add any. Every number in its answer must appear in the facts, and
every ID it mentions must be one we gave it; otherwise we retry once with the problem
pointed out, and if it still fails the puzzle keeps no debrief.
`ai.coach` is hidden until the player has locked in (contracts/README.md).
"""
import argparse
import json
import math
import os
import re
import time

import numpy as np

from pipeline.contracts import ROOT, WEB_DATA_DIR, write

MODELS = tuple(os.environ.get("COACH_MODELS", "").split(",")) if os.environ.get("COACH_MODELS") else (
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")   # tried in order; free tier is 20 requests/day per model
M_S_TO_KT = 1.94384
M_TO_FT = 3.28084
DETECTOR_NAMES = {"cv": "constant-velocity model", "kf": "Kalman filter", "ml": "learned (ML) model"}

SYSTEM = """You write the debrief shown after one round of Runway Zero, a practice game for air traffic control
trainees and airside drivers. The player saw recorded surface traffic at Boston Logan, froze at the moment a
vehicle was at the hold line, and called CLEAR (cross the runway now) or HOLD. Then the recording played out.

Write 2 or 3 short sentences, at most 60 words, plain and calm:
1. What actually happened, in concrete terms (who, how close, when).
2. Why that made the correct call correct.
3. One thing to watch for next time, tied to these facts (for example the aircraft's speed trend, old or lost
   position data, a misleading present-speed estimate, or the models disagreeing).

Rules:
- Use ONLY the facts provided. Do not invent causes, intentions, weather, aircraft types, procedures or rules.
- Use the IDs exactly as given (like AC5863, VH3475). Speeds in kt, distances in ft, times in s.
- Do not repeat the player's score or say "great job". Do not give real-world operational advice.
- If the round is a "what if", do not explain the what-if mechanics; the game already labels it.
- The correct call is correct ONLY because of the closest approach versus the safe distance. Say that plainly; do not
  give any other reason for it. Things like old data, speed changes or a misleading estimate are what made the call
  HARD to see, so they belong in the "watch for" sentence, not the "why" sentence.
- Distances given "at the freeze" are at the freeze; do not move them to another moment.
- Name the forecasters only as given (constant-velocity model, Kalman filter, learned model). If they disagreed, or all
  got it wrong, that is worth saying.
- The "watch for" sentence must be something a player can look at on screen next time, not a statement about the AI.
Return only the debrief text."""


def kt(ms):
    return round((ms or 0) * M_S_TO_KT)


def ft(m):
    return round(m * M_TO_FT)


def callsign(label):
    m = re.match(r"^(aircraft|vehicle|unknown)\s*(\w+)$", label.strip(), re.I)
    if not m:
        return label.upper().replace(" ", "")
    return {"aircraft": "AC", "vehicle": "VH", "unknown": "U"}[m.group(1).lower()] + m.group(2)


def obj_at(frames, t, oid):
    """Object `oid` in the last frame at or before t (frames are 1 s apart)."""
    best = None
    for f in frames:
        if f["t"] <= t + 1e-6:
            best = f
    if best is None:
        return None
    return next((o for o in best["objects"] if o["id"] == oid), None)


def zone_of(map_, pz):
    return next(z for z in map_["zones"] if z["id"] == pz["zone_id"])


def crossing_depth(zone_poly, vehicle_xy):
    """Distance across the crossing zone from the edge nearest the vehicle (same as the game's stop bar)."""
    pts = np.array(zone_poly, dtype=float)
    v = np.array(vehicle_xy, dtype=float)
    best = None
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        d = b - a
        L = float(np.linalg.norm(d)) or 1.0
        u = d / L
        q = a + u * min(max(float(np.dot(v - a, u)), 0.0), L)
        dist = float(np.linalg.norm(v - q))
        if best is None or dist < best[0]:
            best = (dist, q, u)
    _, q, u = best
    n = np.array([-u[1], u[0]])
    if np.dot(v - q, n) < 0:
        n = -n
    return float(max(np.dot(p - q, -n) for p in pts))


def present_speed_eta(aircraft, zone_center):
    """Seconds until the aircraft reaches the crossing at its present speed and heading (None if not heading there)."""
    h = math.radians(aircraft.get("heading") or 0.0)
    u = np.array([math.sin(h), math.cos(h)])
    rel = np.array(zone_center) - np.array([aircraft["x"], aircraft["y"]])
    along = float(np.dot(rel, u))
    miss = abs(rel[0] * u[1] - rel[1] * u[0])
    speed = aircraft.get("speed") or 0.0
    if along > 0 and miss < 150 and speed > 1:
        return along / speed
    return None


def facts_for(map_, pz):
    """Everything the debrief may use, as plain sentences. Also returns the allowed numbers and IDs."""
    d, vid, aid = pz["decision_t"], pz["vehicle_id"], pz["aircraft_id"]
    label = {o["id"]: o["label"] for o in pz["objects"]}
    V, A = callsign(label[vid]), callsign(label[aid])
    zone = zone_of(map_, pz)
    runway = re.search(r"RWY\s*([\w/]+)", zone["name"]).group(1)
    zc = np.mean(np.array(zone["polygon"], dtype=float), axis=0)
    oc = pz["outcome"]

    ac_now = obj_at(pz["pre"], d, aid)
    ac_before = obj_at(pz["pre"], d - 5, aid)
    veh_now = obj_at(pz["pre"], d, vid)
    lines = [
        f"Runway being crossed: {runway}. Condition: {pz['difficulty']}.",
        f"Vehicle {V} was at the hold line asking to cross. Runway traffic: aircraft {A}.",
    ]
    numbers = set()

    if ac_now:
        s_now, s_before = kt(ac_now.get("speed")), kt(ac_before.get("speed")) if ac_before else None
        dist = ft(float(np.linalg.norm(np.array([ac_now["x"], ac_now["y"]]) - zc)))
        trend = ""
        if s_before is not None:
            if s_now - s_before >= 8:
                trend = f", speeding up (it was {s_before} kt 5 s earlier)"
            elif s_before - s_now >= 8:
                trend = f", slowing down (it was {s_before} kt 5 s earlier)"
            else:
                trend = ", holding a steady speed"
            numbers.add(s_before)
        lines.append(f"At the freeze, {A} was {dist} ft from the crossing at {s_now} kt{trend}.")
        numbers |= {s_now, dist, 5}
        status = ac_now.get("status", "ok")
        if status != "ok":
            age = round(ac_now.get("age_s", 0))
            lines.append(f"{A}'s position data was {'lost' if status == 'lost' else 'old'}: last report {age} s before the freeze.")
            numbers.add(age)

    # The present-speed margin the info display showed (deliberately naive: ignores acceleration and braking).
    if ac_now and veh_now:
        vmax = max((o["speed"] for f in pz["pre"] for o in f["objects"] if o["id"] == vid and o.get("speed")), default=0)
        v_speed = vmax if vmax > 1 else 8.0
        need = (crossing_depth(zone["polygon"], (veh_now["x"], veh_now["y"])) + 10) / v_speed
        eta = present_speed_eta(ac_now, zc)
        if eta is not None:
            margin = round(eta - need)
            lines.append(f"The info display's present-speed estimate said {A} would reach the crossing in {round(eta)} s and "
                         f"{V} needed {round(need)} s to cross, a margin of {margin:+d} s. That estimate assumes nobody changes speed.")
            numbers |= {round(eta), round(need), abs(margin)}

    # What really happened.
    sep_ft, after_s = ft(oc["min_sep_m"]), round(oc["min_sep_t"] - d)
    limit_ft = ft(oc["threshold_m"])
    ac_cpa = obj_at(pz["reveal"], oc["min_sep_t"], aid)
    lines.append(f"In the recording, {V} and {A} came within {sep_ft} ft of each other {after_s} s after the freeze "
                 f"(safe distance {limit_ft} ft). {'That is a conflict, so HOLD was correct.' if oc['conflict'] else 'Separation held, so CLEAR was correct.'}")
    numbers |= {sep_ft, after_s, limit_ft}
    if ac_cpa and ac_cpa.get("speed") is not None:
        s_cpa = kt(ac_cpa["speed"])
        lines.append(f"At that moment {A} was doing {s_cpa} kt.")
        numbers.add(s_cpa)

    # The AI players.
    calls = []
    for det, c in (pz["ai"].get("baselines") or {}).items():
        right = c["call"] == oc["correct_call"]
        pct = round(c["confidence"] * 100)
        calls.append(f"the {DETECTOR_NAMES.get(det, det)} said {c['call']} at {pct}% ({'right' if right else 'wrong'})")
        numbers.add(pct)
    if calls:
        lines.append("AI forecasts: " + "; ".join(calls) + ".")
    if pz["source"].get("vehicle_time_shift_s"):
        lines.append("This is a what-if round: the vehicle's real crossing was replayed from the freeze; the aircraft are as recorded.")

    numbers |= {int(x) for x in re.findall(r"\d+", runway)}
    return "\n".join(lines), {str(n) for n in numbers}, {V, A}


def check(text, allowed_numbers, allowed_ids):
    """Return a list of problems with a debrief (empty = OK)."""
    problems = []
    words = len(text.split())
    if not 15 <= words <= 75:
        problems.append(f"it is {words} words; write 2 or 3 sentences, at most 60 words")
    for n in re.findall(r"\d+", re.sub(r"\b(?:AC|VH|U)\d+\b", "", text)):
        if n not in allowed_numbers:
            problems.append(f"the number {n} is not in the facts")
    for ident in re.findall(r"\b(?:AC|VH|U)\d+\b", text):
        if ident not in allowed_ids:
            problems.append(f"{ident} is not one of the given IDs")
    if re.search(r"\b\d+\s*(m/s|metres|meters|m)\b", text):
        problems.append("use kt and ft, not metric units")
    return problems


DRAFTS = os.path.join(ROOT, "data", "derived", "coach_drafts.json")   # accepted drafts, reviewed before --apply
MIN_SECONDS_BETWEEN_CALLS = 13   # free tier: 5 requests per minute per model


class ModelPool:
    """Calls Gemini, moving down MODELS when one is overloaded, timing out, or out of daily free quota."""

    def __init__(self, client):
        self.client = client
        self.alive = list(MODELS)
        self.last_call = {}

    def ask(self, prompt):
        from google.genai import types

        while self.alive:
            model = self.alive[0]
            wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - self.last_call.get(model, 0))
            if wait > 0:
                time.sleep(wait)
            self.last_call[model] = time.time()
            try:
                resp = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.3, max_output_tokens=2048),
                )
                return (resp.text or "").strip().replace("\n", " "), model
            except Exception as e:  # noqa: BLE001 - decide whether to wait, retry or drop this model
                msg = str(e)
                per_minute = "PerMinute" in msg
                delay = re.search(r"retry in ([\d.]+)s", msg)
                if "429" in msg and per_minute and delay:
                    print(f"    {model}: per-minute limit, waiting {float(delay.group(1)):.0f} s")
                    time.sleep(float(delay.group(1)) + 1)
                    continue
                reason = "daily quota used up" if "PerDay" in msg else msg[:60]
                print(f"    {model}: {reason}; switching model")
                self.alive.pop(0)
        raise RuntimeError("all Gemini models are unavailable (daily free quota, overload or timeouts)")


def make_client():
    try:  # some networks intercept TLS; use the OS certificate store when available
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from google import genai
    from google.genai import types

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY first (Google AI Studio -> Get API key).")
    # Without a timeout a stalled request waits forever; 60 s is plenty for a few sentences.
    return genai.Client(http_options=types.HttpOptions(timeout=60_000))


def load_drafts():
    if os.path.exists(DRAFTS):
        with open(DRAFTS, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_drafts(drafts):
    os.makedirs(os.path.dirname(DRAFTS), exist_ok=True)
    with open(DRAFTS, "w", encoding="utf-8") as fh:
        json.dump(drafts, fh, indent=2, ensure_ascii=False)


def generate(data_dir, only=None, redo=False):
    """Draft debriefs for puzzles that don't have an accepted one yet; save them to DRAFTS."""
    with open(os.path.join(data_dir, "map.json")) as fh:
        map_ = json.load(fh)
    with open(os.path.join(data_dir, "puzzles.json")) as fh:
        index = json.load(fh)
    drafts = load_drafts()
    pool = ModelPool(make_client())
    for entry in index["puzzles"]:
        pid = entry["id"]
        if (only and pid not in only) or (pid in drafts and not redo):
            continue
        with open(os.path.join(data_dir, entry["file"])) as fh:
            pz = json.load(fh)
        facts, numbers, ids = facts_for(map_, pz)
        prompt = f"Facts for this round:\n{facts}"
        text, model, problems = "", "", ["no answer"]
        for attempt in range(2):
            try:
                text, model = pool.ask(prompt if attempt == 0 else
                                       f"{prompt}\n\nYour previous answer had problems: {'; '.join(problems)}. "
                                       "Rewrite it using only the facts.")
            except RuntimeError as e:
                print(f"\nStopped: {e}. Accepted drafts so far are saved; run again later to finish.")
                save_drafts(drafts)
                return drafts
            problems = check(text, numbers, ids)
            if not problems:
                break
        print(f"--- {pid} ({pz['difficulty']}, correct: {pz['outcome']['correct_call']}, {model})")
        if problems:
            print(f"  REJECTED ({'; '.join(problems)}): {text}\n")
            continue
        print(f"  {text}\n")
        drafts[pid] = {"text": text, "model": model}
        save_drafts(drafts)
    return drafts


def apply(data_dir):
    """Write the reviewed drafts into ai.coach (no API calls)."""
    with open(os.path.join(data_dir, "puzzles.json")) as fh:
        index = json.load(fh)
    drafts = load_drafts()
    written = 0
    for entry in index["puzzles"]:
        if entry["id"] not in drafts:
            continue
        path = os.path.join(data_dir, entry["file"])
        with open(path) as fh:
            pz = json.load(fh)
        pz["ai"]["coach"] = drafts[entry["id"]]["text"]
        write(pz, entry["file"], data_dir)
        written += 1
    print(f"wrote ai.coach into {written} of {len(index['puzzles'])} puzzles")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draft Gemini debriefs, then write them into ai.coach.")
    ap.add_argument("--apply", action="store_true", help="write the saved drafts into the puzzle files (no API calls)")
    ap.add_argument("--only", help="comma-separated puzzle ids")
    ap.add_argument("--redo", action="store_true", help="redraft puzzles that already have a saved draft")
    ap.add_argument("--data", default=WEB_DATA_DIR, help="folder with puzzles.json (default web/public/data)")
    args = ap.parse_args()
    if args.apply:
        apply(args.data)
    else:
        d = generate(args.data, set(args.only.split(",")) if args.only else None, args.redo)
        print(f"{len(d)} drafts saved in {DRAFTS}. Review them, then: python -m pipeline.coach --apply")
