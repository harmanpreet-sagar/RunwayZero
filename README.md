# RunwayZero

**Clear it or hold it? Can you make the call better than an AI?** Built at HackRPI 2026.

**Play it:** [runway-zero.vercel.app](https://runway-zero.vercel.app/)

RunwayZero is a runway-crossing training game on **real recorded surface traffic at Boston Logan (KBOS)**.
Each round freezes when a vehicle is holding short of an active runway. You call **CLEAR** or **HOLD** and
say how sure you are (50 / 75 / 95%). An ML forecaster makes the same call and shows its reasoning and
forecast cone. Then the recording plays out, and you're both scored on what really happened.

*A practice tool for controllers in training and airside drivers. Not a certified simulator or an ATC system.*

## How a round works

1. Real traffic plays, then **freezes** with a vehicle at the hold line and a runway aircraft heading for the crossing.
2. You transmit **CLEAR** or **HOLD** with a confidence.
3. The AI gives its call, its reasoning, and a forecast cone drawn on the scope.
4. The **recording plays out**. HOLD was right if the vehicle and the aircraft came within the safe distance
   (100 m / 328 ft) while the vehicle was on the runway.
5. Both players get a **Brier score** (confident and right scores high; confident and wrong scores low), and a
   **Gemini debrief** explains what happened and what to watch for next time.

**Conditions:** Clear day · Busy hour · **Fog mode** (position reports go stale or drop out for 6–12 s, so you
only see where aircraft *were*).

## How it's built

```
Amelia KBOS recording ──▶ truth layer ──▶ puzzle generator ──▶ AI players ──▶ Gemini debriefs ──▶ JSON ──▶ Next.js game
  (1 h, 129k reports)     (map, runways,   (threat-aircraft     (CV, Kalman,   (fact-checked,              (static, Vercel)
                           crossings)       rule, what-ifs)      LightGBM)      precomputed)
```

- **Truth layer** (`pipeline/truth.py`): cleans one hour of CMU Amelia-48 data (FAA SWIM), converts its
  km / x-north axes to metres / x-east, rebuilds the runways, finds the active ones and every vehicle crossing.
- **Puzzles** (`pipeline/puzzles.py`): a freeze is used only if a **threat aircraft** passes five checks: it's
  runway traffic, it's on the map, it's heading toward the crossing, it's not already in the crossing, and it
  arrives within 30 s (takeoff acceleration counted). Every real crossing in the hour was safe, so rounds are
  **what-ifs**: the vehicle's own real crossing is replayed from the freeze, with every aircraft exactly as recorded.
  HOLD rounds are capped at half and picked evenly in time, never by whether the AI gets them right.
- **Fog** (`pipeline/faults.py`): dropout, delay and noise on the frames before the freeze. The reveal is always the truth.
- **Three forecasters** (`pipeline/kalman.py`, `pipeline/ml_model.py`, `pipeline/ai_player.py`):
  1. constant velocity (baseline);
  2. a constant-acceleration **Kalman filter** that samples 400 possible futures, calibrated on clean and gappy tracks;
  3. **LightGBM** learning the Kalman filter's error in the aircraft's own frame, 5–30 s ahead. Validated with
     5-fold cross-validation over 10-minute time blocks with a 50 s purge, never trained on puzzle windows.
     Its cones are split-conformal (90%).

  Each forecaster turns its futures into a CLEAR/HOLD call with 50/75/95% confidence. The game's opponent is the ML player.
- **Debriefs** (`pipeline/coach.py`): Gemini writes 2–3 sentences from facts computed in Python. Any number or ID
  not in the facts is rejected, and every debrief was reviewed by hand. Gemini never makes the call or the score.
- **Game** (`web/`): Next.js + TypeScript, SVG surface scope, works on phones. Everything is precomputed, so
  there's no server: the site is static files on Vercel.
- **Contract** (`contracts/`): JSON Schemas shared by the pipeline and the app, with tests on both sides in CI.

## Results

**Forecasting** (held out, 13,616 track slices, error at 10 s ahead):

| | Constant velocity | Kalman | **LightGBM** |
| --- | --- | --- | --- |
| All | 33 m | 33 m | **25 m** |
| Speeding up (takeoff rolls) | 209 m | 173 m | **46 m** |
| Slowing down (landings) | 110 m | 86 m | **33 m** |
| Fog-like data gaps | 48 m | 48 m | **36 m** |
| 90% cone contains the truth | 82% | 91% | **90%** |

**Making the call** on the 20 puzzles (10 HOLD / 10 CLEAR):

| | Right | Brier (lower is better) |
| --- | --- | --- |
| **LightGBM** | **13 / 20 (65%)** | **0.240** |
| Kalman | 11 / 20 (55%) | 0.294 |
| Constant velocity | 11 / 20 (55%) | 0.345 |

Outside fog, the ML player is right on 11 of 14. **In fog, all three get 2 of 6**, and the ML player is sometimes
confidently wrong, which is exactly the situation Fog mode trains.

**Limitations:** one hour at one airport; 20 puzzles is a small sample (the forecasting table is the stronger
evidence); every round is a what-if; fog is simulated on real tracks; the 100 m safe distance is a chosen parameter.

## Running it

**The game**
```
cd web
npm install
npm run dev        # http://localhost:3000, reads web/public/data/
```
Don't run `npm run sync-fixtures` unless you want mock data: it overwrites the real puzzles in `web/public/data/`.

**The pipeline** (Python 3.12)
```
pip install -r requirements.txt
python pipeline/gate_check.py          # download and check the Amelia KBOS hour (data/amelia/, git-ignored)
python -m pipeline.truth               # map.json + background
python -m pipeline.puzzles             # trains the ML model, writes puzzles.json + puzzle_*.json
python -m pipeline.coach               # Gemini debrief drafts (needs GEMINI_API_KEY); read them
python -m pipeline.coach --apply       # writes the reviewed debriefs into the puzzles
python -m pytest -q                    # contract, truth, Kalman/fog, ML and puzzle tests
```
`pipeline.puzzles --skip-training` regenerates with the saved model. Regenerating puzzles wipes the debriefs,
so rerun the coach afterwards (delete `data/derived/coach_drafts.json` first if puzzle IDs were reused).

## Repo layout

```
contracts/     JSON Schemas (v1.1), TypeScript types, the shared scoring function, mock fixtures
pipeline/      Python: truth layer, puzzles, fog, Kalman, ML model, AI players, Gemini coach, tests
web/           Next.js game; public/data/ is the only data it reads
scripts/       sync-fixtures.mjs (mock data only)
data/          raw Amelia data and derived files (git-ignored)
.github/       CI: Python tests, contract type-check, web lint + build
PLAN.md        what we built and how we split the work
```

## Team

- **Harman**: data, truth layer, puzzles, Kalman filter, ML model, evaluation, deployment
- **Harnoor**: the game (Next.js), Gemini debriefs, phone layout

Data: CMU Amelia-48 airport surface movement dataset (FAA SWIM). Map tiles © OpenStreetMap contributors.
