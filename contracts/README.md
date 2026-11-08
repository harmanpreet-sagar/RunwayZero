# Data contract (v1.1)

The pipeline (Python) and the UI (Next.js) talk **only** through these JSON files,
served statically from `web/public/data/`. No backend server.

**v1.1 = the game pivot.** The core is now the puzzle files. Replays + `scenarios.json` are kept
unchanged as a secondary "watch mode"; no new work is planned on them.

| File | Schema | TS type | Written by | Status |
|---|---|---|---|---|
| `map.json` | `schema/map.schema.json` | `AirportMap` | pipeline | core |
| `puzzles.json` | `schema/puzzles.schema.json` | `PuzzleIndex` | pipeline | **core (new)** |
| `puzzle_<id>.json` | `schema/puzzle.schema.json` | `Puzzle` | pipeline | **core (new)** |
| `metrics.json` | `schema/metrics.schema.json` | `Metrics` | pipeline | core (`game` section new) |
| `round_*.json` / localStorage / MongoDB | `schema/round.schema.json` | `Round` | **UI** | **core (new)** |
| `scenarios.json` | `schema/scenarios.schema.json` | `ScenarioIndex` | pipeline | secondary |
| `replay_<scenario>_s<seed>_d<dropout%>.json` | `schema/replay.schema.json` | `Replay` | pipeline | secondary |

`scoring.ts` holds the one scoring function both sides use (Python mirror: `pipeline/contracts.py`).

## How a round works (UI)
1. Load `puzzles.json` → pick a difficulty → load a puzzle by its `file` field (never build filenames).
2. Play `pre` frames up to `decision_t`. **Freeze.** Highlight `vehicle_id`, `aircraft_id` and `zone_id`; show `question`.
3. Player picks **CLEAR / HOLD** and **50 / 75 / 95%**, then locks in.
4. Show the AI's `ai.call` + `ai.confidence`, draw `ai.paths` as an uncertainty cone, show `ai.reasoning`.
5. Play `reveal` (the truth). Mark the closest approach at `outcome.min_sep_t`. Show `outcome.note`.
6. Score both with `makeRound()` from `scoring.ts` and save the `Round`.

**`reveal`, `outcome` and `ai.coach` are the hidden future.** They are in the file, so the UI must not
render them (or log them to the console) before the player locks in.

## Scoring
```
p_safe = confidence if CLEAR, 1 - confidence if HOLD
brier  = (p_safe - (conflict ? 0 : 1))^2      0 = perfect, 1 = confidently wrong
points = round(100 * (1 - brier))
```
CLEAR @95% and safe → 100 · CLEAR @95% and conflict → 10 · anything @50% → 75.
The AI is scored on its **snapped** `confidence` (same rules as humans). Its raw `p_safe` is only used
for `metrics.game.brier_raw`.

## What "conflict" means
A puzzle is cut from a moment where the vehicle **really crossed** in the recording, so the reveal shows
what happened. `conflict = true` when the closest approach between the vehicle and `aircraft_id`, while
the vehicle was inside the crossing zone, was below `conflict_threshold_m` (in `puzzles.json`, copied into
each `outcome.threshold_m`). Then HOLD was the right call.

## Conventions
- **Units:** metres. Local Cartesian frame, x = east, y = north, origin = airport reference point.
- **Time:** seconds from replay/puzzle start; one frame every `dt_s` (1.0).
- **Uncertainty:** every `r` is a 90% radius in metres. Draw `ai.paths` as circles that grow along the path.
- **Observations:** when `status` is `"lost"`, `x/y` is the **last known** position. Never draw it as current.
  In fog mode `pre` holds observed (degraded) positions; `reveal` is always the truth.
- **Detectors:** `cv` = constant velocity, `kf` = Kalman filter, `ml` = learned model. `ai.detector` is the one playing.
- **`is_mock: true`:** show a visible "MOCK DATA" badge. Mock numbers are never reported.
  The mock puzzles are tuned to give the UI variety: 3 conflicts / 3 safe, two 75% AI calls, and the
  AI is confidently **wrong** on `mock_06` (fog).

## Changing the contract
1. Edit the JSON schema (source of truth).
2. Mirror it in `types.ts` (and `scoring.ts` + `pipeline/contracts.py` if scoring changes).
3. Update `pipeline/make_fixtures.py` / `pipeline/mock_puzzles.py` and regenerate: `python -m pipeline.make_fixtures`.
4. `python -m pytest -q` must pass.
5. All in **one commit**, and tell your teammate. No contract changes after 3 PM without both agreeing
   (v1.1 is the agreed game-pivot change).
