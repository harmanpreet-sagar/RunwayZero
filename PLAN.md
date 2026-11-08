# RunwayZero: HackRPI 2026 Build Plan (v2: the game)

**Team:** Harman (data, models, evaluation) · Harnoor (front end)
**Submission:** Devpost by **9:00 AM Sunday** (edits allowed until 11 AM). We aim to submit at **8:30 AM**.
**Pitch line:** *"Controllers clear runway crossings in seconds. Can you do it better than an AI? Play on real Boston Logan traffic."*

> **Pivot (Sat 5:15 PM):** RunwayZero is now a game: **"Can you beat the AI at clearing a runway crossing?"**
> The pipeline work (truth layer, faults, CV/Kalman/ML forecasting, evaluation) is unchanged. It now
> feeds an AI *player* instead of a passive replay viewer. Contract bumped to **v1.1** (see `contracts/README.md`).

---

## 1. What we're building

A controller-training game on **real recorded Boston Logan traffic** (CMU Amelia, FAA SWIM data).
You play the controller; an AI plays against you.

**One round (~20 s):**

1. Real traffic plays, then **freezes** when a vehicle is holding short of a runway.
2. You call **CLEAR** or **HOLD** and how sure you are: **50 / 75 / 95%**.
3. The AI makes the same call with its own confidence, from its forecast of the aircraft's path.
4. **Unfreeze.** The real recording plays out. Did the vehicle and aircraft get close?
5. Both players are scored (**Brier score**). The AI shows its reasoning: predicted aircraft path with an uncertainty cone.

**Difficulty:** Clear day → Busy hour → **Fog mode** (surveillance stale or missing; you only see where things were).

**Who it's for:** air traffic control students and airside vehicle drivers (fire/rescue, ops, maintenance)
practising crossing calls; secondary: safety / human-factors researchers (human vs. AI calibration).
**Not for:** controllers on live duty. It's a practice tool, not a certified simulator or an ATC system.

**The research question we can answer at the fair:** are humans or the AI better at knowing when they're unsure,
and does human overconfidence get worse in fog?

---

## 2. Architecture (decided)

- **No backend server** for the core game. Python precomputes everything to JSON; Next.js serves it statically
  from `web/public/data/`. Deploy = Vercel. (Exception, only if we add MongoDB: two Next.js API routes. See §8.)
- **The contract** lives in `contracts/` (v1.1, see `contracts/README.md`):
  - `schema/*.schema.json`: source of truth · `types.ts`: TS types · `scoring.ts`: the one scoring function
  - `fixtures/`: **mock data only** (`is_mock: true`). Includes 6 mock puzzles. Never report these numbers.
- **Core files the UI reads:** `map.json`, `puzzles.json`, `puzzle_<id>.json`, `metrics.json`.
- **The UI writes:** one `Round` record per round (localStorage now; MongoDB if we add it).
- **Secondary (no new work):** `scenarios.json` + `replay_*.json` from v1.0, kept as an optional "watch mode".
- The UI reads puzzle filenames **from `puzzles.json`**; it never builds them.
- **`reveal`, `outcome`, `ai.coach` are the hidden future.** Never render or log them before lock-in.
- Units: metres, x = east, y = north. Time in seconds. Every `r` = 90% radius.
- When `status === "lost"`, `x/y` is the **last known** position. Draw it as a ghost, not as current.

### Puzzle rules (Harman's generator must follow these)

- **Only cut moments where the vehicle really crossed.** The reveal must show the consequence of crossing.
  If the vehicle held in the recording, we can't know what CLEAR would have done, so skip it.
- **Decision point** = ~8 s before the vehicle enters the crossing zone (it's at or near the hold-short line).
- **Conflict** = closest approach between the vehicle and the aircraft, while the vehicle is in the zone,
  is below `conflict_threshold_m`. **Pick 50 m or 100 m by 8 PM** from the class balance (the hour has 23 pairs
  within 50 m, 115 within 100 m). Aim for **≥30% HOLD-correct puzzles**; otherwise "always CLEAR" wins.
- **Difficulty:** `clear` = few other objects nearby, `busy` = many, `fog` = same kind of moment with
  `pipeline/faults.py` applied to the `pre` frames (aircraft dropout/delay). `reveal` is always the truth.
- **No leakage:** the ML model never trains on time windows used for puzzles (hold them out by time block).
- **The AI only sees `pre`.** It forecasts from the last observed state, exactly what the player saw.

---

## 3. Checkpoints

| Time | Checkpoint | Owner |
| --- | --- | --- |
| ~~4:00 PM~~ | ~~Data gate~~ **Passed** (54 ground aircraft, 30 ground vehicles in the hour) | Harman |
| **5:30 PM** | Contract v1.1 merged. Harnoor builds the game loop on mock puzzles | Both |
| **6:30 PM** | Real `map.json` + truth layer | Harman |
| **8:00 PM** | **First real puzzles** (Clear + Busy), AI = constant velocity, playable in the UI | Both |
| **10:00 PM** | Kalman AI + **Fog mode** puzzles | Both |
| **12:00 AM** | **ML AI** playing every puzzle | Both |
| **1:30 AM** | Metrics **frozen** (model + AI game scores). No number changes after this | Harman |
| **3:30–4:30 AM** | **Playtest:** get 10+ other hackers to play; export rounds | Both |
| **5:00 AM** | **Feature freeze**. Bug fixes only | Both |
| **8:30 AM** | **Submitted** | Both |

---

## 4. Harman: data, models, evaluation, puzzles

| Time | Task | Done when |
| --- | --- | --- |
| 5:15–5:30 | **Merge contract v1.1.** `python -m pytest -q` green; tell Harnoor to re-run `node scripts/sync-fixtures.mjs`. | Harnoor has mock puzzles. |
| 5:30–6:30 | **Truth layer.** Clean tracks (km→m, drop airborne points), build `map.json` from the Amelia graph/OSM with a **zone at every runway crossing** the vehicles use. | Real `map.json` in the UI. |
| 6:30–8:00 | **Puzzle generator v0** (`pipeline/puzzles.py`). Find every real crossing (vehicle track enters a runway zone); pair it with the nearest aircraft on/approaching that runway; cut at the decision point; save 20 s of `pre`, ~30 s of `reveal`; compute `outcome`; tag difficulty. AI = **cv** for now (`snap_confidence()` turns p_safe into the call). Choose the threshold (§2). Write via `contracts.write()`. | `puzzles.json` + 20–40 real puzzles validate; Harnoor plays one. **8 PM checkpoint.** |
| 8:00–10:00 | **Kalman + fog** (`pipeline/kalman.py`, `pipeline/faults.py`). KF forecast of the aircraft from the last observed state, radius grows with data age. `p_safe` = P(closest approach while vehicle is in zone ≥ threshold), Gaussian on the forecast. Fog puzzles: aircraft dropout/delay in `pre`. Template `ai.reasoning`. | KF AI + fog puzzles in the UI. **10 PM checkpoint.** |
| 10:00–12:00 | **Learned model** (`pipeline/model.py`). LightGBM correcting the KF. Features: last few observations, data age, speed, heading, distance to zone, object type, KF state. Target: KF error at 1/3/5 s (extend to the reveal horizon if time). **Split by time block, puzzle windows held out.** Split-conformal radius per age bucket. `ai.detector = "ml"`; cv and kf go in `ai.baselines`. | ML AI on every puzzle. **12 AM checkpoint.** |
| 12:00–1:30 | **Evaluation** (`pipeline/evaluate.py`). Everything in §7. PNG plots for README/Devpost. Write `metrics.json`. | **Numbers frozen at 1:30.** |
| **1:30–3:00** | **Sleep** | |
| 3:00–5:00 | Playtest with other hackers (§6); `pipeline/rounds.py` summarises exported rounds (human vs. AI Brier by difficulty, calibration). Bug fixes. | First human-vs-AI numbers. |

---

## 5. Harnoor: front end (the game)

| Time | Task | Done when |
| --- | --- | --- |
| now | **Pull, then** `node scripts/sync-fixtures.mjs` to get `puzzles.json` + `puzzle_mock_*.json`. Import `Puzzle`, `PuzzleIndex`, `Round` from `@contracts/types` and `makeRound` from `@contracts/scoring`. | Mock puzzles load. |
| now–6:30 | **Map, replay, icons** (what's already in progress). Reuse the renderer for puzzle frames. **MOCK DATA** badge when `is_mock`. | Mock puzzle `pre` frames animate. |
| 6:30–9:00 | **Core loop:** difficulty picker (from `puzzles.json`) → play `pre` → **freeze** at `decision_t` (highlight vehicle, aircraft, zone; show `question`) → CLEAR/HOLD + 50/75/95 buttons → lock in → AI's call + `ai.paths` cone + `ai.reasoning` → play `reveal` → closest-approach marker at `outcome.min_sep_t` + `outcome.note` → both scores via `makeRound()`. | A full round plays end to end on every mock puzzle (`mock_06`: AI is wrong). |
| 9:00–11:00 | **Session + leaderboard + fog.** Nickname, 5 rounds per game (mix of difficulties), running score vs. the AI, end screen. Save each `Round` to localStorage; leaderboard from it; **"Export rounds" button** (downloads JSON). Fog visuals: `stale` = faded + "last seen Xs ago", `lost` = dashed ghost. | A judge can play a full 5-round game. |
| **11:00–12:30** | **Sleep** | |
| 12:30–3:00 | **Results page.** Live: human vs. AI average Brier and points by difficulty (from saved rounds), calibration chart (stated confidence vs. how often right). From `metrics.json`: AI Brier/accuracy per detector (`game`), forecast error by data age, coverage. Hide any optional section that's missing. | Results page reads real `metrics.json` + local rounds. |
| 3:00–5:00 | **Polish and deploy.** Tower-style styling, intro screen (pitch line + "training tool, not an operational system"), deploy to Vercel. | Live URL works on another laptop. |

**Harnoor never waits on Harman.** If real puzzles are late, keep building on mock puzzles.

---

## 6. Both: playtest, then submission

The fair (judging) happens **after** the 8:30 AM submission, so Devpost can only quote overnight playtest
numbers. Get real ones early:

| Time | Task | Owner |
| --- | --- | --- |
| 3:30–4:30 | **Playtest:** 10+ hackers play one game each on the deployed URL; export rounds | Both |
| 5:00 | **Feature freeze** | Both |
| 5:00–7:00 | Run the demo 3×; fix bugs; record a **backup demo video** | Harnoor records; both rehearse |
| 7:00–8:30 | README: method, results table (model + AI-vs-human from playtest), limitations | Harman |
| 7:00–8:30 | Devpost entry, screenshots, opt into tracks | Harnoor |
| **8:30** | **Submit** | Both |
| Fair | Every judge plays → live counter on the results page ("N players so far: humans X, AI Y") | Both |

**Devpost tracks (core):** #1/#2 Best Hack · **Best Game Hack** · Best Machine Learning / Data Science ·
Best In The Clouds · Best Artificial Intelligence Hack.
**Add-on tracks (only if built, §8):** [MLH] Best Use of MongoDB Atlas · [MLH] Best Use of Gemini API ·
[MLH] Best Domain Name (GoDaddy Registry).

---

## 7. Metrics to capture (Harman, frozen at 1:30 AM)

**Model (forecasting the aircraft)**

- Position error at 1/3/5 s by data age (`0-1s`, `1-3s`, `3s+`) for cv, kf, ml → `error_by_age`
- % improvement of ml over kf and over cv
- Conformal coverage vs. 90% target, by age bucket → `coverage`
- Inference time per prediction

**Game (each detector playing every puzzle as the AI)** → `metrics.game`

- Accuracy and Brier (snapped 50/75/95, same rules as humans) per difficulty
- Raw-probability Brier (`brier_raw`) per difficulty
- Puzzle counts: total, per difficulty, % HOLD-correct

**Humans (from playtest rounds, not frozen; reported separately)**

- Human vs. AI mean Brier and points per difficulty; n players, n rounds
- Calibration: for each confidence level, how often humans were right (overconfidence = 95% calls right much less than 95% of the time)

**Data and system**

- Trajectories, position reports and airport-hours used; number of real crossings found; test count

**Honesty rules**

- If the Kalman filter beats ML somewhere, report it. If humans beat the AI, report it.
- Playtest/fair players are hackers and judges, **not** trainees. Say so; it's a convenience sample.
- The outcome is what happened in the recording. Fog is simulated degradation on real tracks.
- Never use fixture numbers.

---

## 8. Add-ons (only after the core loop works, ~11 PM)

| Add-on | Cost | How |
| --- | --- | --- |
| **MongoDB Atlas** | ~1 h | Store `Round` records + leaderboard. Two Next.js API routes on Vercel: `POST /api/rounds` (validate shape), `GET /api/leaderboard`. Keep localStorage as fallback if wifi dies. |
| **Gemini coach** | ~1 h | **Precompute** at export time: pipeline asks Gemini to explain each puzzle's outcome in plain English, stored in `ai.coach` (already in the schema). Shown after the reveal. No live API calls during the demo. |
| **Domain** | 15 min | Register e.g. `runwayzero.tech`, point at Vercel. |
| Fish Audio voice | stretch | Read the coach line or tower-style calls. Only if everything else is done. |

---

## 9. If we fall behind: cut in this order

1. Watch mode / legacy replays (already deprioritized)
2. Add-ons (§8)
3. Busy hour as a separate level (merge into Clear day)
4. Learned uncertainty model (use KF uncertainty + conformal)
5. Fancy animation (static frames are fine)
6. **Never cut:** the freeze → decide → reveal loop on real puzzles, Brier scoring, the Kalman AI, and the evaluation table.

---

## 10. Pitch (2–4 min at the project fair)

1. **Hook (20 s):** Severe runway incursions in the U.S. rose 44% in 2025. Controllers clear crossings in seconds.
   *Can you do it better than an AI?* (Respectful one-liner on the March 2026 LaGuardia collision as why crossings
   matter; **do not** imply its cause, the NTSB investigation is ongoing.)
2. **Play (90 s):** the judge plays one Fog-mode round on real Boston Logan traffic. Freeze → their call →
   AI's call and cone → reveal → scores.
3. **Results (40 s):** human vs. AI Brier so far (live counter), calibration chart, how the ML AI compares
   with the constant-velocity and Kalman baselines.
4. **Limitations (15 s):** a practice tool, not a certified simulator or an ATC system; fog is simulated on real
   tracks; players so far are hackers and judges, not trainees.

**Prepared answers**

- *"How do you know HOLD was right?"* We only use moments where the vehicle really crossed, so the recording shows
  how close they got. HOLD is right when the closest approach was under N m.
- *"Isn't N m arbitrary?"* Yes, it's a parameter; here's the class balance at 50 and 100 m.
- *"Does the AI see the future?"* No. It only gets the frames you saw, and it never trained on puzzle time windows.
- *"Why Brier score?"* It's the standard score for probability forecasts: it rewards being right *and* knowing how sure to be.
- *"Why not just a Kalman filter?"* It's one of the AI's baselines; here's where ML beats it and where it doesn't.

---

## 11. Working rules

- Commit small and often. Pull before you push.
- **Contract changes:** schema + `types.ts` (+ `scoring.ts`/`contracts.py`) + fixtures in one commit,
  `python -m pytest -q` green, and tell the other person. v1.1 is agreed; no further changes without both agreeing.
- Before pushing data: `python -m pytest -q` (validates everything in `web/public/data`, including puzzle consistency).
- Raw data stays in `data/amelia/` (git-ignored). Never commit it.
