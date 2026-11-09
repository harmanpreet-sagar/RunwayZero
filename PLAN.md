# RunwayZero: HackRPI 2026 Build Plan

**Team:** Harman (data, models, evaluation) · Harnoor (front end)
**Pitch line:** *"Controllers clear runway crossings in seconds. Can you do it better than an AI? Play on real Boston Logan traffic."*
**Live:** [runway-zero.vercel.app](https://runway-zero.vercel.app/)

✅ = done · ⬜ = not done (yet)

> **Pivot:** RunwayZero started as a conflict-detection replay viewer and became a game: **"Can you beat the AI
> at clearing a runway crossing?"** The pipeline work (truth layer, faults, CV/Kalman/ML forecasting, evaluation)
> stayed the same; it now feeds an AI *player*. The contract went to **v1.1** (see `contracts/README.md`).

---

## 1. What we're building

A controller-training game on **real recorded Boston Logan traffic** (CMU Amelia, FAA SWIM data).
You play the controller; an AI plays against you.

**One round:**

1. Real traffic plays, then **freezes** when a vehicle is holding short of a runway.
2. You call **CLEAR** or **HOLD** and how sure you are: **50 / 75 / 95%**.
3. The AI makes the same call with its own confidence, from its forecast of the aircraft's path.
4. **Unfreeze.** The recording plays out. Did the vehicle and aircraft get close?
5. Both players are scored (**Brier score**). The AI shows its reasoning and a forecast cone, and a Gemini debrief explains the round.

**Difficulty:** Clear day → Busy hour → **Fog mode** (surveillance stale or missing; you only see where things were).

**Who it's for:** air traffic control students and airside vehicle drivers (fire/rescue, ops, maintenance)
practising crossing calls; secondary: safety / human-factors researchers (human vs. AI calibration).
**Not for:** controllers on live duty. It's a practice tool, not a certified simulator or an ATC system.

**The research question:** are humans or the AI better at knowing when they're unsure, and does human
overconfidence get worse in fog?

---

## 2. Architecture decisions

- **No backend server.** Python precomputes everything (AI calls, forecast cones, Gemini debriefs) to JSON;
  Next.js serves it statically from `web/public/data/`. Deployed on Vercel (Root Directory `web`).
- **The contract** lives in `contracts/` (v1.1, see `contracts/README.md`):
  - `schema/*.schema.json`: source of truth · `types.ts`: TS types · `scoring.ts`: the one scoring function
  - `fixtures/`: **mock data only** (`is_mock: true`). Never report these numbers.
- **Files the UI reads:** `map.json`, `puzzles.json`, `puzzle_<id>.json`, `metrics.json` (optional).
- **The UI writes:** one `Round` record per round, to localStorage.
- The UI reads puzzle filenames **from `puzzles.json`**; it never builds them.
- **`reveal`, `outcome`, `ai.coach` are the hidden future.** Never render or log them before lock-in.
- Units: metres, x = east, y = north. Time in seconds. Every `r` = 90% radius.
- When `status === "lost"`, `x/y` is the **last known** position. Draw it as a ghost, not as current.

### Puzzle rules (final)

- **Threat aircraft required.** A freeze is only used if a runway aircraft:
  1. is runway traffic (heading within 25° of the runway; on it, or up to 4 km beyond the end and faster than 40 kt);
  2. is on the map;
  3. is heading toward the crossing;
  4. is not already in the crossing zone;
  5. arrives within **30 s** at its current speed, or with takeoff acceleration (2 m/s²) if it's on the runway.
- **What-ifs.** Every real crossing in the hour was safe, so the vehicle's **own real crossing** is replayed from
  the freeze, with every aircraft exactly as recorded. Labelled in `source.vehicle_time_shift_s`; the WHAT IF tag
  is shown only after the reveal. (The first rule, "only cut moments where the vehicle really crossed", left no
  HOLD puzzles, so it was replaced.)
- **Conflict** = closest approach between the vehicle and runway traffic while the vehicle is in the zone,
  under **100 m** (328 ft). 50 m left too few HOLD moments.
- **Balance and spacing.** Freezes at least 5 s apart; HOLD capped at half and picked evenly in time.
  **Never pick puzzles by whether the AI gets them right.**
- **Difficulty:** every 3rd HOLD and every 3rd CLEAR is `fog` = `pipeline/faults.py` applied to `pre`
  (the question aircraft is never fully hidden, its ghost stays on the map, and at least 3 s of reports show
  before the dropout). The rest are ranked by objects on screen: top half `busy`, the rest `clear`.
  `reveal` is always the truth.
- **No leakage:** the ML model never trains on puzzle time windows. Any change to the puzzle set means a retrain.
- **The AI only sees `pre`.** It forecasts from the last observed state, exactly what the player saw.

---

## 3. Harman: data, models, evaluation, puzzles

- ✅ **Data check.** Amelia KBOS hour: 54 ground aircraft and 30 ground vehicles; 129,457 position reports.
- ✅ **Contract v1.1** merged, with tests and CI (Python tests, contract type-check, web lint + build).
- ✅ **Truth layer** (`pipeline/truth.py`). Clean tracks (km→m, axis swap, drop airborne points), rebuild the
  runways, find the active ones (15R/33L, 9/27), build `map.json` and the background, find every vehicle crossing.
- ✅ **Puzzle generator** (`pipeline/puzzles.py`). Cut 20 s of `pre` and 30 s of `reveal`, compute `outcome`, tag
  difficulty, write through `contracts.write()`. Final set: **20 puzzles** (10 HOLD / 10 CLEAR; 7 clear, 7 busy, 6 fog).
- ✅ **Runway traffic matches the web app** (`is_runway_user` = Harnoor's `isRunwayTraffic`), so taxiing aircraft
  never count as the question aircraft.
- ✅ **Threat-aircraft rule, spacing and HOLD cap** (§2). 25 of 161 holding-short moments pass; 20 kept.
- ✅ **Kalman filter** (`pipeline/kalman.py`). Constant acceleration, 400 sampled futures, cone calibrated on clean
  and gappy tracks; stops and speed caps so forecasts never reverse.
- ✅ **Fog mode** (`pipeline/faults.py`). Dropout 6–12 s, delay 2–3 s, 8 m noise.
- ✅ **ML model** (`pipeline/ml_model.py`). LightGBM on the Kalman filter's error in the aircraft's frame, 5–30 s
  ahead; 5-fold CV over 10-minute blocks with a 50 s purge; split-conformal radii. 13,616 training slices.
- ✅ **AI players** (`pipeline/ai_player.py`). CV, Kalman and ML each make a call; ML is the opponent, the others
  are `ai.baselines`. Template `ai.reasoning`.
- ✅ **Model evaluation** (`data/derived/ml_report.json`): held-out error by condition, coverage, feature importance.
- ✅ **Puzzle audit.** Contact sheet of every freeze, checked by eye; fixed freezes with the aircraft inside the
  square, already past it, too far away, or off the map.
- ✅ **Auto-fit camera.** At the freeze the scope frames the crossing, vehicle and question aircraft (ghost
  included), with room for their labels.
- ✅ **Merged main** (Gemini debriefs) into the new puzzle set without losing either.
- ✅ **Deployed** to Vercel and checked the live site at phone, tablet and laptop sizes.
- ⬜ `pipeline/evaluate.py` → real `metrics.json` (forecast error by data age, coverage, AI game scores) for the Stats page.
- ⬜ `pipeline/rounds.py` to summarise exported playtest rounds (human vs. AI Brier by difficulty, calibration).
- ⬜ Inference time per prediction.

---

## 4. Harnoor: front end (the game)

- ✅ **Mock data first.** Built against `puzzle_mock_*.json` so the front end never waited on the pipeline.
- ✅ **Map, scope and icons.** SVG surface scope with runways, tags and the OSM background; MOCK DATA badge when `is_mock`.
- ✅ **Core loop.** Shift picker → play `pre` → freeze at `decision_t` → CLEAR/HOLD + 50/75/95 → transmit → AI call,
  cone and reasoning → reveal → closest approach → both scores via `makeRound()`.
- ✅ **Info display and strips.** Runway traffic ETA, vehicle crossing time, margin, surveillance status.
- ✅ **Fog visuals.** `stale` = faded + "last seen X s ago", `lost` = dashed ghost.
- ✅ **Shifts.** Call sign, 5 rounds per shift, running score vs. the AI, end-of-shift screen, **Export rounds** button.
- ✅ **Stats page.** Human vs. AI from rounds saved on the device.
- ✅ **Show the ML player** and the other models' calls on each round.
- ✅ **Gemini debriefs** (`pipeline/coach.py`): facts in Python, Gemini writes, a checker rejects unknown numbers or
  IDs, human review. Debrief panel on the result screen.
- ✅ **Sign-in screen** with the pitch, the "training tool, not an ATC system" line, and the GitHub link.
- ✅ **Phones.** Stacked layout, pinch to zoom, tap for details, no sideways scrolling.
- ⬜ Shared stats across all players (needs a database, §6).

---

## 5. Both: playtest and submission

- ✅ Devpost entry: description, screenshots, tracks, GitHub and live links.
- ✅ Notion workspace with explainers and prepared judge answers.
- ⬜ Playtest with 10+ hackers; export their rounds.
- ⬜ Backup demo video.
- ⬜ Rehearse the pitch (§8) with one Fog-mode round.

**Devpost tracks:** #1/#2 Best Hack · Best Game Hack · Best Machine Learning / Data Science · Best In The Clouds ·
Best Artificial Intelligence Hack · [MLH] Best Use of Gemini API.

---

## 6. Add-ons

| Add-on | Status | Notes |
| --- | --- | --- |
| **Gemini coach** | ✅ Done | Precomputed into `ai.coach`; no live API calls. |
| **MongoDB Atlas** | ⬜ Not done | Would store `Round` records for shared stats: `POST /api/rounds`, `GET /api/stats` on Vercel, keep localStorage as fallback. About 2–3 hours. |
| **Domain** | ⬜ Not done | e.g. `runwayzero.tech`, pointed at Vercel. |
| Fish Audio voice | ⬜ Not done | Read the debrief or tower-style calls. |

---

## 7. Results and metrics

**Forecasting** (held out, error at 10 s): ML **25 m** vs Kalman 33 m vs constant velocity 33 m. Takeoff rolls:
ML **46 m** vs Kalman 173 m. 90% cone coverage: ML 90%, Kalman 91%, constant velocity 82%.

**AI players on the 20 puzzles:** ML **65%** right (Brier 0.240) · Kalman 55% (0.294) · constant velocity 55% (0.345).
In fog all three get 2 of 6.

**Still to measure:** human vs. AI from playtest rounds; calibration (how often 95% calls are actually right).

**Honesty rules**

- If the Kalman filter beats ML somewhere, report it. If humans beat the AI, report it.
- Players at the fair are hackers and judges, **not** trainees. Say so; it's a convenience sample.
- The outcome is what happened in the recording. Fog is simulated degradation on real tracks; every round is a what-if.
- 20 puzzles is a small sample; the held-out forecasting table is the stronger evidence.
- Never use fixture numbers.

---

## 8. Pitch (2–4 min at the project fair)

1. **Hook (20 s):** Severe runway incursions in the U.S. rose 44% in 2025. Controllers clear crossings in seconds.
   *Can you do it better than an AI?* (Respectful one-liner on the March 2026 LaGuardia collision as why crossings
   matter; **do not** imply its cause, the NTSB investigation is ongoing.)
2. **Play (90 s):** the judge plays one Fog-mode round on real Boston Logan traffic. Freeze → their call →
   AI's call and cone → reveal → scores → Gemini debrief.
3. **Results (40 s):** the forecasting table, and how the ML player compares with the Kalman and constant-velocity baselines.
4. **Limitations (15 s):** a practice tool, not a certified simulator or an ATC system; fog is simulated on real
   tracks; rounds are what-ifs; one hour at one airport.

**Prepared answers** (full versions in the Notion explainers)

- *"How do you know HOLD was right?"* The vehicle's real crossing is replayed against the real aircraft; HOLD is
  right when the closest approach was under 100 m while the vehicle was on the runway.
- *"Isn't 100 m arbitrary?"* Yes, it's a parameter; 50 m left too few HOLD moments in one hour.
- *"Does the AI see the future?"* No. It only gets the frames you saw, and it never trained on puzzle time windows.
- *"Why Brier score?"* It's the standard score for probability forecasts: it rewards being right *and* knowing how sure to be.
- *"Why not just a Kalman filter?"* It's one of the AI's baselines. ML beats it on takeoff rolls and landings; in fog all three struggle.

---

## 9. Working rules

- Commit small and often. Pull before you push.
- **Contract changes:** schema + `types.ts` (+ `scoring.ts`/`contracts.py`) + fixtures in one commit,
  `python -m pytest -q` green, and tell the other person.
- Before pushing data: `python -m pytest -q` (validates everything in `web/public/data`, including puzzle consistency).
- **Never run `npm run sync-fixtures` on a branch with real data**: it overwrites `web/public/data/` with mocks.
- After regenerating puzzles, rerun `pipeline.coach` and `--apply` so the debriefs match.
- Raw data stays in `data/amelia/` (git-ignored). Never commit it.
