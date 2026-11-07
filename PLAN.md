# RunwayZero: HackRPI 2026 Build Plan

**Team:** Harman (data, models, evaluation) · Harnoor (front end)
**Submission:** Devpost by **9:00 AM Sunday** (edits allowed until 11 AM). We aim to submit at **8:30 AM**.
**Pitch line:** *"Missing data is not a clear runway."*

---

## 1. What we're building

A testbed for **runway conflict detection under degraded surveillance**:
1. **Ground truth** = real aircraft and vehicle tracks from CMU's Amelia dataset (FAA SWIM data, Boston Logan).
2. **Observed state** = the same tracks with injected sensor faults: noise, dropout, delay (seeded, reproducible).
3. **Three detectors** predict 5 s ahead and flag conflicts in a protected runway-crossing zone:
   - `cv` constant velocity (naive baseline)
   - `kf` Kalman filter (the real baseline; its uncertainty grows as data goes stale)
   - `ml` LightGBM that corrects the Kalman filter, with conformal uncertainty by data age
4. **Scenarios:** cleared crossing during landing roll · vehicle tracking dropout · delayed data.
5. **UI:** replay on the airport map, true vs. observed, uncertainty circles, alerts, detector comparison, metrics page.

**What it is NOT:** an ATC system, a collision-prevention product, or a reconstruction of any real accident.

---

## 2. Architecture (decided, don't change)

- **No backend server.** Python precomputes everything to JSON. Next.js serves it statically from `web/public/data/`. Deploy = Vercel.
- **The contract** lives in `contracts/` (see `contracts/README.md`):
  - `schema/*.schema.json`: source of truth
  - `types.ts`: TS types (`import type { Replay } from "@contracts/types"`)
  - `fixtures/`: **mock data only** (`is_mock: true`). Never report these numbers.
- Files the UI reads: `map.json`, `scenarios.json`, `replay_<scenario>_s<seed>_d<dropout%>.json`, `metrics.json`.
- The UI reads replay filenames **from `scenarios.json`**; it never builds them.
- Units: metres, x = east, y = north. Time in seconds. `pred` = position at `t + pred_horizon_s`, `r` = 90% radius.
- When `obs.status === "lost"`, `obs.x/y` is the **last known** position. Draw it as a ghost, not as current.

---

## 3. Checkpoints

| Time | Checkpoint | Owner |
|---|---|---|
| **4:00 PM** | Data gate: GO (RunwayZero) / NO-GO (switch to ConnectionIQ) | Harman |
| **5:30 PM** | First real data (`map.json` + truth-only replay) in the UI | Both |
| **9:00 PM** | Kalman alerts showing in the UI | Both |
| **11:30 PM** | ML predictions showing in the UI | Both |
| **1:30 AM** | Metrics **frozen**. No number changes after this | Harman |
| **5:00 AM** | **Feature freeze**. Bug fixes only | Both |
| **8:30 AM** | **Submitted** | Both |

---

## 4. Harman: data, models, evaluation

| Time | Task | Done when |
|---|---|---|
| **3:05–4:00** | **Data gate.** `pip install -r requirements.txt` then `python pipeline/gate_check.py`. Check: file loads, Type 1 vehicles on the ground, runway and taxiways visible in `data/gate_check.png`, a runway crossing exists, map assets downloaded. | `GATE: PASS` and the plot looks right. **Otherwise switch to ConnectionIQ.** |
| 4:00–5:30 | **Truth layer.** Clean tracks (km to m, drop airborne points), pick a 10–15 min window with crossing traffic, define zone(s) around the runway crossing, build `map.json` from the Amelia graph/OSM data, export a truth-only replay via `pipeline/contracts.write()`. | **Handoff to Harnoor at 5:30.** `python -m pytest -q` passes on `web/public/data`. |
| 5:30–7:00 | **Fault injector and scenarios** (`pipeline/faults.py`, `pipeline/scenarios.py`). Gaussian noise; bursty dropout; delay; all seeded. *Cleared crossing* = time-shift a real vehicle track into a real landing roll. Label true conflicts from the truth layer. | Same seed = byte-identical output (pytest). |
| 7:00–9:00 | **Baselines and conflict engine** (`pipeline/kalman.py`, `pipeline/conflicts.py`). Last position, constant velocity, constant-velocity Kalman filter predicting through gaps. Conflict = uncertainty circle overlaps zone within 10 s. | `cv` and `kf` preds and alerts in the replays. **9 PM checkpoint.** |
| 9:00–11:30 | **Learned model** (`pipeline/model.py`). Training set across many seeds and dropout levels. Features: last few observations, data age, speed, heading, distance to zone, object type, KF state. Target: KF error at 1/3/5 s. LightGBM. **Split by seed and hour, never random rows.** Split-conformal radius per age bucket (`0-1s`, `1-3s`, `3s+`). | `ml` preds and alerts in the replays. **11:30 checkpoint.** |
| 11:30–1:30 | **Evaluation** (`pipeline/evaluate.py`). Everything in section 7. Save PNG plots for README/Devpost. Write `metrics.json`. | **Numbers frozen at 1:30.** |
| **1:30–3:00** | **Sleep** | |
| 3:00–5:00 | Export every scenario × seed × dropout combination; help connect it to the UI; fix bugs. | Full data set in `web/public/data`. |

---

## 5. Harnoor: front end

| Time | Task | Done when |
|---|---|---|
| 3:05–4:00 | **Setup.** `npx create-next-app@latest web --ts --app --eslint --src-dir --no-tailwind --import-alias "@/*"`, then `node scripts/sync-fixtures.mjs`, then add `"@contracts/*": ["../contracts/*"]` to `web/tsconfig.json` paths. Load `scenarios.json` and one replay. Use **plain SVG** (not PixiJS). | App renders mock data. |
| 4:00–6:00 | **Map and playback.** Draw runways (polygons), taxiways (polylines), zones (highlighted). Aircraft and vehicle icons. Play / pause / reset / 1×, 2×, 5× / timeline scrubber with `events` as markers. Show a **MOCK DATA** badge when `is_mock`. | Mock traffic moves smoothly. |
| 6:00–8:30 | **Real data plus sensor failures.** Switch to Harman's real files (5:30). True vs. observed toggle. `stale` = faded icon + "last seen Xs ago"; `lost` = dashed ghost at last known position. Uncertainty circles from `pred.<detector>.r`. | Real traffic; failures obvious at a glance. |
| 8:30–11:00 | **Core demo.** Scenario / seed / dropout picker built from `scenarios.json`. **Detector comparison panel:** three rows (`cv`, `kf`, `ml`) of alert timelines vs. the true conflict. Alert banner. **Pause-and-inspect:** click an object to see true vs. observed vs. each prediction. | The demo works end to end. |
| **11:00–12:30** | **Sleep** | |
| 12:30–3:00 | **Metrics page** from `metrics.json` (Recharts): recall vs. false alerts (PR curve), error by data age, coverage vs. 90% target, recall vs. dropout. | Results page reads real `metrics.json`. |
| 3:00–5:00 | **Polish and deploy.** Control-room styling, intro screen (stats + pitch line + "research testbed, not an operational system"), deploy to Vercel. | Live URL works on another laptop. |

**Harnoor never waits on Harman.** If real data is late, keep building on fixtures.

---

## 6. Both: 5:00 AM to submission

| Time | Task | Owner |
|---|---|---|
| 5:00 | **Feature freeze** | Both |
| 5:00–7:00 | Run the full demo 3×; fix bugs; record a **backup demo video** in case of wifi trouble | Harnoor records; both rehearse |
| 7:00–8:30 | README: method, results table, limitations | Harman |
| 7:00–8:30 | Devpost entry, screenshots, opt into tracks | Harnoor |
| **8:30** | **Submit** | Both |

**Devpost tracks:** #1/#2 Best Hack · Best Machine Learning / Data Science · Best In The Clouds.

---

## 7. Metrics to capture (Harman, frozen at 1:30 AM)

**Model**
- Position error at 1/3/5 s by data age (`0-1s`, `1-3s`, `3s+`) for cv, kf, ml
- % improvement of ml over kf and over cv
- Conformal coverage vs. 90% target, by age bucket
- Inference time per prediction

**Detection**
- Recall, precision, false alerts per hour, median lead time, PR-AUC per detector
- PR curve (precision vs. recall across thresholds), with the chosen operating point
- Recall at 0 / 30 / 60% dropout (robustness)
- Per-scenario recall and false alerts

**Data and system**
- Trajectories, position reports and airport-hours used
- Number of scenarios × seeds × dropout levels
- Test count

**Honesty rules:** if the Kalman filter beats ML somewhere, report it. Never call simulated results "real-world." Never use fixture numbers.

---

## 8. If we fall behind: cut in this order

1. Delayed-data scenario (keep cleared crossing and dropout)
2. Learned uncertainty model (use KF uncertainty + conformal)
3. Fancy animation (static SVG frames are fine)
4. **Never cut:** the Kalman baseline and the evaluation table.

**If the data gate fails at 4:00:** switch to ConnectionIQ (BTS flight data). Harnoor's app setup and playback controls carry over.

---

## 9. Pitch (2–4 min at the project fair)

1. **Hook (20 s):** Severe runway incursions in the U.S. rose 44% in 2025. The FAA's most effective fix, fixing taxiway geometry, cut incursions 77% at treated sites, but detection still depends on what the sensors can see.
2. **Respectful one-liner** about the March 2026 LaGuardia runway collision as why crossings matter. **Do not** imply its cause; the NTSB investigation is ongoing.
3. **Demo (90 s):** normal traffic → tracking dropout on a crossing vehicle → constant velocity goes blind, Kalman widens its uncertainty, ML alerts with N s lead → pause, show true vs. observed.
4. **Results (40 s):** metrics page: error by data age, coverage, recall vs. false alerts.
5. **Limitations (15 s):** faults are simulated on real tracks; research testbed, not an operational system.

**Prepared answers**
- *"Why not a Kalman filter?"* It's our main baseline; here's where ML beats it and where it doesn't.
- *"Does this replace tower systems?"* No. It's a testbed for how detection degrades when surveillance degrades.
- *"Isn't the data synthetic?"* The trajectories are real FAA SWIM data; only the sensor faults are injected, which is how you test robustness.

---

## 10. Working rules

- Commit small and often to `main`. Pull before you push.
- **Contract changes:** schema + `types.ts` + fixtures in one commit, `python -m pytest -q` green, and tell the other person. None after 3 PM without both agreeing.
- Before pushing data: `python -m pytest -q` (validates everything in `web/public/data`).
- Raw data stays in `data/amelia/` (git-ignored). Never commit it.
