# Data contract (v1.0)

The pipeline (Python) and the UI (Next.js) talk **only** through these JSON files,
served statically from `web/public/data/`. No backend server.

| File | Schema | TS type | Written by |
|---|---|---|---|
| `map.json` | `schema/map.schema.json` | `AirportMap` | pipeline |
| `scenarios.json` | `schema/scenarios.schema.json` | `ScenarioIndex` | pipeline |
| `replay_<scenario>_s<seed>_d<dropout%>.json` | `schema/replay.schema.json` | `Replay` | pipeline |
| `metrics.json` | `schema/metrics.schema.json` | `Metrics` | pipeline |

## Conventions
- **Units:** metres. Local Cartesian frame, x = east, y = north, origin = airport reference point.
- **Time:** seconds from replay start; one frame every `dt_s` (1.0).
- **Predictions:** every `pred` is the position predicted for `t + pred_horizon_s`; `r` is the 90% radius.
- **Observations:** when `obs.status` is `"lost"`, `obs.x/y` is the **last known** position. Never draw it as current.
- **Detectors:** `cv` = constant velocity, `kf` = Kalman filter, `ml` = learned model.
- **Filenames:** the UI reads replay filenames from `scenarios.json`. It never builds them.
- **`is_mock: true`:** the UI should show a visible "MOCK DATA" badge. Mock numbers are never reported.

## Changing the contract
1. Edit the JSON schema (source of truth).
2. Mirror it in `types.ts`.
3. Update `pipeline/make_fixtures.py` and regenerate: `python -m pipeline.make_fixtures`.
4. `python -m pytest -q` must pass.
5. All in **one commit**, and tell your teammate. No contract changes after 3 PM without both agreeing.
