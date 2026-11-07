# RunwayZero

**Runway conflict detection under degraded surveillance.** HackRPI 2026.

> Missing data is not a clear runway.

RunwayZero replays real airport surface traffic (CMU Amelia-48, FAA SWIM data), injects sensor
faults (noise, dropout, delay), and compares conflict detectors: constant velocity, a Kalman
filter, and a learned model that knows when it is uncertain.

*Research and teaching testbed only. Not an operational or certified safety system.*

## Repo layout
```
contracts/            Shared data contract (source of truth for both halves)
  schema/*.schema.json  JSON Schemas, v1.0
  types.ts              TypeScript mirror; import from the web app
  fixtures/             MOCK data that matches the schemas
pipeline/             Python: data, fault injection, models, evaluation, export
  contracts.py          validate() + write(); every export goes through this
  gate_check.py         Amelia data gate (step 0)
  make_fixtures.py      regenerates contracts/fixtures/
  export.py             writes real data to web/public/data/ (tonight)
  tests/                pytest; contract tests run on fixtures AND web/public/data
scripts/
  sync-fixtures.mjs     copies fixtures into web/public/data
web/                  Next.js app (teammate creates it; see below)
  public/data/          the only folder the UI reads data from
data/                 local raw data (git-ignored under data/amelia/)
```

## Who owns what
| Area | Owner |
|---|---|
| `pipeline/`, `data/` | Harman |
| `web/` | Teammate |
| `contracts/` | Both: change only together (see `contracts/README.md`) |

## Setup

**Pipeline**
```
pip install -r requirements.txt
python -m pipeline.make_fixtures      # regenerate mock data
python -m pytest -q                   # contract tests
python pipeline/gate_check.py         # step 0: download + check Amelia data
```

**Web** (first time)
```
npx create-next-app@latest web --ts --app --eslint --src-dir --no-tailwind --import-alias "@/*"
node scripts/sync-fixtures.mjs        # mock data -> web/public/data
```
Then in `web/tsconfig.json`, add to `compilerOptions.paths`:
```json
"@contracts/*": ["../contracts/*"]
```
and import types with `import type { Replay } from "@contracts/types";`.
Load data with `fetch("/data/scenarios.json")`, then the `file` field of each replay entry.
