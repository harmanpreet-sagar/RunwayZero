# Runway Zero: web app

The game UI: a tower controller's position (flight strips, surface scope, information display,
radio console) where you call CLEAR or HOLD on real Boston Logan crossings and play against the AI.
Static Next.js app, no backend: it reads the contract files in `public/data/`.

## Run it

```
npm install
npm run sync-fixtures   # mock data -> public/data (works with no pipeline)
npm run dev             # http://localhost:3000
```

Real data instead of mocks (from the repo root, after `python pipeline/gate_check.py`):

```
python -m pipeline.truth     # map.json + kbos_bg.jpg
python -m pipeline.puzzles   # puzzles.json + puzzle_*.json
```

Optional Gemini debriefs (fills `ai.coach`; needs `GEMINI_API_KEY`, free tier is 20 requests/day per model):

```
python -m pipeline.coach            # drafts -> data/derived/coach_drafts.json; read them all
python -m pipeline.coach --apply    # write the reviewed drafts into the puzzles (no API calls)
```

Re-run both after `pipeline.puzzles`, which rewrites the puzzle files.

The truth and puzzle steps both write into `web/public/data/` (they also overwrite `data/*_check.png`; `git checkout -- data/` if you
don't mean to commit those). `metrics.json` comes from the evaluation step; until then copy
`contracts/fixtures/metrics.json` in.

Needs Node 20.19+ (Next 16). `npm run build` and `npm run lint` are what CI runs.

## Layout

```
src/app/            layout (fonts, theme), page, globals.css (all styles; colours are tokens with light/dark values)
src/components/     App (screens + shift state), SignIn (+ live scope preview), ShiftPicker,
                    Game (scope, strips, info display, console, game loop), KeyPanel, EndOfShift, Stats
src/lib/            data.ts (loads /data), puzzle.ts (positions, runway traffic, estimates), scope.ts (SVG renderer),
                    geom.ts, units.ts (kt/ft), storage.ts (rounds in localStorage)
```

Types and scoring come from the shared contract: `import type { Puzzle } from "@contracts/types"`,
`import { makeRound } from "@contracts/scoring"` (Turbopack's root is the repo root for this; see `next.config.ts`).

## Rules the UI follows

- Puzzle filenames come from `puzzles.json`; the UI never builds them.
- `reveal`, `outcome` and `ai.coach` are never shown before the player transmits. The WHAT IF tag only appears after the reveal.
- Runway traffic = an aircraft on the runway surface (half-width + 10 m) or on the extended centre line beyond the ends
  (within 100 m, moving faster than 40 kt), heading along the runway. Puzzles whose question aircraft fails this are skipped
  (`lib/data.ts`); see `isRunwayTraffic` in `lib/puzzle.ts`.
- Speeds are shown in knots and distances in feet, with metric alongside.
