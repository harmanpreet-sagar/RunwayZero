// Loads the contract files from /data (web/public/data). The UI never builds filenames:
// puzzle files come from puzzles.json, the background image from map.json.

import type { AirportMap, Metrics, Puzzle, PuzzleIndex } from "@contracts/types";
import { namedAircraftIsRunwayTraffic } from "./puzzle";

export interface GameData {
  map: AirportMap;
  index: PuzzleIndex;
  metrics: Metrics | null;
  background?: string;
  /** Puzzles hidden because their question aircraft isn't really runway traffic (pipeline fix pending). */
  dropped: string[];
  isMock: boolean;
}

async function getJson<T>(file: string): Promise<T> {
  const res = await fetch(`/data/${file}`);
  if (!res.ok) throw new Error(`Could not load /data/${file} (${res.status}). Run the pipeline or \`npm run sync-fixtures\`.`);
  return res.json() as Promise<T>;
}

const puzzleCache = new Map<string, Promise<Puzzle>>();
export function loadPuzzle(file: string): Promise<Puzzle> {
  if (!puzzleCache.has(file)) puzzleCache.set(file, getJson<Puzzle>(file));
  return puzzleCache.get(file)!;
}

export async function loadGameData(): Promise<GameData> {
  const [map, index, metrics] = await Promise.all([
    getJson<AirportMap>("map.json"),
    getJson<PuzzleIndex>("puzzles.json"),
    getJson<Metrics>("metrics.json").catch(() => null),
  ]);
  // Load every puzzle once to screen out bad ones; ~100 KB each, so a couple of MB in total.
  const puzzles = await Promise.all(index.puzzles.map((p) => loadPuzzle(p.file)));
  const dropped = index.puzzles.filter((_, i) => !namedAircraftIsRunwayTraffic(map, puzzles[i])).map((p) => p.id);
  return {
    map,
    index: { ...index, puzzles: index.puzzles.filter((p) => !dropped.includes(p.id)) },
    metrics,
    background: map.background ? `/data/${map.background.image}` : undefined,
    dropped,
    isMock: !!(map.is_mock || index.is_mock),
  };
}
