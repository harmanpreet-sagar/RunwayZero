// RunwayZero scoring. Mirrored in pipeline/contracts.py (p_safe_from / brier / points).
// Use these everywhere so the UI, leaderboard and results slide agree.
//
//   p_safe  = confidence if CLEAR, 1 - confidence if HOLD
//   outcome = 1 if the crossing was safe, 0 if conflict
//   brier   = (p_safe - outcome)^2          0 = perfect, 1 = confidently wrong
//   points  = round(100 * (1 - brier))      what the player sees
//
// Examples: CLEAR @95% and safe -> 100 pts; CLEAR @95% and conflict -> 10 pts;
//           any call @50% -> 75 pts (hedging is safe but caps you).
// The AI plays by the same rules: it is scored on its SNAPPED confidence, not raw p_safe.
import type { Call, Confidence, Puzzle, Round, Detector } from "./types";

export function pSafeFrom(call: Call, confidence: Confidence): number {
  return call === "CLEAR" ? confidence : Math.round((1 - confidence) * 10000) / 10000;
}

export function brier(pSafe: number, conflict: boolean): number {
  const y = conflict ? 0 : 1;
  return (pSafe - y) ** 2;
}

export function points(b: number): number {
  return Math.round(100 * (1 - b));
}

export function score(call: Call, confidence: Confidence, conflict: boolean) {
  const p_safe = pSafeFrom(call, confidence);
  const b = brier(p_safe, conflict);
  return { call, confidence, p_safe, brier: Math.round(b * 10000) / 10000, points: points(b) };
}

/** Build the Round record to save after the player locks in and the reveal plays. */
export function makeRound(
  puzzle: Puzzle,
  human: { call: Call; confidence: Confidence; decision_ms: number },
  player: string,
  session_id: string,
): Round {
  const conflict = puzzle.outcome.conflict;
  return {
    schema_version: "1.1",
    puzzle_id: puzzle.id,
    difficulty: puzzle.difficulty,
    player,
    session_id,
    ts: new Date().toISOString(),
    conflict,
    human: { ...score(human.call, human.confidence, conflict), decision_ms: Math.round(human.decision_ms) },
    ai: { ...score(puzzle.ai.call, puzzle.ai.confidence, conflict), detector: puzzle.ai.detector as Detector },
  };
}
