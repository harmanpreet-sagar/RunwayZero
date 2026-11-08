// RunwayZero data contract, TypeScript mirror of contracts/schema/*.schema.json (v1.1).
// The JSON schemas are the source of truth. If you change one, change this file,
// the Python side (pipeline/contracts.py) and regenerate fixtures in the SAME commit.
// Units: metres, local Cartesian frame (x = east, y = north, origin = airport reference point).
// Time: seconds from replay/puzzle start. Headings: degrees clockwise from north.
//
// v1.1 (game pivot): adds puzzles.json, puzzle_<id>.json, Round, metrics.game.
// Replays + scenarios.json are kept as a secondary "watch mode"; no new work planned on them.

export type SchemaVersion = "1.1";
export type Point = [number, number];
export interface Bounds { xmin: number; xmax: number; ymin: number; ymax: number }

// ---------- map.json ----------
export interface AirportMap {
  schema_version: SchemaVersion;
  is_mock?: boolean;
  airport: string;
  units: "m";
  origin?: { lat: number; lon: number };
  bounds: Bounds;
  background?: { image: string; bounds: Bounds }; // image path relative to /data/
  runways: { id: string; polygon: Point[] }[];
  taxiways: { id: string; polyline: Point[] }[];
  zones: { id: string; name: string; polygon: Point[] }[]; // protected crossing zones
}

// ---------- scenarios.json ----------
export type ScenarioId = "normal" | "cleared_crossing" | "tracking_dropout" | "delayed_data";
export interface ScenarioIndex {
  schema_version: SchemaVersion;
  is_mock?: boolean;
  scenarios: { id: ScenarioId; name: string; description: string }[];
  // The UI builds pickers from this list and NEVER constructs filenames itself.
  replays: { scenario_id: ScenarioId; seed: number; dropout: number; file: string }[];
}

// ---------- replay_*.json ----------
export type Detector = "cv" | "kf" | "ml"; // constant velocity, Kalman filter, learned model
export type ObjectType = "aircraft" | "vehicle" | "unknown";
export type ObsStatus = "ok" | "stale" | "lost";

export interface TruePos { x: number; y: number; heading?: number; speed?: number }
export interface ObsPos {
  x: number; y: number; // when status === "lost", this is the LAST KNOWN position
  age_s: number;        // seconds since the last real sensor update
  status: ObsStatus;
}
export interface Pred { x: number; y: number; r: number } // position at t + pred_horizon_s, 90% radius

export interface FrameObject {
  id: string;
  true: TruePos;
  obs: ObsPos;
  pred: Partial<Record<Detector, Pred>>;
}
export interface Alert {
  detector: Detector;
  zone_id: string;
  p: number;        // 0..1 conflict probability (cv is always 1)
  lead_s?: number;  // seconds before the true conflict; absent = false alert
  objects: string[];
}
export interface TruthConflict { zone_id: string; objects: string[] }
export interface Frame {
  t: number;
  objects: FrameObject[];
  alerts: Alert[];
  truth_conflicts: TruthConflict[];
}
export type EventKind =
  | "truth_conflict_start" | "truth_conflict_end" | "first_alert"
  | "dropout_start" | "dropout_end" | "clearance";
export interface ReplayEvent {
  t: number; kind: EventKind; note: string;
  detector?: Detector; zone_id?: string; objects?: string[];
}
export interface Replay {
  schema_version: SchemaVersion;
  is_mock?: boolean;
  scenario_id: ScenarioId;
  seed: number;
  dropout: number;
  dt_s: number;
  pred_horizon_s: number;
  detectors: Detector[];
  objects: { id: string; type: ObjectType; label: string }[];
  frames: Frame[];
  events: ReplayEvent[];
}

// ---------- metrics.json ----------
export type AgeBucket = "0-1s" | "1-3s" | "3s+";
export interface Metrics {
  schema_version: SchemaVersion;
  is_mock?: boolean;
  notes?: string;
  detectors: Detector[];
  error_by_age: { detector: Detector; age_bucket: AgeBucket; horizon_s: number; mean_err_m: number }[];
  coverage: { detector: Detector; age_bucket: AgeBucket; target: number; empirical: number }[];
  // Each detector playing every puzzle as the AI player. Lower Brier is better.
  game?: { detector: Detector; difficulty: Difficulty; n: number; accuracy: number; brier: number; brier_raw: number }[];
  // Legacy replay-scenario metrics: OPTIONAL since v1.1. Hide the chart if missing.
  summary?: { detector: Detector; recall: number; precision: number; false_alerts_per_hr: number; median_lead_s: number; pr_auc: number }[];
  pr_curve?: { detector: Detector; threshold: number; precision: number; recall: number }[];
  robustness?: { detector: Detector; dropout: number; recall: number }[];
  per_scenario?: { scenario_id: ScenarioId; detector: Detector; recall: number; false_alerts_per_hr: number }[];
}

// ---------- game: shared ----------
export type Difficulty = "clear" | "busy" | "fog"; // Clear day, Busy hour, Fog mode
export type Call = "CLEAR" | "HOLD";
export type Confidence = 0.5 | 0.75 | 0.95;
export const CONFIDENCES: Confidence[] = [0.5, 0.75, 0.95];

// ---------- puzzles.json ----------
// Index only: NO outcomes in here, so the menu can't spoil anything.
export interface PuzzleIndex {
  schema_version: SchemaVersion;
  is_mock?: boolean;
  conflict_threshold_m: number;
  difficulties: { id: Difficulty; name: string; description: string }[];
  puzzles: { id: string; difficulty: Difficulty; title: string; file: string }[]; // UI never builds filenames
}

// ---------- puzzle_<id>.json ----------
export interface PuzzleObs {
  id: string;
  x: number; y: number;      // when status === "lost", this is the LAST KNOWN position
  heading?: number; speed?: number;
  age_s: number;             // 0 outside fog mode
  status: ObsStatus;
}
export interface PuzzleTrue { id: string; x: number; y: number; heading?: number; speed?: number }
export interface PathPoint { t: number; x: number; y: number; r: number } // r = 90% radius, metres
export interface AICall { call: Call; p_safe: number; confidence: Confidence }

export interface Puzzle {
  schema_version: SchemaVersion;
  is_mock?: boolean;
  id: string;
  difficulty: Difficulty;
  source: {
    airport: string;
    recording: string;
    t0_s: number;
    // What-if puzzles only: the vehicle's own real crossing, replayed this many seconds
    // away from when it really happened (negative = earlier). Aircraft are exactly as
    // recorded. Absent/0 = fully real. Show a "WHAT IF" tag only AFTER the reveal
    // (before it, the tag would give away the answer: real crossings were all safe).
    vehicle_time_shift_s?: number;
  };
  dt_s: number;
  decision_t: number;          // freeze here. pre: t <= decision_t, reveal: t > decision_t
  zone_id: string;             // crossing zone in map.json
  vehicle_id: string;
  aircraft_id: string;
  question: string;
  objects: { id: string; type: ObjectType; label: string }[];
  pre: { t: number; objects: PuzzleObs[] }[];   // what the player sees (observed in fog mode)
  ai: {
    detector: Detector;
    call: Call;
    p_safe: number;            // raw model probability that crossing now is safe
    confidence: Confidence;    // snapped; used for the head-to-head score
    paths: Record<string, PathPoint[]>; // object id -> forecast after decision_t; includes aircraft_id
    reasoning: string;
    coach?: string;            // HIDDEN until reveal (it explains what really happened)
    baselines?: Partial<Record<Detector, AICall>>;
  };
  // ---- HIDDEN until the player locks in ----
  reveal: { t: number; objects: PuzzleTrue[] }[]; // TRUE positions from the recording
  outcome: {
    conflict: boolean;         // min_sep_m < threshold_m while the vehicle was in the zone
    threshold_m: number;
    min_sep_m: number;
    min_sep_t: number;
    correct_call: Call;        // HOLD if conflict, else CLEAR
    note: string;
  };
}

// ---------- Round (UI writes one per round: localStorage now, MongoDB later) ----------
export interface PlayerScore { call: Call; confidence: Confidence; p_safe: number; brier: number; points: number }
export interface Round {
  schema_version: SchemaVersion;
  is_mock?: boolean;
  puzzle_id: string;
  difficulty: Difficulty;
  player: string;
  session_id: string;
  ts: string;                  // ISO date-time
  conflict: boolean;
  human: PlayerScore & { decision_ms: number };
  ai: PlayerScore & { detector: Detector };
}
