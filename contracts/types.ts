// RunwayZero data contract, TypeScript mirror of contracts/schema/*.schema.json (v1.0).
// The JSON schemas are the source of truth. If you change one, change this file,
// the Python side (pipeline/contracts.py) and regenerate fixtures in the SAME commit.
// Units: metres, local Cartesian frame (x = east, y = north, origin = airport reference point).
// Time: seconds from replay start. Headings: degrees clockwise from north.

export type SchemaVersion = "1.0";
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
  summary: { detector: Detector; recall: number; precision: number; false_alerts_per_hr: number; median_lead_s: number; pr_auc: number }[];
  error_by_age: { detector: Detector; age_bucket: AgeBucket; horizon_s: number; mean_err_m: number }[];
  coverage: { detector: Detector; age_bucket: AgeBucket; target: number; empirical: number }[];
  pr_curve: { detector: Detector; threshold: number; precision: number; recall: number }[];
  robustness: { detector: Detector; dropout: number; recall: number }[];
  per_scenario: { scenario_id: ScenarioId; detector: Detector; recall: number; false_alerts_per_hr: number }[];
}
