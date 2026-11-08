// Everything the UI derives from a puzzle: interpolated positions, names, runway traffic,
// the controller's present-speed estimates. Pure functions, no DOM.
//
// HIDDEN FUTURE RULE (contracts/README.md): `reveal`, `outcome` and `ai.coach` must not be
// rendered or logged before the player locks in. Functions that read them say so.

import type { AirportMap, ObjectType, Puzzle } from "@contracts/types";
import {
  type RunwayFrame, type Vec, add, angleDiff, centroid, compass, dot, headingVec, len, lerp, mul, runwayFrame, sub, unit,
} from "./geom";
import { M_S_TO_KT } from "./units";

/** One object at one moment. Pre frames carry age/status (fog); reveal frames are always true positions. */
export interface Track {
  id: string;
  x: number;
  y: number;
  heading?: number;
  speed?: number;
  age_s: number;
  status: "ok" | "stale" | "lost";
}

type Frame = { t: number; objects: { id: string; x: number; y: number; heading?: number; speed?: number; age_s?: number; status?: Track["status"] }[] };

/** Position of `id` at time t, linearly interpolated between 1 s frames. A lost track stays frozen at its last known position. */
export function at(seq: Frame[], t: number, id: string): Track | null {
  let i = 0;
  while (i < seq.length - 1 && seq[i + 1].t <= t) i++;
  const f = seq[i];
  const g = seq[Math.min(i + 1, seq.length - 1)];
  const a = f.objects.find((o) => o.id === id);
  if (!a) return null;
  const b = g.objects.find((o) => o.id === id) ?? a;
  const u = g.t === f.t ? 0 : Math.max(0, Math.min(1, (t - f.t) / (g.t - f.t)));
  const frozen = a.status === "lost";
  return {
    id,
    x: frozen ? a.x : lerp(a.x, b.x, u),
    y: frozen ? a.y : lerp(a.y, b.y, u),
    heading: a.heading,
    speed: a.speed != null && b.speed != null ? lerp(a.speed, b.speed, u) : a.speed,
    age_s: a.age_s != null ? lerp(a.age_s, b.age_s ?? a.age_s, u) : 0,
    status: a.status ?? "ok",
  };
}

/** What the player may see at time t: pre frames up to the freeze, reveal frames only once revealed. */
export function visibleAt(pz: Puzzle, t: number, revealed: boolean, id: string): Track | null {
  if (revealed && t > pz.decision_t) return at(pz.reveal as Frame[], t, id);
  return at(pz.pre as Frame[], Math.min(t, pz.decision_t), id);
}

// ---------- names ----------
export const zoneOf = (map: AirportMap, pz: Puzzle) => map.zones.find((z) => z.id === pz.zone_id) ?? map.zones[0];
export const runwayIdOf = (map: AirportMap, pz: Puzzle) => (zoneOf(map, pz).name.match(/RWY\s*([\w/]+)/i) ?? [])[1] ?? "";
export const taxiwayOf = (map: AirportMap, pz: Puzzle) => (zoneOf(map, pz).name.match(/TWY\s*(\w+)/i) ?? [])[1] ?? "";
export const shortLabel = (label: string) => label.replace(/\s*\(.*?\)\s*/g, "").trim();
export const labelOf = (pz: Puzzle, id: string) => pz.objects.find((o) => o.id === id)?.label ?? id;
export const typeOf = (pz: Puzzle, id: string): ObjectType => pz.objects.find((o) => o.id === id)?.type ?? "unknown";

/** Short scope ID. Amelia has no callsigns, so "Aircraft 5849" -> AC5849, "Vehicle 3475" -> VH3475. */
export function callsign(label: string): string {
  const s = shortLabel(label);
  let m = s.match(/^aircraft\s*(\w+)$/i);
  if (m) return `AC${m[1]}`;
  m = s.match(/^vehicle\s*(\w+)$/i);
  if (m) return `VH${m[1]}`;
  m = s.match(/^unknown\s*(\w+)$/i);
  if (m) return `U${m[1]}`;
  return s.toUpperCase().replace(/\s+/g, "");
}

const NATO: Record<string, string> = {
  A: "Alpha", B: "Bravo", C: "Charlie", D: "Delta", E: "Echo", F: "Foxtrot", G: "Golf", H: "Hotel", J: "Juliett", K: "Kilo",
  L: "Lima", M: "Mike", N: "November", P: "Papa", Q: "Quebec", R: "Romeo", S: "Sierra", T: "Tango", U: "Uniform", V: "Victor",
  W: "Whiskey", X: "X-ray", Y: "Yankee", Z: "Zulu",
};

/** The radio instruction, in FAA phrasing. */
export function phrase(map: AirportMap, pz: Puzzle, call: "CLEAR" | "HOLD"): string {
  const vehicle = shortLabel(labelOf(pz, pz.vehicle_id));
  const tw = taxiwayOf(map, pz);
  const twSpoken = tw ? tw.split("").map((c) => NATO[c.toUpperCase()] ?? c).join(" ") : "";
  const rw = runwayIdOf(map, pz);
  return call === "CLEAR" ? `${vehicle}, cross Runway ${rw}${twSpoken ? ` at ${twSpoken}` : ""}.` : `${vehicle}, hold short of Runway ${rw}.`;
}

/** Second line of a data tag: what the target is doing. */
export function category(type: ObjectType, label: string, s: Track | null): string {
  if (type === "vehicle") return "VEH";
  if (type === "unknown") return "UNK";
  const l = label.toLowerCase();
  if (/land|arriv/.test(l)) return "ARR";
  if (/depart|takeoff/.test(l)) return "DEP";
  if (/taxi/.test(l)) return "TAXI";
  return s && (s.speed ?? 0) * M_S_TO_KT > 40 ? "RWY" : "TAXI";
}

/** Recording clock in UTC when the recording name carries a Unix time (KBOS_538_1674511200), else T+mm:ss. */
export function clockAt(pz: Puzzle, t: number): string {
  const m = String(pz.source?.recording ?? "").match(/(\d{10})/);
  if (m) return new Date((Number(m[1]) + (pz.source.t0_s || 0) + t) * 1000).toISOString().slice(11, 19) + "Z";
  const s = Math.max(0, Math.round(t));
  return `T+${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

// ---------- runway traffic ----------
const frames = new Map<string, RunwayFrame>();
export function runwayOf(map: AirportMap, pz: Puzzle) {
  const id = runwayIdOf(map, pz);
  const r = map.runways.find((x) => x.id === id);
  if (!r) return null;
  if (!frames.has(id)) frames.set(id, runwayFrame(r.polygon as Vec[]));
  return { ...r, f: frames.get(id)! };
}

/**
 * Runway traffic = an aircraft that can actually reach the crossing along the runway.
 *  - heading within 25 deg of the runway, either direction (as pipeline/ai_player.is_runway_user);
 *  - between the runway ends: ON the runway surface (half-width + 10 m), not on a parallel taxiway;
 *  - beyond the ends: on the extended centre line (within 100 m, up to 4 km out) and faster than 40 kt.
 * The pipeline's 100 m band alone also catches parallel taxiways (see r15r33l_v3662_t2035).
 */
export function isRunwayTraffic(map: AirportMap, pz: Puzzle, s: Track | null): boolean {
  const r = runwayOf(map, pz);
  if (!r || !s) return false;
  const { f } = r;
  const rel = sub([s.x, s.y], f.c);
  const along = dot(rel, f.u);
  const lateral = Math.abs(dot(rel, f.v));
  const d = angleDiff(s.heading ?? 0, compass(f.u));
  if (!(d <= 25 || d >= 155)) return false;
  if (Math.abs(along) <= f.L / 2) return lateral <= f.W / 2 + 10;
  return Math.abs(along) <= f.L / 2 + 4000 && lateral <= 100 && (s.speed ?? 0) * M_S_TO_KT > 40;
}

/** Is the puzzle's named aircraft runway traffic in what the player can see? (pre frames only, so no spoilers) */
export function namedAircraftIsRunwayTraffic(map: AirportMap, pz: Puzzle): boolean {
  return pz.pre.slice(-10).some((f) => {
    const o = f.objects.find((x) => x.id === pz.aircraft_id);
    return !!o && isRunwayTraffic(map, pz, { ...o, age_s: o.age_s ?? 0, status: o.status ?? "ok" });
  });
}

export interface Eta { along: number; miss: number; eta: number | null; past: boolean; away: boolean }

/** Seconds until an aircraft reaches the crossing at its present speed and heading. */
export function etaToCrossing(map: AirportMap, pz: Puzzle, s: Track): Eta {
  const zc = centroid(zoneOf(map, pz).polygon as Vec[]);
  const u = headingVec(s.heading ?? 0);
  const rel = sub(zc, [s.x, s.y]);
  const along = dot(rel, u);
  const miss = Math.abs(rel[0] * u[1] - rel[1] * u[0]);
  const toward = along > 0 && miss < 150;
  return { along, miss, eta: toward && (s.speed ?? 0) > 1 ? along / (s.speed ?? 1) : null, past: along <= 0, away: along > 0 && miss >= 150 };
}

export interface TrafficItem { id: string; label: string; s: Track; eta: Eta }

export function runwayTraffic(map: AirportMap, pz: Puzzle, t: number): TrafficItem[] {
  return pz.objects
    .filter((o) => o.type === "aircraft")
    .map((o) => ({ id: o.id, label: o.label, s: at(pz.pre as Frame[], Math.min(t, pz.decision_t), o.id) }))
    .filter((x): x is { id: string; label: string; s: Track } => !!x.s && isRunwayTraffic(map, pz, x.s))
    .map((x) => ({ ...x, eta: etaToCrossing(map, pz, x.s) }))
    .sort((a, b) => (a.eta.eta ?? 1e9) - (b.eta.eta ?? 1e9));
}

// ---------- hold line ----------
export interface StopBar { c: Vec; d: Vec; n: Vec; depth: number }

/** The crossing-zone edge nearest the vehicle at the freeze: where the stop bar sits. n points back toward the vehicle. */
export function stopBar(map: AirportMap, pz: Puzzle): StopBar {
  const z = zoneOf(map, pz).polygon as Vec[];
  const last = pz.pre[pz.pre.length - 1].objects.find((o) => o.id === pz.vehicle_id) ?? pz.pre[0].objects[0];
  const vp: Vec = [last.x, last.y];
  let best = { dist: Infinity, q: z[0], u: [1, 0] as Vec };
  for (let i = 0; i < z.length; i++) {
    const a = z[i];
    const b = z[(i + 1) % z.length];
    const L = len(sub(b, a));
    const u = unit(sub(b, a));
    const s = Math.max(0, Math.min(L, dot(sub(vp, a), u)));
    const q = add(a, mul(u, s));
    const dist = len(sub(vp, q));
    if (dist < best.dist) best = { dist, q, u };
  }
  let n: Vec = [-best.u[1], best.u[0]];
  if (dot(sub(vp, best.q), n) < 0) n = mul(n, -1);
  return { c: best.q, d: best.u, n, depth: Math.max(...z.map((p) => dot(sub(p, best.q), mul(n, -1)))) };
}

// ---------- the controller's estimates (constant speed, deliberately naive) ----------
export interface Estimates {
  traffic: TrafficItem[];
  degraded: Track[];
  vehicleSpeed: number;
  vehicleSpeedAssumed: boolean;
  crossDistance: number;
  vehicleNeeds: number;
  first?: TrafficItem;
  margin?: number;
}

export function estimates(map: AirportMap, pz: Puzzle, t: number): Estimates {
  const tt = Math.min(t, pz.decision_t);
  const traffic = runwayTraffic(map, pz, tt);
  const degraded = pz.objects
    .filter((o) => o.type === "aircraft")
    .map((o) => at(pz.pre as Frame[], tt, o.id))
    .filter((o): o is Track => !!o && o.status !== "ok")
    .sort((a, b) => b.age_s - a.age_s);
  const vmax = Math.max(0, ...pz.pre.map((f) => f.objects.find((o) => o.id === pz.vehicle_id)?.speed ?? 0));
  const vehicleSpeed = vmax > 1 ? vmax : 8;
  const crossDistance = stopBar(map, pz).depth + 10;
  const vehicleNeeds = crossDistance / vehicleSpeed;
  const first = traffic.find((x) => x.eta.eta != null);
  return {
    traffic, degraded, vehicleSpeed, vehicleSpeedAssumed: !(vmax > 1), crossDistance, vehicleNeeds, first,
    margin: first ? (first.eta.eta as number) - vehicleNeeds : undefined,
  };
}

// ---------- camera ----------
export type View = [number, number, number, number]; // SVG viewBox: x, y (= -north), width, height

export function defaultViews(map: AirportMap, pz: Puzzle, ratio: number): { wide: View; tape: View } {
  const [cx, cy] = centroid(zoneOf(map, pz).polygon as Vec[]);
  let d = 300;
  for (const f of pz.pre.slice(-12))
    for (const o of f.objects)
      if (o.id === pz.vehicle_id || o.id === pz.aircraft_id) d = Math.max(d, Math.abs(o.x - cx) * 1.1, Math.abs(o.y - cy) * 1.1 * ratio);
  const W = Math.min(4400, Math.max(1000, d * 2.2));
  const tw = Math.max(900, W * 0.8);
  return { wide: [cx - W / 2, -cy - W / ratio / 2, W, W / ratio], tape: [cx - tw / 2, -cy - tw / ratio / 2, tw, tw / ratio] };
}

/** Separation between the vehicle and the named aircraft at time t. Reads `reveal`: only call after lock-in. */
export function separationAfterReveal(pz: Puzzle, t: number): number | null {
  const a = at(pz.reveal as Frame[], t, pz.vehicle_id);
  const b = at(pz.reveal as Frame[], t, pz.aircraft_id);
  return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : null;
}
