// The surface-surveillance scope, drawn straight into an <svg> (imperative on purpose: it redraws
// every animation frame, which is cheaper than re-rendering React). Styling comes from CSS classes
// (.s-*) in globals.css, so light/dark themes switch without redrawing.

import type { AirportMap, Puzzle } from "@contracts/types";
import { type Vec, add, compass, headingVec, mul, runwayFrame } from "./geom";
import {
  type StopBar, type Track, type View, at, callsign, category, isRunwayTraffic, stopBar, visibleAt, zoneOf,
} from "./puzzle";
import { M_S_TO_KT, M_TO_FT, fmt } from "./units";

const NS = "http://www.w3.org/2000/svg";
type Attrs = Record<string, string | number>;
function el(tag: string, attrs: Attrs, parent?: Element): SVGElement {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  parent?.appendChild(e);
  return e;
}
const P = (p: number[]) => `${p[0]},${-p[1]}`;
const PLANE =
  "M0,-1 L.13,-.35 L1,.12 L1,.3 L.13,.12 L.1,.68 L.36,.88 L.36,1 L0,.92 L-.36,1 L-.36,.88 L-.1,.68 L-.13,.12 L-1,.3 L-1,.12 L-.13,-.35Z";
const NSS = { "vector-effect": "non-scaling-stroke" };

export type Phase = "attract" | "watch" | "decide" | "ai" | "reveal" | "result";

export interface Scene {
  pz: Puzzle;
  phase: Phase;
  t: number;
  call?: "CLEAR" | "HOLD";
  coneProgress?: number; // 0..1, how much of the AI forecast to draw (it "grows" after lock-in)
  selected?: string | null;
  alert?: boolean;
  /** Highlight the puzzle's aircraft as THE traffic (false when it isn't really runway traffic). */
  focusAircraft: boolean;
  /** Which ai.paths to draw: only real runway traffic + the vehicle. */
  coneIds: Set<string>;
}

export interface Hover extends Track { label: string; type: string; traffic: boolean }

export class ScopeRenderer {
  private svg: SVGSVGElement;
  private dyn: SVGGElement | null = null;
  private zoneEl: SVGElement | null = null;
  private sb: StopBar | null = null;
  private pz: Puzzle | null = null;
  last = new Map<string, Hover>();

  constructor(svg: SVGSVGElement, private map: AirportMap, private background?: string) {
    this.svg = svg;
  }

  get puzzle() { return this.pz; }

  /** Draw everything that doesn't move: background, taxiways, runways, crossing zone, hold line. */
  build(pz: Puzzle, view: View) {
    const { svg, map } = this;
    svg.innerHTML = "";
    this.pz = pz;
    const k = view[2] / (svg.clientWidth || 900);
    el("rect", { x: -40000, y: -40000, width: 80000, height: 80000, class: "s-bg" }, svg);
    if (map.background && this.background) {
      const b = map.background.bounds;
      el("image", { href: this.background, x: b.xmin, y: -b.ymax, width: b.xmax - b.xmin, height: b.ymax - b.ymin, preserveAspectRatio: "none", class: "s-bgimg" }, svg);
    }
    const S = el("g", {}, svg);
    for (const tw of map.taxiways)
      el("polyline", { points: tw.polyline.map(P).join(" "), class: "s-twy", "stroke-width": 23, "stroke-linejoin": "round", "stroke-linecap": "round" }, S);
    for (const r of map.runways) {
      el("polygon", { points: r.polygon.map(P).join(" "), class: "s-rwy", "stroke-width": 1, ...NSS }, S);
      const f = runwayFrame(r.polygon as Vec[]);
      const ends = r.id.split("/");
      if (ends.length !== 2) continue;
      const h = compass(f.u);
      const diff = Math.abs(((parseInt(ends[0]) * 10 - h + 540) % 360) - 180);
      const lo = diff < 90 ? ends[0] : ends[1];
      const hi = lo === ends[0] ? ends[1] : ends[0];
      for (const [e, txt, rot] of [[-1, lo, h], [1, hi, h + 180]] as const) {
        const p = add(f.c, mul(f.u, e * (f.L / 2 - 40)));
        const t = el("text", { x: p[0], y: -p[1], "text-anchor": "middle", "dominant-baseline": "middle", "font-size": k * 12, class: "s-txt", transform: `rotate(${rot} ${p[0]} ${-p[1]})` }, S);
        t.textContent = txt;
      }
    }
    // named taxiways only (the real map has hundreds of auto-named T0000 segments)
    for (const tw of map.taxiways.filter((x) => !/^T\d+$/.test(x.id))) {
      const pts = tw.polyline;
      const mid = pts[Math.floor(pts.length / 2)];
      const t = el("text", { x: mid[0] + k * 10, y: -mid[1] - k * 6, "font-size": k * 11, class: "s-txt" }, S);
      t.textContent = tw.id;
    }
    const z = zoneOf(map, pz);
    this.zoneEl = el("polygon", { points: z.polygon.map(P).join(" "), class: "s-zone", "stroke-width": 1.2, "stroke-dasharray": "5 4", ...NSS }, S);
    const sb = (this.sb = stopBar(map, pz));
    for (const o of [7, 10]) {
      const a = add(add(sb.c, mul(sb.n, o)), mul(sb.d, -14));
      const b = add(add(sb.c, mul(sb.n, o)), mul(sb.d, 14));
      el("line", { x1: a[0], y1: -a[1], x2: b[0], y2: -b[1], class: "s-hold", "stroke-width": 1.2, ...NSS }, S);
    }
    this.dyn = el("g", {}, svg) as SVGGElement;
  }

  /** Draw the moving layer for one moment. */
  draw(sc: Scene, cam: View, now: number) {
    const { svg, map, sb, dyn: D } = this;
    if (!D || !sb || !this.zoneEl) return;
    svg.setAttribute("viewBox", cam.join(" "));
    const k = cam[2] / (svg.clientWidth || 900);
    D.innerHTML = "";
    this.last.clear();
    const { pz, phase: ph, t } = sc;
    const revealed = ph === "reveal" || ph === "result";
    const subjects = sc.focusAircraft ? [pz.vehicle_id, pz.aircraft_id] : [pz.vehicle_id];
    const live = ph !== "attract";
    this.zoneEl.setAttribute("class", "s-zone" + (ph === "decide" || sc.alert ? " hot" : ""));

    // stop bar (red) until the player clears the crossing, then the lead-on path (green)
    if (live && sc.call === "CLEAR" && (ph === "ai" || revealed)) {
      const b = add(sb.c, mul(sb.n, -(sb.depth + 6)));
      el("line", { x1: sb.c[0], y1: -sb.c[1], x2: b[0], y2: -b[1], class: "s-lead", "stroke-width": 2, "stroke-dasharray": "4 3", ...NSS }, D);
    } else {
      const a = add(add(sb.c, mul(sb.n, 2)), mul(sb.d, -14));
      const b = add(add(sb.c, mul(sb.n, 2)), mul(sb.d, 14));
      el("line", { x1: a[0], y1: -a[1], x2: b[0], y2: -b[1], class: "s-stop", "stroke-width": 3, ...NSS }, D);
    }

    // AI forecast: only for the vehicle and real runway traffic
    if (ph === "ai" || revealed) {
      const end = pz.ai.paths[pz.aircraft_id]?.slice(-1)[0]?.t ?? pz.decision_t + 30;
      const tMax = pz.decision_t + (sc.coneProgress ?? 1) * (end - pz.decision_t);
      for (const [id, path] of Object.entries(pz.ai.paths)) {
        if (!sc.coneIds.has(id)) continue;
        const isAc = id === pz.aircraft_id;
        const pts = path.filter((p) => p.t <= tMax);
        if (!pts.length) continue;
        pts.forEach((p, i) => {
          if (i % 2) return;
          el("circle", { cx: p.x, cy: -p.y, r: Math.max(p.r, k * 2), class: "s-aiS", "stroke-width": 1, "stroke-opacity": revealed ? 0.25 : isAc ? 0.65 : 0.4, ...NSS }, D);
        });
        el("polyline", { points: pts.map((p) => `${p.x},${-p.y}`).join(" "), class: "s-aiS", "stroke-width": 1.4, "stroke-dasharray": "6 4", "stroke-opacity": revealed ? 0.35 : 0.9, ...NSS }, D);
        if (isAc && !revealed)
          for (const p of pts) {
            const dt = Math.round(p.t - pz.decision_t);
            if (dt > 0 && dt % 10 === 0) {
              const tt = el("text", { x: p.x, y: -p.y - p.r - k * 5, "text-anchor": "middle", "font-size": k * 10, class: "s-aiT" }, D);
              tt.textContent = `+${dt}s`;
            }
          }
      }
    }

    // closest approach, once the recording reaches it
    if (revealed && t >= pz.outcome.min_sep_t && (sc.focusAircraft || pz.outcome.conflict)) {
      const a = at(pz.reveal as never, pz.outcome.min_sep_t, pz.vehicle_id);
      const b = at(pz.reveal as never, pz.outcome.min_sep_t, pz.aircraft_id);
      if (a && b) {
        const bad = pz.outcome.conflict;
        el("line", { x1: a.x, y1: -a.y, x2: b.x, y2: -b.y, class: bad ? "s-bad" : "s-good", "stroke-width": 1.6, ...NSS }, D);
        const tt = el("text", { x: (a.x + b.x) / 2 + k * 8, y: -(a.y + b.y) / 2 - k * 8, "font-size": k * 12.5, class: bad ? "s-badT" : "s-goodT", "font-weight": 600 }, D);
        tt.textContent = `CLOSEST ${fmt(pz.outcome.min_sep_m * M_TO_FT)} ft`;
      }
    }

    // targets: others first, the two subjects last so they sit on top
    const order = [...pz.objects.filter((o) => !subjects.includes(o.id)), ...pz.objects.filter((o) => subjects.includes(o.id))];
    for (const meta of order) {
      const s = visibleAt(pz, t, revealed, meta.id);
      if (!s) continue;
      const role = !live ? null : meta.id === pz.vehicle_id ? "veh" : sc.focusAircraft && meta.id === pz.aircraft_id ? "ac" : null;
      const sel = sc.selected === meta.id;
      const st = s.status;
      const traffic = meta.type === "aircraft" && isRunwayTraffic(map, pz, s);
      this.last.set(meta.id, { ...s, label: meta.label, type: meta.type, traffic });
      const dim = live && !role && !sel ? (ph === "decide" || ph === "ai" ? 0.42 : 0.72) : 1;
      const g = el("g", { "data-id": meta.id, opacity: dim, class: "s-target" }, D);

      // history dots: where it was over the last 5 s
      const seq = (revealed && t > pz.decision_t ? pz.reveal : pz.pre) as never;
      const tNow = revealed && t > pz.decision_t ? t : Math.min(t, pz.decision_t);
      for (let h = 1; h <= 5; h++) {
        const p = at(seq, tNow - h, meta.id);
        if (!p || p.status === "lost") continue;
        el("circle", { cx: p.x, cy: -p.y, r: k * 1.8, class: "s-hist", "fill-opacity": (0.55 - h * 0.08).toFixed(2) }, g);
      }
      // predicted track line: 10 s at present speed, for runway traffic and the subjects
      if (meta.type === "aircraft" && (role || traffic) && (s.speed ?? 0) > 1) {
        const e = add([s.x, s.y], mul(headingVec(s.heading ?? 0), (s.speed ?? 0) * 10));
        el("line", { x1: s.x, y1: -s.y, x2: e[0], y2: -e[1], class: "s-ptl", "stroke-width": 1, "stroke-opacity": 0.75, ...NSS }, g);
      }
      // role ring: yellow = the vehicle you are clearing, red = the traffic it must avoid
      if (role) {
        const r = k * (ph === "decide" ? 24 + 3 * Math.sin(now / 200) : 22);
        el("circle", { cx: s.x, cy: -s.y, r, class: role === "veh" ? "s-ringV" : "s-ringA", fill: "none", "stroke-width": 2, ...NSS }, g);
      }
      // the target itself
      const h = s.heading ?? 0;
      if (meta.type === "aircraft") {
        const sz = k * 17;
        el("path", {
          d: PLANE, transform: `translate(${s.x},${-s.y}) rotate(${h}) scale(${sz})`,
          class: st === "lost" ? "s-lost" : `${sel ? "s-sel" : "s-tgt"} s-halo`, "fill-opacity": st === "stale" ? 0.45 : 1,
          ...(st === "lost" ? { "stroke-width": 1.2, "stroke-dasharray": "3 2", ...NSS } : {}),
        }, g);
      } else {
        const sz = k * (meta.type === "vehicle" ? 11 : 8);
        el("rect", {
          x: -sz / 2, y: -sz / 2, width: sz, height: sz, transform: `translate(${s.x},${-s.y}) rotate(45)`,
          class: st === "lost" ? "s-vlost" : `${sel ? "s-sel" : "s-veh"} s-halo`, "fill-opacity": st === "stale" ? 0.45 : 1,
          ...(st === "lost" ? { "stroke-width": 1.2, ...NSS } : {}),
        }, g);
      }
      if (st === "lost") el("circle", { cx: s.x, cy: -s.y, r: k * 16, class: "s-lostRing", "stroke-dasharray": "4 4", "stroke-width": 1.2, ...NSS }, g);
      el("rect", { x: s.x - k * 16, y: -s.y - k * 16, width: k * 32, height: k * 32, fill: "transparent" }, g);

      // data tag: subjects, selected and moving aircraft only (the real hour has ~27 objects)
      if (!(role || sel || (meta.type === "aircraft" && (s.speed ?? 0) > 2))) continue;
      const fs = k * 12.5;
      const below = role === "veh"; // the vehicle's tag goes down-left so it never sits on the traffic's tag
      const lx = below ? s.x - k * 24 : s.x + k * 22;
      const ly = below ? -s.y + k * 34 : -s.y - k * 26;
      const anchor = below ? "end" : "start";
      el("line", { x1: s.x + (below ? -k * 9 : k * 9), y1: -s.y + (below ? k * 9 : -k * 9), x2: lx + (below ? k * 2 : -k * 2), y2: ly - (below ? fs * 1.2 : -fs * 0.4), class: role || sel ? "s-selS" : "s-tagS", "stroke-width": 1, "stroke-opacity": 0.75, ...NSS }, g);
      const cls = role === "veh" ? "s-roleV" : role === "ac" ? "s-roleA" : sel ? "s-sel" : "s-tag";
      if (role) {
        const chip = el("text", { x: lx, y: ly - fs * 1.35, "font-size": fs * 0.78, class: cls, "font-weight": 700, "letter-spacing": "0.08em", "text-anchor": anchor }, g);
        chip.textContent = role === "veh" ? "▸ CROSSING VEHICLE" : "▸ RUNWAY TRAFFIC";
      }
      const t1 = el("text", { x: lx, y: ly - fs * 0.15, "font-size": fs, class: cls, "font-weight": 600, "text-anchor": anchor }, g);
      t1.textContent = callsign(meta.label);
      const speed = s.speed != null ? `${Math.round(s.speed * M_S_TO_KT)} kt` : "– kt";
      const extra = st === "lost" ? ` LOST ${fmt(s.age_s)}s` : st === "stale" ? ` OLD ${fmt(s.age_s)}s` : "";
      const t2 = el("text", { x: lx, y: ly + fs, "font-size": fs, class: st === "lost" ? "s-badT" : cls, "text-anchor": anchor }, g);
      t2.textContent = `${category(meta.type, meta.label, s)} ${speed}${extra}`;
    }

    // fog: after the reveal starts, keep a marker where the player last saw the traffic
    if (revealed) {
      const ls = pz.pre[pz.pre.length - 1].objects.find((o) => o.id === pz.aircraft_id);
      if (ls && ls.status && ls.status !== "ok") {
        el("circle", { cx: ls.x, cy: -ls.y, r: k * 10, class: "s-lostRing", "stroke-width": 1, "stroke-dasharray": "2 3", ...NSS }, D);
        const tt = el("text", { x: ls.x, y: -ls.y + k * 22, "text-anchor": "middle", "font-size": k * 10, class: "s-txt" }, D);
        tt.textContent = "LAST RADAR";
      }
    }
  }
}

/** Which forecast paths to draw: the vehicle plus aircraft that are runway traffic at the freeze. */
export function coneIdsFor(map: AirportMap, pz: Puzzle): Set<string> {
  const ids = new Set<string>([pz.vehicle_id]);
  for (const id of Object.keys(pz.ai.paths)) {
    const s = at(pz.pre as never, pz.decision_t, id);
    if (s && isRunwayTraffic(map, pz, s)) ids.add(id);
  }
  return ids;
}

