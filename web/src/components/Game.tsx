"use client";

import { makeRound, score } from "@contracts/scoring";
import type { Call, Confidence, Puzzle, Round } from "@contracts/types";
import { useCallback, useEffect, useRef, useState } from "react";
import { type GameData, loadPuzzle } from "@/lib/data";
import { inPoly, type Vec } from "@/lib/geom";
import {
  type View, at, callsign, category, clockAt, defaultViews, estimates, etaToCrossing, labelOf, namedAircraftIsRunwayTraffic,
  phrase, runwayIdOf, runwayTraffic, separationAfterReveal, shortLabel, taxiwayOf, zoneOf,
} from "@/lib/puzzle";
import { type Hover, type Phase, ScopeRenderer, coneIdsFor } from "@/lib/scope";
import { saveRound } from "@/lib/storage";
import { KT_TO_MPH, M_S_TO_KT, M_TO_FT, distanceText, fmt, kt, speedsToKnots } from "@/lib/units";
import type { Shift } from "./App";
import KeyPanel from "./KeyPanel";

type Play = Exclude<Phase, "attract">;
const CONFS: { v: Confidence; label: string }[] = [
  { v: 0.5, label: "Unsure" },
  { v: 0.75, label: "Likely" },
  { v: 0.95, label: "Certain" },
];
const WATCH_SPEED = 2; // pre frames play at 2x: ~10 s before the freeze
const TAPE_SPEED = 3; // the recording plays at 3x, slowing down near the closest approach

/** Mutable per-frame state for the animation loop (kept out of React state so it can change 60 times a second). */
interface Loop {
  t: number;
  phase: Play;
  coneP: number;
  cam: View;
  zoom: number;
  pan: [number, number];
  dragging: boolean;
  slow: boolean;
  freezeAt: number;
  views: { wide: View; tape: View } | null;
  done: boolean;
}

/** What the panels show, refreshed ~10 times a second from the loop. */
interface Snap { t: number; sep: number | null; slow: boolean; alert: boolean; scaleFt: number; scalePx: number; decisionS: number }

export default function Game({ data, shift, keyOpen, setKeyOpen, onRound, onIndex, onEnd }: {
  data: GameData; shift: Shift; keyOpen: boolean; setKeyOpen: (f: (k: boolean) => boolean) => void;
  onRound: (r: Round) => void; onIndex: (i: number) => void; onEnd: () => void;
}) {
  const { map } = data;
  const [i, setI] = useState(0);
  const [pz, setPz] = useState<Puzzle | null>(null);
  const [phase, setPhaseState] = useState<Play>("watch");
  const [call, setCall] = useState<Call | null>(null);
  const [conf, setConf] = useState<Confidence | null>(null);
  const [round, setRound] = useState<Round | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [snap, setSnap] = useState<Snap>({ t: 0, sep: null, slow: false, alert: false, scaleFt: 1000, scalePx: 80, decisionS: 0 });
  const [toast, setToast] = useState<string | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; lines: string[] } | null>(null);

  const svgRef = useRef<SVGSVGElement>(null);
  const renderer = useRef<ScopeRenderer | null>(null);
  const loop = useRef<Loop>({ t: 0, phase: "watch", coneP: 0, cam: [0, 0, 1, 1], zoom: 1, pan: [0, 0], dragging: false, slow: false, freezeAt: 0, views: null, done: false });
  const entry = shift.list[i];
  const acRel = pz ? namedAircraftIsRunwayTraffic(map, pz) : false;

  const setPhase = useCallback((p: Play) => {
    loop.current.phase = p;
    if (p === "decide") loop.current.freezeAt = performance.now();
    setPhaseState(p);
  }, []);

  // ---- load the round's puzzle ----
  useEffect(() => {
    let cancelled = false;
    loadPuzzle(entry.file).then((p) => {
      if (cancelled) return;
      Object.assign(loop.current, { t: p.pre[0].t, phase: "watch", coneP: 0, zoom: 1, pan: [0, 0], done: false, views: null });
      setPz(p);
      setPhaseState("watch");
      setCall(null);
      setConf(null);
      setRound(null);
      setSelected(p.vehicle_id);
      setToast(`RND ${i + 1}/${shift.list.length} · ${data.index.difficulties.find((d) => d.id === p.difficulty)?.name ?? p.difficulty} · ${entry.title}`);
      setTimeout(() => setToast(i === 0 ? "New here? Press K for what the symbols and units mean. Scroll to zoom, drag to pan, hover a target for details." : null), 2600);
      setTimeout(() => setToast(null), i === 0 ? 8600 : 2600);
    });
    return () => { cancelled = true; };
  }, [entry, i, shift.list.length, data.index.difficulties]);

  // ---- build the scope for this puzzle and keep it sized to its box ----
  useEffect(() => {
    const svg = svgRef.current;
    if (!pz || !svg) return;
    const r = new ScopeRenderer(svg, map, data.background);
    renderer.current = r;
    const fit = () => {
      const ratio = (svg.clientWidth || 900) / (svg.clientHeight || 500);
      const views = defaultViews(map, pz, ratio);
      loop.current.views = views;
      r.build(pz, views.wide);
      loop.current.cam = camTarget(loop.current);
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(svg);
    return () => ro.disconnect();
  }, [pz, map, data.background]);

  // ---- the animation loop ----
  const finish = useCallback(() => {
    const L = loop.current;
    if (!pz) return;
    L.t = pz.reveal[pz.reveal.length - 1].t;
    if (!L.done && round) {
      L.done = true;
      saveRound(round);
      onRound(round);
    }
    setPhase("result");
  }, [pz, round, onRound, setPhase]);

  useEffect(() => {
    if (!pz) return;
    let raf = 0;
    let last = performance.now();
    let lastSnap = 0;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const coneIds = coneIdsFor(map, pz);
    const step = (now: number) => {
      const L = loop.current;
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (L.phase === "watch") {
        L.t += dt * WATCH_SPEED;
        if (L.t >= pz.decision_t) {
          L.t = pz.decision_t;
          setPhase("decide");
        }
      } else if (L.phase === "ai") {
        L.coneP = reduce ? 1 : Math.min(1, L.coneP + dt / 1.2);
      } else if (L.phase === "reveal") {
        L.slow = !reduce && Math.abs(L.t - pz.outcome.min_sep_t) < 2.5;
        L.t += dt * (L.slow ? 0.8 : TAPE_SPEED);
        if (L.t >= pz.reveal[pz.reveal.length - 1].t) finish();
      }
      if (L.views && !L.dragging) {
        const target = camTarget(L);
        const a = reduce ? 1 : Math.min(1, dt * 2);
        L.cam = L.cam.map((c, j) => c + (target[j] - c) * a) as View;
      }
      const revealed = L.phase === "reveal" || L.phase === "result";
      const sep = L.phase === "result" ? pz.outcome.min_sep_m : L.phase === "reveal" ? separationAfterReveal(pz, L.t) : null;
      const alert = revealed && sep != null && sep < pz.outcome.threshold_m;
      renderer.current?.draw(
        { pz, phase: L.phase, t: L.t, call: round?.human.call, coneProgress: revealed ? 1 : L.coneP, selected, alert, focusAircraft: acRel, coneIds },
        L.cam, now,
      );
      if (now - lastSnap > 100) {
        lastSnap = now;
        const mpp = L.cam[2] / (svgRef.current?.clientWidth || 900);
        const scaleFt = [50, 100, 200, 500, 1000, 2000, 5000].find((v) => v / M_TO_FT / mpp >= 70) ?? 5000;
        setSnap({ t: L.t, sep, slow: L.slow && L.phase === "reveal", alert, scaleFt, scalePx: scaleFt / M_TO_FT / mpp, decisionS: L.phase === "decide" ? (now - L.freezeAt) / 1000 : 0 });
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [pz, map, acRel, round, selected, finish, setPhase]);

  // ---- actions ----
  const skipToDecision = useCallback(() => { if (pz) { loop.current.t = pz.decision_t; setPhase("decide"); } }, [pz, setPhase]);
  const lockIn = useCallback(() => {
    if (!pz || !call || !conf) return;
    setRound(makeRound(pz, { call, confidence: conf, decision_ms: performance.now() - loop.current.freezeAt }, shift.player, shift.sessionId));
    loop.current.coneP = 0;
    setPhase("ai");
  }, [pz, call, conf, shift.player, shift.sessionId, setPhase]);
  const rollTape = useCallback(() => { if (pz) { loop.current.t = pz.decision_t; setPhase("reveal"); } }, [pz, setPhase]);
  const next = useCallback(() => {
    if (i + 1 >= shift.list.length) onEnd();
    else { setI(i + 1); onIndex(i + 1); }
  }, [i, shift.list.length, onEnd, onIndex]);
  const zoomBy = useCallback((f: number, fx = 0.5, fy = 0.5) => zoomAt(loop.current, f, fx, fy), []);
  const resetView = useCallback(() => { loop.current.zoom = 1; loop.current.pan = [0, 0]; }, []);

  // ---- keyboard ----
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      const k = e.key.toLowerCase();
      if (k === "k") return setKeyOpen((x) => !x);
      if (k === "escape") return setKeyOpen(() => false);
      if (k === "=" || k === "+") return zoomBy(1.35);
      if (k === "-") return zoomBy(1 / 1.35);
      if (k === "0") return resetView();
      if (phase === "decide") {
        if (k === "c") setCall("CLEAR");
        else if (k === "h") setCall("HOLD");
        else if (k === "1" || k === "2" || k === "3") setConf(CONFS[Number(k) - 1].v);
        else if (k === "enter") { e.preventDefault(); lockIn(); }
      } else if (k === " " && (phase === "watch" || phase === "reveal")) {
        e.preventDefault();
        if (phase === "watch") skipToDecision(); else finish();
      } else if (k === "enter") {
        e.preventDefault();
        if (phase === "ai") rollTape(); else if (phase === "result") next();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, lockIn, rollTape, next, finish, skipToDecision, zoomBy, resetView, setKeyOpen]);

  // ---- scope mouse: wheel zoom, drag pan, click select, hover details ----
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    let drag: { x: number; y: number; pan: [number, number]; cam: View; moved: boolean; id: number } | null = null;
    let suppressClick = false;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = svg.getBoundingClientRect();
      zoomAt(loop.current, e.deltaY < 0 ? 1.2 : 1 / 1.2, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
    };
    const onDown = (e: PointerEvent) => { const L = loop.current; drag = { x: e.clientX, y: e.clientY, pan: [...L.pan], cam: [...L.cam] as View, moved: false, id: e.pointerId }; };
    const onMove = (e: PointerEvent) => {
      const L = loop.current;
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (drag.moved || Math.hypot(dx, dy) >= 4) {
          if (!drag.moved) { drag.moved = true; L.dragging = true; svg.classList.add("drag"); try { svg.setPointerCapture(drag.id); } catch { /* ignore */ } }
          const s = drag.cam[2] / svg.clientWidth;
          L.pan = [drag.pan[0] - dx * s, drag.pan[1] - dy * s];
          L.cam = [drag.cam[0] - dx * s, drag.cam[1] - dy * s, drag.cam[2], drag.cam[3]];
          setTip(null);
          return;
        }
      }
      const g = (e.target as Element).closest?.("[data-id]");
      const h = g && renderer.current?.last.get(g.getAttribute("data-id") ?? "");
      setTip(h && pz ? { x: e.clientX, y: e.clientY, lines: hoverLines(h, map, pz) } : null);
    };
    const onUp = () => { if (drag?.moved) { suppressClick = true; setTimeout(() => (suppressClick = false), 0); } drag = null; loop.current.dragging = false; svg.classList.remove("drag"); };
    const onClick = (e: MouseEvent) => {
      if (suppressClick) return;
      const g = (e.target as Element).closest?.("[data-id]");
      if (g) setSelected(g.getAttribute("data-id"));
    };
    const onLeave = () => setTip(null);
    svg.addEventListener("wheel", onWheel, { passive: false });
    svg.addEventListener("pointerdown", onDown);
    svg.addEventListener("pointermove", onMove);
    svg.addEventListener("pointerup", onUp);
    svg.addEventListener("pointercancel", onUp);
    svg.addEventListener("click", onClick);
    svg.addEventListener("pointerleave", onLeave);
    return () => {
      svg.removeEventListener("wheel", onWheel);
      svg.removeEventListener("pointerdown", onDown);
      svg.removeEventListener("pointermove", onMove);
      svg.removeEventListener("pointerup", onUp);
      svg.removeEventListener("pointercancel", onUp);
      svg.removeEventListener("click", onClick);
      svg.removeEventListener("pointerleave", onLeave);
    };
  }, [pz, map]);

  if (!pz) return <div className="loading">Loading round…</div>;

  const rw = runwayIdOf(map, pz);
  const vehicleId = callsign(labelOf(pz, pz.vehicle_id));
  const acId = callsign(labelOf(pz, pz.aircraft_id));
  const status = phase === "watch" ? "LIVE" : phase === "decide" || phase === "ai" ? "FROZEN" : phase === "reveal" ? (snap.slow ? "SLOW" : `TAPE ${TAPE_SPEED}×`) : "TAPE END";

  return (
    <section className="view">
      <div className="ws">
        <div className="top3">
          <Strips pz={pz} data={data} t={snap.t} round={round} acRel={acRel} selected={selected} onSelect={setSelected} />
          <div className="scopewrap">
            <svg id="scope" ref={svgRef} role="img" aria-label="Surface surveillance scope" />
            <div className="sov tl">ASDE · {map.airport} · {zoneOf(map, pz).name}</div>
            <div className="sov tr"><span className={`phase ${status === "FROZEN" ? "frozen" : status === "SLOW" ? "slow" : ""}`}>{status}</span></div>
            <div className="sov bl">
              <svg width={Math.round(snap.scalePx)} height="8" style={{ display: "block" }}><path d={`M0.5 0 V7 H${(snap.scalePx - 0.5).toFixed(1)} V0`} fill="none" stroke="currentColor" /></svg>
              {snap.scaleFt} ft
            </div>
            <div className="sov br">{map.background && <span className="osm">© OpenStreetMap contributors · </span>}N ↑</div>
            {snap.alert && <div className="alertbar">RWY {rw} ALERT {acId} / {vehicleId}</div>}
            {toast && <div className="toast">{toast}</div>}
            <div className="zoomctl" aria-label="Scope zoom">
              <button title="Zoom in (or scroll)" onClick={() => zoomBy(1.35)}>+</button>
              <button title="Zoom out" onClick={() => zoomBy(1 / 1.35)}>−</button>
              <button id="zFit" title="Reset view" onClick={resetView}>Fit</button>
            </div>
            {keyOpen && <KeyPanel onClose={() => setKeyOpen(() => false)} />}
          </div>
          <InfoDisplay data={data} pz={pz} phase={phase} t={snap.t} sep={snap.sep} acRel={acRel} />
        </div>
        <Console
          data={data} pz={pz} phase={phase} call={call} conf={conf} round={round} acRel={acRel} decisionS={snap.decisionS}
          index={i} total={shift.list.length} rounds={shift.rounds}
          onCall={setCall} onConf={setConf} onLock={lockIn} onTape={rollTape} onSkip={phase === "watch" ? skipToDecision : finish} onNext={next}
          onFocus={setSelected} t={snap.t}
        />
      </div>
      {tip && (
        <div id="tip" style={{ left: Math.min(window.innerWidth - 300, tip.x + 14), top: Math.min(window.innerHeight - 120, tip.y + 14) }}>
          {tip.lines.map((l, j) => (j ? <div key={j}>{l}</div> : <b key={j}>{l}</b>))}
        </div>
      )}
    </section>
  );
}

// ---------- camera ----------
function baseView(L: Loop): View {
  return L.phase === "reveal" || L.phase === "result" ? L.views!.tape : L.views!.wide;
}
function camTarget(L: Loop): View {
  const T = baseView(L);
  const w = T[2] / L.zoom, h = T[3] / L.zoom;
  const cx = T[0] + T[2] / 2 + L.pan[0], cy = T[1] + T[3] / 2 + L.pan[1];
  return [cx - w / 2, cy - h / 2, w, h];
}
/** Zoom by f keeping the point under (fx, fy) of the scope fixed. */
function zoomAt(L: Loop, f: number, fx: number, fy: number) {
  if (!L.views) return;
  const nz = Math.max(0.6, Math.min(8, L.zoom * f));
  f = nz / L.zoom;
  if (f === 1) return;
  L.zoom = nz;
  const C = L.cam;
  const wx = C[0] + fx * C[2], wy = C[1] + fy * C[3];
  const nw = C[2] / f, nh = C[3] / f;
  const x0 = wx - fx * nw, y0 = wy - fy * nh;
  const T = baseView(L);
  L.pan = [x0 + nw / 2 - (T[0] + T[2] / 2), y0 + nh / 2 - (T[1] + T[3] / 2)];
  L.cam = [x0, y0, nw, nh];
}

function hoverLines(h: Hover, map: GameData["map"], pz: Puzzle): string[] {
  const v = h.speed ?? 0;
  const lines = [
    `${callsign(h.label)} · ${h.type === "aircraft" ? "Aircraft" : h.type === "vehicle" ? "Ground vehicle" : "Unidentified"}`,
    `Speed ${Math.round(v * M_S_TO_KT)} kt (${Math.round(v * M_S_TO_KT * KT_TO_MPH)} mph, ${Math.round(v * 3.6)} km/h)`,
    `Heading ${String(Math.round(h.heading ?? 0)).padStart(3, "0")}°`,
  ];
  const onRunway = map.runways.find((r) => inPoly([h.x, h.y], r.polygon as Vec[]));
  lines.push(onRunway ? `On runway ${onRunway.id}` : "On a taxiway or apron");
  if (h.status !== "ok") lines.push(h.status === "lost" ? `Track lost: last seen ${fmt(h.age_s)} s ago` : `Position report ${fmt(h.age_s)} s old`);
  if (h.id === pz.vehicle_id) lines.push("This is the vehicle asking to cross");
  if (h.traffic) {
    const e = etaToCrossing(map, pz, h);
    lines.push(e.eta != null ? `Runway traffic: reaches the crossing in about ${fmt(e.eta)} s` : e.past ? "Runway traffic: already past the crossing" : "Runway traffic: not heading for the crossing");
  }
  return lines;
}

// ---------- flight strips ----------
function Strips({ pz, data, t, round, acRel, selected, onSelect }: {
  pz: Puzzle; data: GameData; t: number; round: Round | null; acRel: boolean; selected: string | null; onSelect: (id: string) => void;
}) {
  const { map } = data;
  const tt = Math.min(t, pz.decision_t);
  const rw = runwayIdOf(map, pz);
  const tw = taxiwayOf(map, pz);
  const traffic = runwayTraffic(map, pz, tt).map((x) => x.id);
  const ground = pz.objects.filter((o) => o.type === "aircraft" && !traffic.includes(o.id)).map((o) => o.id);
  const others = pz.objects.filter((o) => o.type !== "aircraft" && o.id !== pz.vehicle_id).length;

  const strip = (id: string) => {
    const m = pz.objects.find((o) => o.id === id)!;
    const s = at(pz.pre, tt, id);
    const isFocus = id === pz.vehicle_id || (acRel && id === pz.aircraft_id);
    const color = m.type === "vehicle" ? "var(--veh)" : traffic.includes(id) ? "var(--alert)" : "var(--muted)";
    const onRw = s ? map.runways.find((r) => inPoly([s.x, s.y], r.polygon as Vec[])) : undefined;
    let state = id === pz.vehicle_id ? `HOLDING SHORT RWY ${rw}${tw ? ` AT ${tw}` : ""}` : onRw ? `ON RWY ${onRw.id}` : traffic.includes(id) ? `LINED UP RWY ${rw}` : s && kt(s.speed) > 3 ? "TAXIING" : "STOPPED";
    if (s?.status === "lost") state += " · TRACK LOST";
    if (traffic.includes(id) && s) {
      const e = etaToCrossing(map, pz, s);
      if (e.eta != null) state += ` · XING ${fmt(e.eta)} s`;
      else if (e.past) state += " · PAST XING";
    }
    return (
      <div key={id} className={`strip ${isFocus ? "focus" : ""} ${selected === id ? "sel" : ""}`} style={{ ["--c" as string]: color }} onClick={() => onSelect(id)}>
        <span className="edge" />
        <div className="body">
          <span className="cs">{callsign(m.label)}</span>
          <span className="cat">{category(m.type, m.label, s)} {s?.speed != null ? kt(s.speed) : "–"} kt</span>
          <span className="st">{state}</span>
          {id === pz.vehicle_id && round && (
            <span className="mark" style={{ color: round.human.call === "CLEAR" ? "var(--clear-c)" : "var(--hold-c)" }}>
              {round.human.call === "CLEAR" ? "X" : "HS"} {rw} · {round.human.confidence * 100}%
            </span>
          )}
        </div>
      </div>
    );
  };
  const bay = (name: string, ids: string[], empty: string) => (
    <>
      <div className="bay"><span>{name}</span><span>{ids.length}</span></div>
      {ids.length ? ids.map(strip) : <p className="bayempty">{empty}</p>}
    </>
  );
  return (
    <div className="pane strips">
      <div className="ph"><span>Strips</span></div>
      <div className="pb">
        {bay("Crossing", [pz.vehicle_id], "")}
        {bay(`Runway ${rw}`, traffic, "No traffic on or lined up with the runway.")}
        {bay("Ground aircraft", ground, "None")}
        <p className="bayempty">{others} other vehicles are on the scope only. Click any target to inspect it.</p>
      </div>
    </div>
  );
}

// ---------- information display ----------
function Dist({ m }: { m: number }) {
  const d = distanceText(m);
  return <>{d.main} <small>({d.alt})</small></>;
}
function K({ text, tip }: { text: string; tip: string }) {
  return <span className="k" title={tip}>{text}</span>;
}
function InfoDisplay({ data, pz, phase, t, sep, acRel }: { data: GameData; pz: Puzzle; phase: Play; t: number; sep: number | null; acRel: boolean }) {
  const { map } = data;
  const e = estimates(map, pz, t);
  const thr = pz.outcome.threshold_m;
  const rw = runwayIdOf(map, pz);
  const tw = taxiwayOf(map, pz);
  const deg = e.degraded[0];
  const revealed = phase === "reveal" || phase === "result";
  const pos = (v: number) => Math.min(100, 100 * Math.sqrt(v / Math.max(thr * 4, 400)));
  return (
    <div className="pane ids">
      <div className="ph"><span>Info display</span><span>{pz.difficulty === "fog" ? "DEGRADED" : ""}</span></div>
      <div className="pb">
        <div className="blk"><K text="Time · UTC" tip="Coordinated Universal Time, also called Zulu (Z). Aviation uses it so clocks agree across time zones." /><span className="clock">{clockAt(pz, revealed ? t : Math.min(t, pz.decision_t))}</span></div>
        <div className="blk">
          <K text="Crossing" tip="The runway the vehicle wants to cross." />
          <div className="kv"><span>Runway</span><b>{rw}</b></div>
          {tw ? <div className="kv"><span>At taxiway</span><b>{tw}</b></div> : <div className="kv"><span>Zone</span><b>{zoneOf(map, pz).name.replace(/^RWY\s*[\w/]+\s*/i, "")}</b></div>}
          <div className="kv" title="If the vehicle and the runway traffic get closer than this while the vehicle is on the runway, it is a conflict and HOLD was right."><span>Safe distance</span><b>{fmt(thr * M_TO_FT)} ft <small>({fmt(thr)} m)</small></b></div>
        </div>
        <div className="blk">
          <K text="Surveillance" tip="How fresh the position data is. In fog mode some reports are old or missing." />
          <div className="kv"><span>Data</span><b className={deg ? "warnc" : "good"}>{deg ? (deg.status === "lost" ? "TRACK LOST" : "DATA OLD") : "UP TO DATE"}</b></div>
          {deg && <div className="kv"><span>{callsign(labelOf(pz, deg.id))} last seen</span><b className="warnc">{fmt(deg.age_s)} s ago</b></div>}
        </div>
        {revealed ? (
          acRel || pz.outcome.conflict ? (
            <div className="blk">
              <K text={`${phase === "result" ? "Closest approach" : "Distance now"} · ${callsign(labelOf(pz, pz.aircraft_id))}`} tip="Distance between the vehicle and the runway traffic. The white tick is the safe distance." />
              <b className={`big ${sep != null && sep < thr ? "bad" : ""}`}>{sep != null ? <Dist m={sep} /> : "–"}</b>
              <div className={`sepbar ${sep != null && sep < thr ? "bad" : ""}`}><i style={{ width: `${sep != null ? pos(sep) : 0}%` }} /><u style={{ left: `${pos(thr)}%` }} /></div>
            </div>
          ) : (
            <div className="blk"><K text={`Runway ${rw} traffic`} tip="Aircraft on the runway or lined up with it." /><b className="big good">NONE</b><span className="note">No aircraft used the runway during the crossing.</span></div>
          )
        ) : (
          <>
            <div className="blk">
              <K text={`Runway ${rw} traffic`} tip="Aircraft on the runway or lined up with it. Right column: seconds until it reaches the crossing at its current speed." />
              {e.traffic.length ? (
                <>
                  {e.traffic.slice(0, 4).map((x) => (
                    <div className="kv" key={x.id}>
                      <span>{callsign(x.label)} · {kt(x.s.speed)} kt{x.s.status !== "ok" ? (x.s.status === "lost" ? " · lost" : " · old") : ""}</span>
                      <b>{x.eta.eta != null ? `in ${fmt(x.eta.eta)} s` : x.eta.past ? "passed" : x.eta.away ? "not toward" : "stopped"}</b>
                    </div>
                  ))}
                  <span className="note">Time until it reaches the crossing at its current speed.</span>
                </>
              ) : (
                <><b className="big good">NONE</b><span className="note">No aircraft on or lined up with the runway.</span></>
              )}
            </div>
            <div className="blk">
              <K text={`${callsign(labelOf(pz, pz.vehicle_id))} · to cross`} tip="How far the vehicle must drive to be clear of the runway, and how long that takes at the speed it was driving before it stopped." />
              <div className="kv"><span>Distance</span><b><Dist m={e.crossDistance} /></b></div>
              <div className="kv"><span>Time at {kt(e.vehicleSpeed)} kt{e.vehicleSpeedAssumed ? " (assumed)" : ""}</span><b>{fmt(e.vehicleNeeds)} s</b></div>
            </div>
            {e.margin != null && e.first ? (
              <div className="blk">
                <K text={`Margin vs ${callsign(e.first.label)}`} tip="Aircraft arrival time minus the vehicle's crossing time. Positive: the vehicle should be clear first. Negative: the aircraft gets there first." />
                <b className={`big ${e.margin < 0 ? "bad" : "good"}`}>{e.margin >= 0 ? "+" : ""}{fmt(e.margin)} s</b>
                <span className="note">{e.margin >= 0 ? `Vehicle clears about ${fmt(e.margin)} s before the aircraft arrives, if nobody changes speed.` : `Aircraft arrives about ${fmt(-e.margin)} s before the vehicle would be clear, if nobody changes speed.`}</span>
                <span className="note">A rough estimate: ignores braking, turn-offs and take-off.{deg ? " Some data is old." : ""}</span>
              </div>
            ) : e.traffic.length ? (
              <div className="blk"><K text="Margin" tip="" /><b className="good">No traffic heading for the crossing</b></div>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

// ---------- radio console ----------
function Console(p: {
  data: GameData; pz: Puzzle; phase: Play; call: Call | null; conf: Confidence | null; round: Round | null; acRel: boolean; decisionS: number;
  index: number; total: number; rounds: Round[]; t: number;
  onCall: (c: Call) => void; onConf: (c: Confidence) => void; onLock: () => void; onTape: () => void; onSkip: () => void; onNext: () => void; onFocus: (id: string) => void;
}) {
  const { data, pz, phase, call, conf, round } = p;
  const { map } = data;
  const rw = runwayIdOf(map, pz);
  const pips = (
    <div className="pips">
      {Array.from({ length: p.total }, (_, j) => {
        const r = p.rounds[j];
        const cls = r ? (r.human.points > r.ai.points ? "w" : r.human.points < r.ai.points ? "l" : "d") : j === p.index ? "cur" : "";
        return <i key={j} className={cls} />;
      })}
    </div>
  );
  const ac = at(pz.pre, Math.min(p.t, pz.decision_t), pz.aircraft_id);
  // Who the question is about: shown in every phase, click to highlight on the scope.
  const subjects = (
    <div className="subjects">
      <span className="subject" style={{ ["--c" as string]: "var(--veh)" }} onClick={() => p.onFocus(pz.vehicle_id)}>
        <span>Vehicle</span><b>{callsign(labelOf(pz, pz.vehicle_id))}</b><span>holding short RWY {rw}</span>
      </span>
      {p.acRel && (
        <span className="subject" style={{ ["--c" as string]: "var(--alert)" }} onClick={() => p.onFocus(pz.aircraft_id)}>
          <span>Traffic</span><b>{callsign(labelOf(pz, pz.aircraft_id))}</b><span>{ac ? `${kt(ac.speed)} kt` : ""}{ac && ac.status !== "ok" ? " · data old" : ""} on RWY {rw}</span>
        </span>
      )}
    </div>
  );
  const side = (h: string, col: string, c: Call, meta: string, pts?: number, win?: boolean) => (
    <div className={`side ${win ? "win" : ""}`} style={{ ["--c" as string]: col }}>
      <span className="h"><span>{h}</span>{win && <span>ROUND</span>}</span>
      <span className={`v ${c}`}>{c}</span>
      <span className="m">{meta}</span>
      {pts != null && <span className="p">+{pts}</span>}
    </div>
  );

  if (phase === "watch")
    return (
      <div className="console">
        <div><p className="cl">Situation</p><p className="q">Monitoring. Traffic freezes when {shortLabel(labelOf(pz, pz.vehicle_id))} reaches the hold line in <b className="mono">{fmt(Math.max(0, (pz.decision_t - p.t) / WATCH_SPEED))}</b> s.</p>{subjects}</div>
        <div className="stack"><p className="cl">Shift</p>{pips}</div>
        <div className="stack" style={{ justifyItems: "end" }}><button className="secondary" onClick={p.onSkip}>Skip to decision<kbd>Space</kbd></button></div>
      </div>
    );

  if (phase === "decide") {
    const stakes = call && conf ? `If right +${score(call, conf, call === "HOLD").points} · if wrong +${score(call, conf, call !== "HOLD").points}` : "Choose a call and a confidence.";
    return (
      <div className="console">
        <div><p className="cl">Situation</p><p className="q">{pz.question}</p>{subjects}</div>
        <div>
          <p className="cl">Call and confidence</p>
          <div className="callrow">
            <button className="call c" aria-pressed={call === "CLEAR"} onClick={() => p.onCall("CLEAR")}>CLEAR<kbd>C</kbd></button>
            <button className="call h" aria-pressed={call === "HOLD"} onClick={() => p.onCall("HOLD")}>HOLD<kbd>H</kbd></button>
          </div>
          <div className="confrow">
            {CONFS.map((o, j) => (
              <button key={o.v} className="conf" aria-pressed={conf === o.v} onClick={() => p.onConf(o.v)}>{o.v * 100}%<small>{o.label} · {j + 1}</small></button>
            ))}
          </div>
          <div className="stakes">{stakes}</div>
        </div>
        <div className="radio">
          <p className="cl">Transmission <span className="timer">{fmt(p.decisionS, 1)} s</span></p>
          <div className={`tx ${call ? "" : "empty"}`}>{call ? <><span className="who">TWR</span> {phrase(map, pz, call)}</> : "Your instruction will read here."}</div>
          <button className="transmit" disabled={!call || !conf} onClick={p.onLock}>TRANSMIT<kbd>Enter</kbd></button>
        </div>
      </div>
    );
  }

  if (!round) return <div className="console" />;
  const det = pz.ai.detector.toUpperCase();

  if (phase === "ai")
    return (
      <div className="console">
        <div className="duel">
          {side("You", "var(--you)", round.human.call, `${round.human.confidence * 100}% · ${fmt(round.human.decision_ms / 1000, 1)} s`)}
          {side(`AI · ${det}`, "var(--ai)", pz.ai.call, `${pz.ai.confidence * 100}%`)}
        </div>
        <div>
          <p className="cl">AI reasoning</p>
          <p className="reason">{speedsToKnots(pz.ai.reasoning)}</p>
          {pz.ai.baselines && (
            <div className="base">
              {Object.entries(pz.ai.baselines).filter(([d]) => d !== pz.ai.detector).map(([d, b]) => `${d.toUpperCase()}: ${b!.call} ${b!.confidence * 100}%`).join(" · ")}
            </div>
          )}
        </div>
        <div className="stack" style={{ justifyItems: "end" }}>
          <button className="transmit" style={{ width: "100%" }} onClick={p.onTape}>PLAY RECORDING<kbd>Enter</kbd></button>
          <span className="note">The AI&apos;s forecast is drawn on the scope.</span>
        </div>
      </div>
    );

  if (phase === "reveal")
    return (
      <div className="console">
        <div className="duel">
          {side("You", "var(--you)", round.human.call, `${round.human.confidence * 100}%`)}
          {side("AI", "var(--ai)", pz.ai.call, `${pz.ai.confidence * 100}%`)}
        </div>
        <div className="stack"><p className="cl">Recording</p><p className="q" style={{ color: "var(--muted)" }}>What really happened after the freeze.</p></div>
        <div className="stack" style={{ justifyItems: "end" }}><button className="secondary" onClick={p.onSkip}>Skip to result<kbd>Space</kbd></button></div>
      </div>
    );

  // result
  const oc = pz.outcome;
  const hc = round.human.call;
  const verdict: [string, "good" | "bad", string] =
    hc === "CLEAR"
      ? oc.conflict ? ["Loss of separation", "bad", "You cleared it with traffic coming."] : ["Good crossing", "good", "Cleared, and separation held."]
      : oc.conflict ? ["Good hold", "good", "It crossed in the recording. Your hold would have prevented this."] : ["Unneeded hold", "bad", "Separation held in the recording. The hold cost time."];
  const whatIf = !!pz.source.vehicle_time_shift_s;
  const hw = round.human.points > round.ai.points;
  const aw = round.ai.points > round.human.points;
  return (
    <div className="console">
      <div className="verdict">
        <b className={verdict[1]}>{verdict[0]} {whatIf && <span className="wi">WHAT IF</span>}</b>
        <span>{oc.note} {verdict[2]}</span>
        {pz.ai.coach && <p className="coach">{pz.ai.coach}</p>}
      </div>
      <div className="duel">
        {side("You", "var(--you)", round.human.call, `${round.human.confidence * 100}% · Brier ${fmt(round.human.brier, 3)}`, round.human.points, hw)}
        {side(`AI · ${det}`, "var(--ai)", pz.ai.call, `${pz.ai.confidence * 100}% · Brier ${fmt(round.ai.brier, 3)}`, round.ai.points, aw)}
      </div>
      <div className="stack" style={{ justifyItems: "end" }}>
        {pips}
        <button className="secondary" style={{ width: "100%" }} onClick={p.onTape}>Replay recording</button>
        <button className="transmit" style={{ width: "100%" }} onClick={p.onNext}>{p.index + 1 < p.total ? "NEXT ROUND" : "END SHIFT"}<kbd>Enter</kbd></button>
      </div>
    </div>
  );
}

