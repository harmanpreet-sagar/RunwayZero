"use client";

import type { Puzzle } from "@contracts/types";
import { useEffect, useRef, useState } from "react";
import { type GameData, loadPuzzle } from "@/lib/data";
import { centroid, type Vec } from "@/lib/geom";
import { at, callsign, clockAt, labelOf, runwayIdOf, zoneOf } from "@/lib/puzzle";
import { ScopeRenderer } from "@/lib/scope";
import { M_TO_FT, fmt, kt } from "@/lib/units";

/** Live scope preview on the landing page: loops one puzzle's PRE frames only (never the reveal, so nothing is spoiled). */
function AttractScope({ data }: { data: GameData }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [info, setInfo] = useState<{ clock: string; targets: number; acId: string; acSpeed: string; vhId: string; vhState: string; rw: string; zone: string } | null>(null);

  useEffect(() => {
    const entry = data.index.puzzles.find((p) => p.difficulty === "busy") ?? data.index.puzzles[0];
    if (!entry || !svgRef.current) return;
    let raf = 0;
    let cancelled = false;
    loadPuzzle(entry.file).then((pz: Puzzle) => {
      if (cancelled || !svgRef.current) return;
      const svg = svgRef.current;
      const r = new ScopeRenderer(svg, data.map, data.background);
      const [cx, cy] = centroid(zoneOf(data.map, pz).polygon as Vec[]);
      const W = 3000;
      const ratio = (svg.clientWidth || 700) / (svg.clientHeight || 420);
      const view: [number, number, number, number] = [cx - W * 0.6, -cy - W / ratio / 2, W, W / ratio];
      r.build(pz, view);
      const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
      let t = pz.pre[0].t;
      let last = performance.now();
      let lastInfo = -1;
      const loop = (now: number) => {
        t += reduce ? 0 : (Math.min(0.1, (now - last) / 1000)) * 1.5;
        last = now;
        if (t >= pz.decision_t) t = pz.pre[0].t;
        r.draw({ pz, phase: "attract", t, focusAircraft: false, coneIds: new Set() }, view, now);
        if (Math.floor(t) !== lastInfo) {
          lastInfo = Math.floor(t);
          const ac = at(pz.pre, t, pz.aircraft_id);
          const vh = at(pz.pre, t, pz.vehicle_id);
          setInfo({
            clock: clockAt(pz, t),
            targets: pz.pre.find((f) => f.t >= t)?.objects.length ?? pz.objects.length,
            acId: callsign(labelOf(pz, pz.aircraft_id)),
            acSpeed: ac ? `${kt(ac.speed)} kt` : "–",
            vhId: callsign(labelOf(pz, pz.vehicle_id)),
            vhState: vh && (vh.speed ?? 0) > 0.5 ? "TAXIING" : "HOLDING",
            rw: runwayIdOf(data.map, pz),
            zone: zoneOf(data.map, pz).name,
          });
        }
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    });
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [data]);

  return (
    <div className="preview" aria-hidden="true">
      <div className="ph"><span>ASDE · Surface</span><span className="rec"><i />Recorded replay</span></div>
      <div className="pscope">
        <svg id="attract" ref={svgRef} />
        <div className="sov tl">{data.map.airport} · {info?.zone}</div>
        <div className="sov tr">{info?.clock}</div>
        <div className="sov br">{data.map.background && <span className="osm">© OpenStreetMap contributors · </span>}N ↑</div>
      </div>
      <div className="pstrip">
        <div><span>Targets</span><b>{info?.targets ?? "–"}</b></div>
        <div><span>{info?.acId ?? "Aircraft"}</span><b>{info?.acSpeed ?? "–"}</b></div>
        <div><span>{info?.vhId ?? "Vehicle"}</span><b>{info?.vhState ?? "–"}</b></div>
        <div><span>Crossing</span><b>RWY {info?.rw}</b></div>
      </div>
    </div>
  );
}

export default function SignIn({ data, player, onNext }: { data: GameData; player: string; onNext: (name: string) => void }) {
  const [name, setName] = useState(player);
  return (
    <section className="view">
      <div className="landing">
        <form
          className="signin"
          onSubmit={(e) => {
            e.preventDefault();
            const n = name.trim().toUpperCase().slice(0, 12);
            if (n) onNext(n);
          }}
        >
          <div className="kicker"><i />Tower position · recorded traffic</div>
          <h1>RUNWAY <span>ZERO</span></h1>
          <p className="tagline">Clear it or hold it. Can you make the call better than an AI?</p>
          <p>Real recorded traffic from Boston Logan. A vehicle is holding short of a runway. You are on the tower frequency.</p>
          <div className="field">
            <label htmlFor="nick">Operating initials or call sign</label>
            <input id="nick" maxLength={12} autoComplete="off" placeholder="TWR 1" required value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="row"><button className="btn primary" type="submit">Next →</button></div>
          <dl className="facts">
            <dt>Crossings</dt><dd>{data.index.puzzles.length}</dd>
            <dt>Conditions</dt><dd>{data.index.difficulties.length}</dd>
            <dt>Safe distance</dt><dd>{fmt(data.index.conflict_threshold_m * M_TO_FT)} ft</dd>
          </dl>
          <p className="fine">A practice tool for controllers in training and airside drivers. Not a certified simulator or an ATC system.</p>
        </form>
        <AttractScope data={data} />
      </div>
    </section>
  );
}
