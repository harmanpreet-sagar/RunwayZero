"use client";

import type { Round } from "@contracts/types";
import { useState } from "react";
import type { GameData } from "@/lib/data";
import { loadRounds } from "@/lib/storage";
import { M_TO_FT, fmt } from "@/lib/units";

const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);
const DNAME: Record<string, string> = { cv: "Constant velocity", kf: "Kalman filter", ml: "Learned model" };

/** Grouped bars with a hover readout. Colours are CSS variables so both themes work. */
function Bars({ cats, catNames, series, get, ymax, yfmt, tipFor }: {
  cats: string[]; catNames: Record<string, string>; series: { key: string; color: string; name: string }[];
  get: (cat: string, key: string) => number | null | undefined; ymax: number; yfmt: (v: number) => string; tipFor: (cat: string, key: string, v: number) => string;
}) {
  const [tip, setTip] = useState<string | null>(null);
  const W = 440, H = 200, P = { l: 44, r: 8, t: 10, b: 24 };
  const cw = (W - P.l - P.r) / cats.length;
  const bw = Math.min(24, (cw - 16) / series.length - 2);
  return (
    <>
      <div className="lg">{series.map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.name}</span>)}</div>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {[0, 1, 2, 3, 4].map((i) => {
          const y = H - P.b - ((H - P.t - P.b) * i) / 4;
          return <g key={i}><line x1={P.l} x2={W - P.r} y1={y} y2={y} className="gl" /><text x={P.l - 6} y={y + 3} textAnchor="end" className="ax">{yfmt((ymax * i) / 4)}</text></g>;
        })}
        {cats.map((c, i) => {
          const x0 = P.l + i * cw + (cw - (bw + 2) * series.length) / 2;
          return (
            <g key={c}>
              {series.map((s, j) => {
                const v = get(c, s.key);
                if (v == null) return null;
                const h = Math.max(1, ((H - P.t - P.b) * v) / ymax), x = x0 + j * (bw + 2), y = H - P.b - h, r = Math.min(3, bw / 2, h);
                return <path key={s.key} className="mk" d={`M${x},${H - P.b} V${y + r} Q${x},${y} ${x + r},${y} H${x + bw - r} Q${x + bw},${y} ${x + bw},${y + r} V${H - P.b}Z`} style={{ fill: s.color }} onPointerEnter={() => setTip(tipFor(c, s.key, v))} onPointerLeave={() => setTip(null)} />;
              })}
              <text x={P.l + i * cw + cw / 2} y={H - 7} textAnchor="middle" className="ax">{catNames[c] ?? c}</text>
            </g>
          );
        })}
      </svg>
      <p className="note" style={{ minHeight: 18, margin: 0 }}>{tip ?? " "}</p>
    </>
  );
}

function Calibration({ rounds }: { rounds: Round[] }) {
  const W = 440, H = 220, P = { l: 44, r: 12, t: 10, b: 28 };
  const X = (v: number) => P.l + ((v - 0.4) / 0.6) * (W - P.l - P.r);
  const Y = (v: number) => H - P.b - v * (H - P.t - P.b);
  const who = [{ k: "human" as const, color: "var(--you)", name: "Humans", off: -4 }, { k: "ai" as const, color: "var(--ai)", name: "AI", off: 4 }];
  return (
    <>
      <div className="lg">{who.map((w) => <span key={w.k}><i style={{ background: w.color }} />{w.name}</span>)}</div>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => <g key={v}><line x1={P.l} x2={W - P.r} y1={Y(v)} y2={Y(v)} className="gl" /><text x={P.l - 6} y={Y(v) + 3} textAnchor="end" className="ax">{Math.round(v * 100)}%</text></g>)}
        {[0.5, 0.75, 0.95].map((v) => <text key={v} x={X(v)} y={H - P.b + 14} textAnchor="middle" className="ax">{Math.round(v * 100)}% sure</text>)}
        <line x1={X(0.4)} y1={Y(0.4)} x2={X(1)} y2={Y(1)} className="gl" strokeDasharray="4 3" />
        {who.map((w) => {
          const pts = [0.5, 0.75, 0.95].map((c) => {
            const s = rounds.filter((r) => r[w.k].confidence === c);
            return s.length ? { c, hit: s.filter((r) => (r[w.k].call === "HOLD") === r.conflict).length / s.length, n: s.length } : null;
          }).filter((p): p is { c: number; hit: number; n: number } => !!p);
          return (
            <g key={w.k}>
              {pts.length > 1 && <polyline points={pts.map((p) => `${X(p.c) + w.off},${Y(p.hit)}`).join(" ")} fill="none" style={{ stroke: w.color }} strokeWidth={2} />}
              {pts.map((p) => <circle key={p.c} cx={X(p.c) + w.off} cy={Y(p.hit)} r={5} style={{ fill: w.color, stroke: "var(--panel)" }} strokeWidth={2}><title>{`${w.name} · said ${p.c * 100}% · right ${Math.round(p.hit * 100)}% · n=${p.n}`}</title></circle>)}
            </g>
          );
        })}
      </svg>
    </>
  );
}

export default function Stats({ data }: { data: GameData }) {
  const [rounds] = useState<Round[]>(() => (typeof window === "undefined" ? [] : loadRounds()));
  const diffs = data.index.difficulties.map((d) => d.id);
  const dn = Object.fromEntries(data.index.difficulties.map((d) => [d.id, d.name]));
  const by = (d: string, w: "human" | "ai") => mean(rounds.filter((r) => r.difficulty === d).map((r) => r[w].brier));
  const mx = Math.max(0.25, ...diffs.flatMap((d) => [by(d, "human") ?? 0, by(d, "ai") ?? 0]));
  const M = data.metrics;
  return (
    <section className="view">
      <div className="scroll">
        <div className="card">
          <h2>Humans vs the AI</h2>
          <p className="note" style={{ margin: "4px 0 0" }}>
            {rounds.length ? "From rounds played on this device. A shared database would make this everyone who played." : "No rounds on this device yet. Play a shift and your numbers appear here."}
          </p>
        </div>
        <div className="kpis">
          <div className="kpi"><b>{new Set(rounds.map((r) => r.session_id)).size}</b><span>shifts</span></div>
          <div className="kpi"><b>{rounds.length}</b><span>rounds</span></div>
          <div className="kpi"><b style={{ color: "var(--you)" }}>{fmt(mean(rounds.map((r) => r.human.brier)), 3)}</b><span>human mean Brier (lower is better)</span></div>
          <div className="kpi"><b style={{ color: "var(--ai)" }}>{fmt(mean(rounds.map((r) => r.ai.brier)), 3)}</b><span>AI mean Brier</span></div>
        </div>
        <div className="charts">
          {rounds.length > 0 && (
            <div className="card chart">
              <h4>Who forecasts better?</h4>
              <p>Mean Brier score by difficulty. Lower is better.</p>
              <Bars cats={diffs} catNames={dn} series={[{ key: "human", color: "var(--you)", name: "Humans" }, { key: "ai", color: "var(--ai)", name: "AI" }]}
                get={(d, k) => by(d, k as "human" | "ai")} ymax={Math.ceil(mx * 10) / 10} yfmt={(v) => v.toFixed(2)} tipFor={(d, k, v) => `${k === "human" ? "Humans" : "AI"} · ${dn[d]} · ${v.toFixed(3)}`} />
            </div>
          )}
          {rounds.length > 0 && (
            <div className="card chart">
              <h4>Calibration</h4>
              <p>How often each stated confidence was right. On the dashed line is well calibrated; below it is overconfident.</p>
              <Calibration rounds={rounds} />
            </div>
          )}
          {M?.game && M.game.length > 0 && (
            <div className="card chart">
              <h4>Which AI plays best?{M.is_mock ? " (mock)" : ""}</h4>
              <p>Each forecasting model playing every puzzle. Mean Brier, lower is better.</p>
              <Bars cats={diffs} catNames={dn} series={[...new Set(M.game.map((g) => g.detector))].map((d) => ({ key: d, color: `var(--${d})`, name: DNAME[d] ?? d }))}
                get={(d, k) => M.game!.find((g) => g.difficulty === d && g.detector === k)?.brier} ymax={0.5} yfmt={(v) => v.toFixed(2)} tipFor={(d, k, v) => `${DNAME[k]} · ${dn[d]} · ${v.toFixed(3)}`} />
            </div>
          )}
          {M && M.error_by_age.length > 0 && (() => {
            const hz = Math.max(...M.error_by_age.map((e) => e.horizon_s));
            const ymax = Math.ceil(Math.max(...M.error_by_age.filter((e) => e.horizon_s === hz).map((e) => e.mean_err_m * M_TO_FT)) / 50) * 50;
            return (
              <div className="card chart">
                <h4>Why stale data is hard{M.is_mock ? " (mock)" : ""}</h4>
                <p>Error forecasting the aircraft {hz} s ahead, by age of the last position report. Feet.</p>
                <Bars cats={["0-1s", "1-3s", "3s+"]} catNames={{ "0-1s": "0–1 s", "1-3s": "1–3 s", "3s+": "3 s+" }} series={M.detectors.map((d) => ({ key: d, color: `var(--${d})`, name: DNAME[d] ?? d }))}
                  get={(b, d) => { const e = M.error_by_age.find((x) => x.detector === d && x.age_bucket === b && x.horizon_s === hz); return e ? e.mean_err_m * M_TO_FT : null; }}
                  ymax={ymax} yfmt={(v) => v.toFixed(0)} tipFor={(b, d, v) => `${DNAME[d]} · age ${b} · ${fmt(v)} ft`} />
              </div>
            );
          })()}
        </div>
      </div>
    </section>
  );
}
