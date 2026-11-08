"use client";

import { useState } from "react";
import type { GameData } from "@/lib/data";
import type { Mode } from "./App";

export default function ShiftPicker({ data, player, onStart, onBack }: { data: GameData; player: string; onStart: (m: Mode) => void; onBack: () => void }) {
  const [mode, setMode] = useState<Mode>("mixed");
  const count = (d: string) => data.index.puzzles.filter((p) => p.difficulty === d).length;
  const options = [
    { id: "mixed" as Mode, name: "Full shift", description: "Five rounds across clear, busy and fog conditions.", n: Math.min(5, data.index.puzzles.length) },
    ...data.index.difficulties.map((d) => ({ id: d.id as Mode, name: d.name, description: d.description, n: Math.min(5, count(d.id)) })),
  ].filter((o) => o.n > 0);

  return (
    <section className="view">
      <div className="center">
        <div className="shift">
          <div>
            <h2>{player}, choose your shift</h2>
            <p className="sub">Each round freezes when a vehicle is holding short of the runway.</p>
            <div className="opts">
              {options.map((o) => (
                <button key={o.id} className="opt" aria-pressed={mode === o.id} onClick={() => setMode(o.id)}>
                  <b>{o.name}</b><small>{o.description}</small><em>{o.n} rnd</em>
                </button>
              ))}
            </div>
            <div className="row" style={{ marginTop: 12 }}>
              <button className="btn primary" onClick={() => onStart(mode)}>Start shift</button>
              <button className="secondary" onClick={onBack}>Change name</button>
            </div>
          </div>
          <div className="brief">
            <h3>Position briefing</h3>
            <ol>
              <li>Watch the surface scope. Traffic freezes at the hold line.</li>
              <li>Find the <b>crossing vehicle</b> (yellow ring) and the <b>runway traffic</b> (red ring).</li>
              <li>Transmit <b>CLEAR</b> or <b>HOLD</b> and how sure you are.</li>
              <li>The AI gives its call. Then the recording plays out.</li>
            </ol>
            <p className="fine">Speeds are in knots (kt), the aviation unit: 1 kt is about 1.15 mph or 1.85 km/h. Distances are in feet. Press <b>K</b> during play for the full key.</p>
            <h3>Scoring (Brier)</h3>
            <div className="pts">
              <span className="mono">100</span><span>Right at 95%</span>
              <span className="mono">94</span><span>Right at 75%</span>
              <span className="mono">75</span><span>Any call at 50%</span>
              <span className="mono">44</span><span>Wrong at 75%</span>
              <span className="mono">10</span><span>Wrong at 95%</span>
            </div>
            <p className="fine">HOLD is right when the vehicle and the runway traffic came within the safe distance while the vehicle was in the crossing zone.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
