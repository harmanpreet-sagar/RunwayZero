"use client";

import { downloadRounds } from "@/lib/storage";
import type { Shift } from "./App";

export default function EndOfShift({ shift, onAgain, onStats }: { shift: Shift; onAgain: () => void; onStats: () => void }) {
  const rs = shift.rounds;
  const n = Math.max(1, rs.length);
  const you = rs.reduce((a, r) => a + r.human.points, 0);
  const ai = rs.reduce((a, r) => a + r.ai.points, 0);
  const right = rs.filter((r) => (r.human.call === "HOLD") === r.conflict).length;
  const avg = you / n;
  const rank = avg >= 92 ? "Tower supervisor" : avg >= 82 ? "Certified ground controller" : avg >= 68 ? "Developmental controller" : "Trainee";
  const wrong95 = rs.filter((r) => r.human.confidence === 0.95 && (r.human.call === "HOLD") !== r.conflict);
  const fog = rs.filter((r) => r.difficulty === "fog");
  const badges: [string, string, boolean][] = [
    ["Calibrated", "No wrong calls at 95%.", rs.some((r) => r.human.confidence === 0.95) && wrong95.length === 0],
    ["Low visibility", "Right in every fog round.", fog.length > 0 && fog.every((r) => (r.human.call === "HOLD") === r.conflict)],
    ["Beat the AI", "More points than the AI.", you > ai],
    ["Hedger", "Used 50% three or more times.", rs.filter((r) => r.human.confidence === 0.5).length >= 3],
  ];
  return (
    <section className="view">
      <div className="scroll">
        <div className="endtop">
          <div className="card">
            <h3>End of shift · {shift.player}</h3>
            <h2>{you > ai ? "You beat the AI" : you < ai ? "The AI wins this shift" : "Even shift"}</h2>
            <div className="final"><span className="y">{you}</span><span>vs</span><span className="a">{ai}</span></div>
            <div className="rank">
              <b>{rank}</b>
              <div className="note">Right call {right} of {rs.length}.{wrong95.length ? ` ${wrong95.length} wrong call${wrong95.length > 1 ? "s" : ""} at 95%, which cost the most.` : ""}</div>
            </div>
          </div>
          <div className="card">
            <h3>Qualifications</h3>
            <div className="badges">
              {badges.map(([t, d, on]) => <div key={t} className={`badge ${on ? "" : "off"}`}><b>{t}</b><span>{d}</span></div>)}
            </div>
          </div>
        </div>
        <div className="card tw">
          <table className="t">
            <thead><tr><th>Round</th><th>Outcome</th><th>You</th><th style={{ textAlign: "right" }}>Pts</th><th>AI</th><th style={{ textAlign: "right" }}>Pts</th></tr></thead>
            <tbody>
              {rs.map((r, i) => (
                <tr key={i}>
                  <td>{i + 1}. {shift.list[i]?.title}</td>
                  <td>{r.conflict ? "Loss of separation · HOLD right" : "Separation held · CLEAR right"}</td>
                  <td><span className={`tag ${r.human.call}`}>{r.human.call} {r.human.confidence * 100}%</span></td>
                  <td className="n">{r.human.points}</td>
                  <td><span className={`tag ${r.ai.call}`}>{r.ai.call} {r.ai.confidence * 100}%</span></td>
                  <td className="n">{r.ai.points}</td>
                </tr>
              ))}
              <tr><td colSpan={3}><b>Total</b></td><td className="n"><b>{you}</b></td><td /><td className="n"><b>{ai}</b></td></tr>
            </tbody>
          </table>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="btn primary" onClick={onAgain}>New shift</button>
          <button className="secondary" onClick={() => downloadRounds(rs, `runwayzero_${shift.player}_${shift.sessionId}.json`)}>Export rounds</button>
          <button className="secondary" onClick={onStats}>Compare with everyone</button>
        </div>
      </div>
    </section>
  );
}
