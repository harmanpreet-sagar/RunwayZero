"use client";

import type { Difficulty, Round } from "@contracts/types";
import { useCallback, useEffect, useState } from "react";
import { type GameData, loadGameData } from "@/lib/data";
import { getPlayer, getTheme, setPlayer as storePlayer, setTheme } from "@/lib/storage";
import EndOfShift from "./EndOfShift";
import Game from "./Game";
import ShiftPicker from "./ShiftPicker";
import SignIn from "./SignIn";
import Stats from "./Stats";

export type Mode = "mixed" | Difficulty;
type View = "signin" | "shift" | "game" | "end" | "stats";
export type Entry = GameData["index"]["puzzles"][number];

export interface Shift {
  player: string;
  sessionId: string;
  list: Entry[];
  rounds: Round[];
}

function shuffle<T>(a: T[]): T[] {
  const b = a.slice();
  for (let i = b.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [b[i], b[j]] = [b[j], b[i]];
  }
  return b;
}

/** Five rounds: a mix of difficulties for a full shift, or up to five of one kind. */
function pickPuzzles(all: Entry[], mode: Mode): Entry[] {
  if (mode !== "mixed") return shuffle(all.filter((p) => p.difficulty === mode)).slice(0, 5);
  const used = new Set<string>();
  const out: Entry[] = [];
  for (const d of ["clear", "busy", "fog", "busy", "fog"] as Difficulty[]) {
    const p = shuffle(all.filter((x) => x.difficulty === d && !used.has(x.id)))[0] ?? shuffle(all.filter((x) => !used.has(x.id)))[0];
    if (p) {
      used.add(p.id);
      out.push(p);
    }
  }
  return out;
}

export default function App() {
  const [data, setData] = useState<GameData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("signin");
  const [player, setPlayerState] = useState("");
  const [shift, setShift] = useState<Shift | null>(null);
  const [theme, setThemeState] = useState<"light" | "dark">("dark");
  const [keyOpen, setKeyOpen] = useState(false);
  const [roundIdx, setRoundIdx] = useState(0);

  useEffect(() => {
    loadGameData().then(setData, (e: Error) => setError(e.message));
    // Read browser-only state once on mount (it isn't available during server rendering).
    /* eslint-disable react-hooks/set-state-in-effect */
    setPlayerState(getPlayer());
    const saved = getTheme();
    setThemeState(saved ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    setTheme(next);
    setThemeState(next);
  };

  const start = (mode: Mode) => {
    if (!data) return;
    setShift({ player, sessionId: `s-${Date.now().toString(36)}`, list: pickPuzzles(data.index.puzzles, mode), rounds: [] });
    setRoundIdx(0);
    setView("game");
  };

  const onRound = useCallback((r: Round) => setShift((s) => (s ? { ...s, rounds: [...s.rounds, r] } : s)), []);

  const you = shift?.rounds.reduce((a, r) => a + r.human.points, 0) ?? 0;
  const ai = shift?.rounds.reduce((a, r) => a + r.ai.points, 0) ?? 0;
  const inShift = view === "game" || view === "end";
  const badge = data?.isMock ? "MOCK DATA" : view === "stats" && data?.metrics?.is_mock ? "MOCK METRICS" : null;

  return (
    <div className="app">
      <header className="bar">
        <div className="wm">RUNWAY <span>ZERO</span></div>
        {inShift && shift && (
          <div className="score">
            <div><i style={{ background: "var(--you)" }} /><span>{shift.player}</span><b>{you}</b></div>
            <div><i style={{ background: "var(--ai)" }} /><span>AI</span><b>{ai}</b></div>
            <div>RND {view === "game" ? roundIdx + 1 : shift.list.length}/{shift.list.length}</div>
          </div>
        )}
        <div className="tools">
          {badge && <span className="mock">{badge}</span>}
          <button className="tbtn" aria-current={view !== "stats" ? "page" : undefined} onClick={() => setView(shift && view === "stats" ? (shift.rounds.length >= shift.list.length ? "end" : "game") : player ? "shift" : "signin")}>Play</button>
          <button className="tbtn" aria-current={view === "stats" ? "page" : undefined} onClick={() => setView("stats")}>Stats</button>
          <button className="tbtn" title="What the symbols and units mean (K)" onClick={() => setKeyOpen((k) => !k)} disabled={view !== "game"}>Key</button>
          <button className="tbtn" onClick={toggleTheme}>{theme === "dark" ? "Light mode" : "Dark mode"}</button>
        </div>
      </header>
      <main>
        {error && <div className="error"><b>Data not found.</b><p>{error}</p></div>}
        {!error && !data && <div className="loading">Loading Boston Logan…</div>}
        {data && view === "signin" && (
          <SignIn data={data} player={player} onNext={(n) => { setPlayerState(n); storePlayer(n); setView("shift"); }} />
        )}
        {data && view === "shift" && <ShiftPicker data={data} player={player} onStart={start} onBack={() => setView("signin")} />}
        {data && view === "game" && shift && (
          <Game data={data} shift={shift} keyOpen={keyOpen} setKeyOpen={setKeyOpen} onRound={onRound} onIndex={setRoundIdx} onEnd={() => setView("end")} />
        )}
        {data && view === "end" && shift && (
          <EndOfShift shift={shift} onAgain={() => { setShift(null); setView("shift"); }} onStats={() => setView("stats")} />
        )}
        {data && view === "stats" && <Stats data={data} />}
      </main>
    </div>
  );
}
