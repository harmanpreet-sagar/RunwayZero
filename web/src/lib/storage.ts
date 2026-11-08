// Per-device storage. Every access is guarded: private windows and blocked storage just fall back to memory.
import type { Round } from "@contracts/types";

const ROUNDS = "rz_rounds_v1";
let memory: Round[] = [];

function get(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function set(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: keep going in memory */
  }
}

export function loadRounds(): Round[] {
  const raw = get(ROUNDS);
  if (raw == null) return memory;
  try {
    return JSON.parse(raw) as Round[];
  } catch {
    return memory;
  }
}

export function saveRound(r: Round) {
  memory = [...memory, r];
  let stored: Round[] = [];
  const raw = get(ROUNDS);
  if (raw) {
    try {
      stored = JSON.parse(raw) as Round[];
    } catch {
      stored = [];
    }
  }
  set(ROUNDS, JSON.stringify([...stored, r]));
}

export const getPlayer = () => get("rz_nick") ?? "";
export const setPlayer = (n: string) => set("rz_nick", n);
export const getTheme = () => get("rz_theme") as "light" | "dark" | null;
export const setTheme = (t: "light" | "dark") => set("rz_theme", t);

/** Download rounds as a JSON file (for the playtest). */
export function downloadRounds(rounds: Round[], name = "runwayzero_rounds.json") {
  const url = URL.createObjectURL(new Blob([JSON.stringify(rounds, null, 2)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
