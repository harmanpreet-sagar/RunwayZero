// Aviation units. The data contract is metric (metres, m/s); the screen uses what
// controllers use: knots for speed and feet for distance, with metric alongside.

export const M_S_TO_KT = 1.94384; // 1 m/s in knots
export const M_TO_FT = 3.28084;
export const KT_TO_MPH = 1.15078;
export const FT_PER_NM = 6076.12;

export const fmt = (v: number | null | undefined, digits = 0) =>
  v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(digits);

export const kt = (ms: number | null | undefined) => Math.round((ms ?? 0) * M_S_TO_KT);

/** Feet (rounded to 10) below half a nautical mile, NM above. */
export function distanceText(m: number): { main: string; alt: string } {
  const ft = m * M_TO_FT;
  if (ft >= FT_PER_NM / 2) return { main: `${fmt(ft / FT_PER_NM, 1)} NM`, alt: `${fmt(m / 1000, 1)} km` };
  return { main: `${fmt(Math.round(ft / 10) * 10)} ft`, alt: `${fmt(m)} m` };
}

/** The pipeline writes AI reasoning in m/s; show knots like the rest of the screen. */
export const speedsToKnots = (text: string) =>
  text.replace(/(\d+(?:\.\d+)?)\s*m\/s/g, (_, v: string) => `${Math.round(Number(v) * M_S_TO_KT)} kt`);
