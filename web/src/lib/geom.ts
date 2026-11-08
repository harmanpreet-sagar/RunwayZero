// Small 2D helpers. World frame = the contract's: metres, x = east, y = north.
// SVG's y axis points down, so anything drawn uses -y.

export type Vec = [number, number];

export const sub = (a: Vec, b: Vec): Vec => [a[0] - b[0], a[1] - b[1]];
export const add = (a: Vec, b: Vec): Vec => [a[0] + b[0], a[1] + b[1]];
export const mul = (a: Vec, s: number): Vec => [a[0] * s, a[1] * s];
export const dot = (a: Vec, b: Vec) => a[0] * b[0] + a[1] * b[1];
export const len = (a: Vec) => Math.hypot(a[0], a[1]);
export const unit = (a: Vec): Vec => {
  const l = len(a) || 1;
  return [a[0] / l, a[1] / l];
};
export const lerp = (a: number, b: number, u: number) => a + (b - a) * u;

export const centroid = (poly: Vec[]): Vec => [
  poly.reduce((s, p) => s + p[0], 0) / poly.length,
  poly.reduce((s, p) => s + p[1], 0) / poly.length,
];

export function inPoly(pt: Vec, poly: Vec[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Degrees clockwise from north for a direction vector. */
export const compass = (u: Vec) => ((Math.atan2(u[0], u[1]) * 180) / Math.PI + 360) % 360;
/** Unit vector for a heading in degrees clockwise from north. */
export const headingVec = (deg: number): Vec => [Math.sin((deg * Math.PI) / 180), Math.cos((deg * Math.PI) / 180)];
/** Smallest angle between two headings, 0..180. */
export const angleDiff = (a: number, b: number) => Math.abs(((a - b + 540) % 360) - 180);

/** A rectangle-ish runway polygon as centre, long axis u, short axis v, length and width. */
export interface RunwayFrame { c: Vec; u: Vec; v: Vec; L: number; W: number }

export function runwayFrame(poly: Vec[]): RunwayFrame {
  let best = { L: -1, u: [1, 0] as Vec };
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const L = len(sub(b, a));
    if (L > best.L) best = { L, u: unit(sub(b, a)) };
  }
  const u = best.u;
  const v: Vec = [-u[1], u[0]];
  const c = centroid(poly);
  const pu = poly.map((p) => dot(sub(p, c), u));
  const pv = poly.map((p) => dot(sub(p, c), v));
  return { c, u, v, L: Math.max(...pu) - Math.min(...pu), W: Math.max(...pv) - Math.min(...pv) };
}
