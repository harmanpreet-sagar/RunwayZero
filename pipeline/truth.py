"""Truth layer: turn the raw Amelia recording into clean tracks, a real airport map,
and the list of real runway crossings that the puzzle generator cuts puzzles from.

    python -m pipeline.truth                 # writes everything below
    python -m pipeline.truth --out some/dir  # put map.json somewhere else (default web/public/data)

Inputs (downloaded by pipeline/gate_check.py into data/amelia/):
    traj .../raw_trajectories/kbos/KBOS_*.csv   one hour of position reports, 1 per second
    graph .../kbos/semantic_graph.pkl           map line segments: runways, taxiways, hold lines
    assets/kbos/limits.json                      reference point + background-image bounds
    assets/kbos/bkg_map.png                      OpenStreetMap background image

Outputs:
    <out>/map.json                   contract file for the UI (runways, taxiways, crossing zones)
    <out>/kbos_bg.jpg                downscaled background image referenced by map.json
    data/derived/tracks.csv.gz       clean tracks, contract units (metres, m/s, x=east, y=north)
    data/derived/crossings.csv       every real vehicle runway crossing (input to the puzzles)
    data/derived/truth_summary.json  counts + per-runway activity (for the README / results)
    data/truth_check.png             picture to eyeball that everything lines up

Things about the raw data that are easy to get wrong (all checked against the data):
  * Amelia's x/y are in KILOMETRES and x points NORTH, y points EAST. Our contract is
    metres with x = east, y = north, so we swap and scale (see `to_contract_xy`).
  * Speed is in knots. Heading is degrees clockwise from north (same as the contract).
  * Altitude is 12.5 for everything on the ground, higher means airborne (feet).
  * The runway segments in the map continue far past the real runway ends (they include
    the extended approach centreline), so we trim each runway using its hold-short lines.
  * Runway 4R/22L is full of slow vehicles for the whole hour and has almost no fast
    aircraft, so it looks closed. We flag every runway as active / inactive from the
    traffic itself instead of assuming.
"""
import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline import safe_pickle
from pipeline.contracts import ROOT, SCHEMA_VERSION, WEB_DATA_DIR, write

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
AMELIA = os.path.join(ROOT, "data", "amelia", "data")
TRACKS_CSV = os.path.join(AMELIA, "traj_data_a10v08", "raw_trajectories", "kbos", "KBOS_538_1674511200.csv")
GRAPH_PKL = os.path.join(AMELIA, "graph_data_a10v01os", "kbos", "semantic_graph.pkl")
LIMITS_JSON = os.path.join(AMELIA, "assets", "kbos", "limits.json")
BKG_PNG = os.path.join(AMELIA, "assets", "kbos", "bkg_map.png")
DERIVED = os.path.join(ROOT, "data", "derived")

# --------------------------------------------------------------------------------------
# Units and thresholds (all in contract units: metres, seconds, m/s)
# --------------------------------------------------------------------------------------
KNOTS_TO_MS = 0.514444
GROUND_ALT_FT = 12.5               # Amelia reports exactly 12.5 for anything on the ground
FAST_ROLL_KT = 40                  # an aircraft on the ground faster than this is taking off / landing

RUNWAY_HALF_WIDTH_M = 23           # Logan runways are 150 ft (46 m) wide; used for drawing
CROSSING_HALF_WIDTH_M = 60         # "on the runway" for crossing purposes: runway + shoulders.
                                   # Hold-short lines sit ~80-90 m from the centreline, so a
                                   # vehicle past 60 m is definitely beyond the hold line.
HOLD_LINE_PERP_M = (50, 130)       # a hold segment belongs to a runway if its outer end is this far from the centreline...
HOLD_LINE_MIN_ANGLE_DEG = 45       # ...and it points at the runway (closer to perpendicular than parallel)
RUNWAY_END_MARGIN_M = 60           # runway ends sit a bit beyond the outermost hold lines
MAX_CROSSING_S = 120               # spending longer than this on the runway is driving ALONG it, not crossing
MAX_GAP_S = 3                      # tolerate this many missing seconds inside one crossing
ZONE_CLUSTER_GAP_M = 150           # crossings closer than this along a runway share one zone
ZONE_MIN_LENGTH_M = 120            # every zone is at least this long along the runway
ACTIVE_MIN_AIRCRAFT = 3            # a runway is "active" if at least this many aircraft rolled fast on it

# Logan runway designators. The map file has no names, so we match each fitted runway to
# this table by its true heading (low-numbered end) and, for parallel pairs, by which side
# it is on ("left" = left of a pilot flying the low-numbered heading). Published lengths
# are only used as a sanity check that our trimmed runway ends are reasonable.
KBOS_RUNWAYS = [
    # name,      true heading of the low end, side,   published length (m)
    ("4L/22R",   20,  "left",  2396),
    ("4R/22L",   20,  "right", 3050),
    ("9/27",     76,  None,    2134),
    ("14/32",    126, None,    1524),
    ("15L/33R",  135, "left",  779),
    ("15R/33L",  135, "right", 3073),
]


# --------------------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------------------
def to_contract_xy(amelia_x_km, amelia_y_km):
    """Amelia (x=north km, y=east km) -> contract (x=east m, y=north m)."""
    return np.asarray(amelia_y_km) * 1000.0, np.asarray(amelia_x_km) * 1000.0


def latlon_to_xy(lat, lon, ref_lat, ref_lon):
    """Lat/lon -> local metres around the reference point (flat-earth; < 10 m error over the airport).

    Only used for the background image bounds; tracks and map lines already come with
    Amelia's own x/y, and the two agree to within ~8 m (checked)."""
    r_earth = 6371000.0
    x = np.radians(np.asarray(lon) - ref_lon) * r_earth * np.cos(np.radians(ref_lat))
    y = np.radians(np.asarray(lat) - ref_lat) * r_earth
    return x, y


def heading_vec(deg):
    """Unit vector (east, north) for a compass heading in degrees (clockwise from north)."""
    r = np.radians(deg)
    return np.array([np.sin(r), np.cos(r)])


# --------------------------------------------------------------------------------------
# Runways
# --------------------------------------------------------------------------------------
@dataclass
class Runway:
    """A runway as a straight centreline: centre point + unit direction, trimmed to its real ends.

    Positions are described in runway coordinates:
      along = metres along the centreline from `center` (positive towards the high-numbered end's
              approach, i.e. in the direction of the low-numbered heading)
      perp  = signed metres sideways from the centreline (positive = left of that heading)
    """
    name: str
    center: np.ndarray            # (x, y) metres
    u: np.ndarray                 # unit vector along the runway (low-numbered heading)
    along_min: float = 0.0        # trimmed runway ends, in `along` metres
    along_max: float = 0.0
    fast_aircraft: int = 0        # aircraft seen rolling fast on it (takeoff / landing)
    vehicle_seconds: int = 0      # vehicle position reports on the runway surface
    active: bool = False
    notes: list = field(default_factory=list)

    @property
    def n(self):
        """Unit vector to the LEFT of the runway direction."""
        return np.array([-self.u[1], self.u[0]])

    @property
    def length(self):
        return self.along_max - self.along_min

    def to_runway_coords(self, x, y):
        """World (x, y) -> (along, perp) for this runway. Works on arrays."""
        dx, dy = np.asarray(x) - self.center[0], np.asarray(y) - self.center[1]
        return dx * self.u[0] + dy * self.u[1], dx * self.n[0] + dy * self.n[1]

    def to_world(self, along, perp):
        """(along, perp) -> world (x, y)."""
        p = self.center + np.multiply.outer(along, self.u) + np.multiply.outer(perp, self.n)
        return p[..., 0], p[..., 1]

    def rectangle(self, a0, a1, half_width):
        """Polygon (list of [x, y]) covering along a0..a1 and perp -half_width..+half_width."""
        corners = [(a0, -half_width), (a1, -half_width), (a1, half_width), (a0, half_width)]
        return [[round(float(v), 1) for v in self.to_world(a, p)] for a, p in corners]


def fit_line(points):
    """Best straight line through a cloud of points (principal direction). Returns (center, unit dir)."""
    center = points.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((points - center).T))
    return center, vecs[:, -1]          # eigenvector with the largest eigenvalue


def group_runway_segments(seg_a, seg_b):
    """Split all runway segments into one group per physical runway.

    Segments of the same runway share (almost) the same direction and the same sideways
    offset from the origin, so we bucket on (heading, offset) and merge nearby buckets."""
    d = seg_b - seg_a
    heading = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 180       # 0..180: direction, ignoring sense
    normal = np.c_[-np.cos(np.radians(heading)), np.sin(np.radians(heading))]
    offset = (seg_a * normal).sum(axis=1)

    groups = []                                                      # each: dict(heading, offset, idx)
    for i in np.argsort(heading):
        for g in groups:
            if abs(g["heading"] - heading[i]) < 5 and abs(g["offset"] - offset[i]) < 80:
                g["idx"].append(i)
                break
        else:
            groups.append({"heading": heading[i], "offset": offset[i], "idx": [i]})
    return [g["idx"] for g in groups if len(g["idx"]) >= 10]          # ignore stray fragments


def name_runways(lines):
    """Attach KBOS designators to fitted (center, u) lines. Returns a list of Runway."""
    runways = []
    for center, u in lines:
        hdg = np.degrees(np.arctan2(u[0], u[1])) % 360
        # candidates with a matching heading (either end of the runway)
        cands = [r for r in KBOS_RUNWAYS
                 if min(abs((hdg - r[1] + 180) % 360 - 180), abs((hdg - r[1]) % 360 - 180)) < 6]
        if not cands:
            raise ValueError(f"No KBOS runway matches a line with heading {hdg:.1f}")
        # point u along the low-numbered heading so 'left' means the same thing for every runway
        if np.dot(u, heading_vec(cands[0][1])) < 0:
            u = -u
        runways.append([cands, Runway(name="?", center=center, u=u)])

    # parallel pairs: the runway further to the LEFT of the shared heading gets the 'L' name
    for cands, rw in runways:
        if len(cands) == 1:
            rw.name = cands[0][0]
            continue
        left = heading_vec(cands[0][1] - 90)
        partner = [o for c, o in runways if c == cands and o is not rw]
        is_left = all(np.dot(rw.center, left) > np.dot(o.center, left) for o in partner)
        rw.name = next(c[0] for c in cands if c[2] == ("left" if is_left else "right"))
    names = [r.name for _, r in runways]
    if len(set(names)) != len(names):
        raise ValueError(f"Runway naming clash: {names}")
    return [r for _, r in runways]


def trim_runway(rw, hold_a, hold_b, centre_pts):
    """Find the real runway ends.

    Every taxiway that enters a runway has a hold-short position ~80-90 m from the runway
    centreline. In Amelia's map these are short "hold" segments drawn ALONG the taxiway,
    from the hold line towards the runway (checked: they are ~perpendicular to the runway,
    median 33 m long). The outermost ones along a runway mark (almost) its ends; runway
    segments beyond them are the extended approach centreline."""
    a_a, p_a = rw.to_runway_coords(hold_a[:, 0], hold_a[:, 1])
    a_b, p_b = rw.to_runway_coords(hold_b[:, 0], hold_b[:, 1])
    far_perp = np.maximum(np.abs(p_a), np.abs(p_b))               # the hold-line end of the segment
    near_perp = np.minimum(np.abs(p_a), np.abs(p_b))              # the end pointing at the runway
    d = hold_b - hold_a
    angle = np.degrees(np.abs(np.arctan2(d[:, 0] * rw.u[1] - d[:, 1] * rw.u[0], d @ rw.u)))
    angle = np.minimum(angle, 180 - angle)                        # 0 = parallel, 90 = perpendicular
    along = (a_a + a_b) / 2
    mine = (far_perp >= HOLD_LINE_PERP_M[0]) & (far_perp <= HOLD_LINE_PERP_M[1]) \
        & (near_perp < far_perp - 5) & (angle >= HOLD_LINE_MIN_ANGLE_DEG)

    # never extend past the drawn runway line itself
    a_line, _ = rw.to_runway_coords(centre_pts[:, 0], centre_pts[:, 1])
    if mine.sum() >= 2:
        rw.along_min = max(along[mine].min() - RUNWAY_END_MARGIN_M, a_line.min())
        rw.along_max = min(along[mine].max() + RUNWAY_END_MARGIN_M, a_line.max())
    else:   # no hold lines found: fall back to the whole drawn line and say so
        rw.along_min, rw.along_max = a_line.min(), a_line.max()
        rw.notes.append("no hold lines found; using the full drawn line")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_reference():
    """Reference point and background-image bounds from limits.json."""
    with open(LIMITS_JSON) as fh:
        lim = json.load(fh)
    return lim


def load_tracks(path=TRACKS_CSV):
    """Raw Amelia CSV -> clean DataFrame in contract units, sorted by (id, t).

    Columns: t (s), id (str), kind (aircraft/vehicle/unknown), x, y (m), speed (m/s),
             heading (deg), alt_ft, on_ground (bool)
    Airborne points are KEPT (flagged on_ground=False): an aircraft on short final is
    exactly what a crossing vehicle has to worry about, so the puzzles need it."""
    raw = pd.read_csv(path)
    x, y = to_contract_xy(raw["x"], raw["y"])
    df = pd.DataFrame({
        "t": raw["Frame"].astype(float),                   # Frame = seconds from the start of the hour
        "id": raw["ID"].astype(str),
        "kind": raw["Type"].map({0.0: "aircraft", 1.0: "vehicle", 2.0: "unknown"}).fillna("unknown"),
        "x": x.round(2),
        "y": y.round(2),
        "speed": (raw["Speed"] * KNOTS_TO_MS).round(2),
        "heading": raw["Heading"].round(1),
        "alt_ft": raw["Altitude"],
        "on_ground": raw["Altitude"] <= GROUND_ALT_FT,
    })
    # The same agent can appear twice in one frame (rare interpolation artefacts): keep the first.
    df = df.drop_duplicates(["id", "t"]).sort_values(["id", "t"]).reset_index(drop=True)
    return df


def load_map_segments():
    """Map line segments by type, as (start_xy, end_xy) arrays in contract metres.

    Each row of all_polylines is one segment:
      [lat0, lon0, x0_km, y0_km, lat1, lon1, x1_km, y1_km, type, index]
    with type 1 = hold-short line, 3 = runway centreline, 4 = taxiway centreline."""
    graph = safe_pickle.load(GRAPH_PKL)
    poly = graph["map_infos"]["all_polylines"].astype(float)
    out = {}
    for name, code in (("hold", 1), ("runway", 3), ("taxiway", 4)):
        rows = poly[poly[:, 8] == code]
        a = np.column_stack(to_contract_xy(rows[:, 2], rows[:, 3]))     # segment starts, shape (N, 2)
        b = np.column_stack(to_contract_xy(rows[:, 6], rows[:, 7]))     # segment ends
        out[name] = (a, b)
    return out


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
def build_runways(segs, tracks):
    """Fit, name, trim and classify (active / inactive) every runway."""
    run_a, run_b = segs["runway"]
    lines = []
    for idx in group_runway_segments(run_a, run_b):
        pts = np.r_[run_a[idx], run_b[idx]]
        lines.append(fit_line(pts) + (pts,))
    runways = name_runways([(c, u) for c, u, _ in lines])
    for rw, (_, _, pts) in zip(runways, lines):
        trim_runway(rw, *segs["hold"], pts)

    # Activity, measured from the traffic itself.
    fast = tracks[(tracks.kind == "aircraft") & tracks.on_ground & (tracks.speed > FAST_ROLL_KT * KNOTS_TO_MS)]
    veh = tracks[tracks.kind == "vehicle"]
    for rw in runways:
        a, p = rw.to_runway_coords(fast.x.values, fast.y.values)
        # Only count aircraft moving ALONG this runway (either direction). Without this, a jet
        # rolling down 15R/33L would also count for every runway it passes through at an
        # intersection.
        rw_hdg = np.degrees(np.arctan2(rw.u[0], rw.u[1]))
        off = np.abs((fast.heading.values - rw_hdg + 90) % 180 - 90)          # 0 = aligned either way
        on = (np.abs(p) < 40) & (a >= rw.along_min) & (a <= rw.along_max) & (off < 20)
        rw.fast_aircraft = int(fast[on].id.nunique())
        a, p = rw.to_runway_coords(veh.x.values, veh.y.values)
        rw.vehicle_seconds = int(((np.abs(p) < RUNWAY_HALF_WIDTH_M) & (a >= rw.along_min) & (a <= rw.along_max)).sum())
        rw.active = rw.fast_aircraft >= ACTIVE_MIN_AIRCRAFT
        if not rw.active and rw.vehicle_seconds > 3600:
            rw.notes.append("vehicles on it for long periods and no fast traffic: probably closed this hour")

        published = next(r[3] for r in KBOS_RUNWAYS if r[0] == rw.name)
        if abs(rw.length - published) > 0.35 * published:
            rw.notes.append(f"trimmed length {rw.length:.0f} m vs published {published} m: check the plot")
    return sorted(runways, key=lambda r: r.name)


def chain_segments(seg_a, seg_b, snap_m=0.5):
    """Join 2-point taxiway segments into longer polylines (smaller map.json, smoother drawing).

    Endpoints closer than `snap_m` are treated as the same node. We walk every chain of
    nodes that have exactly two neighbours; junctions (3+ neighbours) end a chain."""
    key = lambda p: (round(p[0] / snap_m), round(p[1] / snap_m))
    coords, nbrs = {}, {}
    for a, b in zip(seg_a, seg_b):
        ka, kb = key(a), key(b)
        if ka == kb:
            continue
        coords[ka], coords[kb] = a, b
        nbrs.setdefault(ka, set()).add(kb)
        nbrs.setdefault(kb, set()).add(ka)

    used, chains = set(), []

    def walk(start, nxt):
        chain = [start, nxt]
        used.add(frozenset((start, nxt)))
        while len(nbrs[chain[-1]]) == 2:
            step = next(n for n in nbrs[chain[-1]] if n != chain[-2])
            edge = frozenset((chain[-1], step))
            if edge in used:
                break
            used.add(edge)
            chain.append(step)
        return chain

    # start from junctions / dead ends first, then sweep up closed loops
    starts = [k for k, v in nbrs.items() if len(v) != 2] + list(nbrs)
    for s in starts:
        for n in nbrs[s]:
            if frozenset((s, n)) not in used:
                chains.append(walk(s, n))
    return [[[round(float(coords[k][0]), 1), round(float(coords[k][1]), 1)] for k in c] for c in chains]


def find_crossings(tracks, runways):
    """Every time a vehicle drove ACROSS a runway (in one side, out the other, quickly).

    For each vehicle and runway, mark the seconds it was on the runway surface (within
    CROSSING_HALF_WIDTH_M of the centreline and between the runway ends). Each unbroken
    stretch of those seconds is a candidate; it's a crossing if the vehicle entered on one
    side and left on the other, and the stretch was short (driving along the runway, e.g.
    maintenance on a closed runway, is not a crossing).

    Returns one row per crossing: vehicle id, runway, t_enter, t_exit, the along-runway
    position where it crossed, and which side it came from."""
    rows = []
    vehicles = tracks[tracks.kind == "vehicle"]
    for vid, trk in vehicles.groupby("id"):
        t = trk.t.values
        for rw in runways:
            along, perp = rw.to_runway_coords(trk.x.values, trk.y.values)
            on = (np.abs(perp) <= CROSSING_HALF_WIDTH_M) & (along >= rw.along_min) & (along <= rw.along_max)
            if not on.any():
                continue
            idx = np.flatnonzero(on)
            # split the "on runway" seconds into separate visits wherever there's a gap
            breaks = np.flatnonzero(np.diff(t[idx]) > MAX_GAP_S) + 1
            for visit in np.split(idx, breaks):
                first, last = visit[0], visit[-1]
                if first == 0 or last == len(t) - 1:
                    continue                            # track starts/ends on the runway: can't tell
                side_in, side_out = np.sign(perp[first - 1]), np.sign(perp[last + 1])
                duration = t[last] - t[first]
                if side_in == side_out or side_in == 0 or duration > MAX_CROSSING_S:
                    continue                            # came back out the same side, or drove along it
                rows.append({
                    "vehicle_id": vid, "runway": rw.name,
                    "t_enter": float(t[first]), "t_exit": float(t[last]),
                    "duration_s": float(duration),
                    "along_m": round(float(np.median(along[visit])), 1),
                    "from_side": "left" if side_in > 0 else "right",
                })
    cols = ["vehicle_id", "runway", "t_enter", "t_exit", "duration_s", "along_m", "from_side"]
    return pd.DataFrame(rows, columns=cols).sort_values(["runway", "t_enter"]).reset_index(drop=True)


def build_zones(crossings, runways):
    """One protected zone per place where vehicles actually cross a runway.

    Crossings on the same runway within ZONE_CLUSTER_GAP_M of each other (along the
    runway) share a zone. Adds a zone_id column to `crossings` and returns the zone list
    in contract format: {id, name, polygon}."""
    by_name = {r.name: r for r in runways}
    zones, zone_of = [], pd.Series(index=crossings.index, dtype=object)
    for rname, grp in crossings.groupby("runway"):
        rw = by_name[rname]
        grp = grp.sort_values("along_m")
        cluster_id = (grp.along_m.diff() > ZONE_CLUSTER_GAP_M).cumsum()
        for k, (_, cl) in enumerate(grp.groupby(cluster_id), start=1):
            a0, a1 = cl.along_m.min() - 40, cl.along_m.max() + 40
            if a1 - a0 < ZONE_MIN_LENGTH_M:                 # widen tiny clusters to a sensible size
                mid = (a0 + a1) / 2
                a0, a1 = mid - ZONE_MIN_LENGTH_M / 2, mid + ZONE_MIN_LENGTH_M / 2
            zid = f"Z_{rname.replace('/', '_')}_{k}"
            zones.append({"id": zid, "name": f"RWY {rname} crossing {k}",
                          "polygon": rw.rectangle(a0, a1, CROSSING_HALF_WIDTH_M)})
            zone_of[cl.index] = zid
    crossings = crossings.assign(zone_id=zone_of)
    return zones, crossings


def make_background(lim, out_dir, width_px=2400):
    """Downscale the 9k-pixel OpenStreetMap background to a web-friendly JPEG.

    Returns the map.json `background` entry (image path relative to /data/ + bounds in metres),
    or None if Pillow or the image is missing (the map still works without it)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if not os.path.exists(BKG_PNG):
        return None
    Image.MAX_IMAGE_PIXELS = None                      # the source image is large but trusted-size
    img = Image.open(BKG_PNG).convert("RGB")
    img = img.resize((width_px, round(img.height * width_px / img.width)), Image.LANCZOS)
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, "kbos_bg.jpg"), quality=80, optimize=True)

    b = lim["espg_4326"]                                # image edges in lat/lon
    xmin, ymin = latlon_to_xy(b["south"], b["west"], lim["ref_lat"], lim["ref_lon"])
    xmax, ymax = latlon_to_xy(b["north"], b["east"], lim["ref_lat"], lim["ref_lon"])
    bounds = {k: round(float(v), 1) for k, v in (("xmin", xmin), ("xmax", xmax), ("ymin", ymin), ("ymax", ymax))}
    return {"image": "kbos_bg.jpg", "bounds": bounds}


def build_map(lim, runways, taxiways, zones, background):
    """Assemble map.json (contract v1.1)."""
    airport_map = {
        "schema_version": SCHEMA_VERSION,
        "airport": "KBOS",
        "units": "m",
        "origin": {"lat": lim["ref_lat"], "lon": lim["ref_lon"]},
        # view = the background image's extent (covers the whole airfield)
        "bounds": background["bounds"] if background else {"xmin": -3300, "xmax": 3300, "ymin": -3100, "ymax": 3100},
        "runways": [{"id": r.name, "polygon": r.rectangle(r.along_min, r.along_max, RUNWAY_HALF_WIDTH_M)} for r in runways],
        "taxiways": [{"id": f"T{i:04d}", "polyline": pl} for i, pl in enumerate(taxiways)],
        "zones": zones,
    }
    if background:
        airport_map["background"] = background
    return airport_map


def plot_check(tracks, runways, zones, crossings, background, out_png, bg_dir):
    """Picture for a human to confirm: runways sit on the background's runways, zones sit
    where red vehicle tracks cross them, and crossings (black x) are inside zones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 13))
    if background:
        img = plt.imread(os.path.join(bg_dir, background["image"]))
        b = background["bounds"]
        ax.imshow(img, extent=(b["xmin"], b["xmax"], b["ymin"], b["ymax"]), alpha=0.55)
    veh = tracks[tracks.kind == "vehicle"]
    ac = tracks[(tracks.kind == "aircraft") & tracks.on_ground]
    ax.scatter(ac.x, ac.y, s=0.2, c="tab:blue", label="aircraft (ground)")
    ax.scatter(veh.x, veh.y, s=0.2, c="tab:red", label="vehicles")
    for r in runways:
        poly = np.array(r.rectangle(r.along_min, r.along_max, RUNWAY_HALF_WIDTH_M) + [r.rectangle(r.along_min, r.along_max, RUNWAY_HALF_WIDTH_M)[0]])
        ax.plot(poly[:, 0], poly[:, 1], c="green" if r.active else "gray", lw=1.5)
        cx, cy = r.center
        ax.text(cx, cy, f"{r.name}\n{'ACTIVE' if r.active else 'inactive'}", fontsize=9, weight="bold",
                color="green" if r.active else "dimgray", ha="center")
    for z in zones:
        p = np.array(z["polygon"] + [z["polygon"][0]])
        ax.fill(p[:, 0], p[:, 1], color="gold", alpha=0.5)
    by_name = {r.name: r for r in runways}
    for _, c in crossings.iterrows():
        x, y = by_name[c.runway].to_world(c.along_m, 0.0)
        ax.plot(x, y, "kx", ms=6)
    ax.set_aspect("equal")
    if background:
        b = background["bounds"]
        ax.set_xlim(b["xmin"], b["xmax"])
        ax.set_ylim(b["ymin"], b["ymax"])
    ax.legend(markerscale=20, loc="lower left")
    ax.set_title(f"KBOS truth layer: {len(zones)} crossing zones, {len(crossings)} vehicle crossings "
                 "(green = active runway, gold = zone, x = crossing)")
    fig.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(out_dir=WEB_DATA_DIR):
    lim = load_reference()
    tracks = load_tracks()
    segs = load_map_segments()

    runways = build_runways(segs, tracks)
    taxiways = chain_segments(*segs["taxiway"])
    crossings = find_crossings(tracks, runways)
    zones, crossings = build_zones(crossings, runways)
    background = make_background(lim, out_dir)

    # map.json goes through the contract validator; the derived files are pipeline-internal.
    print("wrote", write(build_map(lim, runways, taxiways, zones, background), "map.json", out_dir))

    os.makedirs(DERIVED, exist_ok=True)
    tracks.to_csv(os.path.join(DERIVED, "tracks.csv.gz"), index=False)
    crossings.to_csv(os.path.join(DERIVED, "crossings.csv"), index=False)

    active = {r.name for r in runways if r.active}
    summary = {
        "recording": os.path.splitext(os.path.basename(TRACKS_CSV))[0],
        "seconds": int(tracks.t.max() - tracks.t.min() + 1),
        "position_reports": int(len(tracks)),
        "agents": {k: int(v) for k, v in tracks.groupby("kind").id.nunique().items()},
        "runways": [{"name": r.name, "length_m": round(r.length), "active": r.active,
                     "fast_aircraft": r.fast_aircraft, "vehicle_seconds_on_runway": r.vehicle_seconds,
                     "notes": r.notes} for r in runways],
        "taxiway_polylines": len(taxiways),
        "zones": len(zones),
        "crossings": int(len(crossings)),
        "crossings_on_active_runways": int(crossings.runway.isin(active).sum()),
        "crossings_by_runway": {k: int(v) for k, v in crossings.runway.value_counts().items()},
    }
    with open(os.path.join(DERIVED, "truth_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    plot_check(tracks, runways, zones, crossings, background, os.path.join(ROOT, "data", "truth_check.png"), out_dir)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=WEB_DATA_DIR, help="where map.json + kbos_bg.jpg go (default web/public/data)")
    main(ap.parse_args().out)
