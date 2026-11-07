"""Data gate: download one busy hour + map assets for one airport from Amelia-10,
check that ground vehicles are present, and save a plot.

Usage (from the repo root):
    pip install -r requirements.txt
    python pipeline/gate_check.py            # defaults to KBOS
    python pipeline/gate_check.py kjfk       # another Amelia-10 airport
Outputs: data/gate_check.png, data/gate_summary.json, raw files under data/amelia/
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

# Anaconda sets SSL_CERT_FILE to its own cacert.pem; if that file is missing every
# HTTPS request crashes with FileNotFoundError. Drop it so the OS cert store is used.
for _var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    if os.environ.get(_var) and not os.path.isfile(os.environ[_var]):
        del os.environ[_var]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "AmeliaCMU/Amelia-10"
AIRPORT = (sys.argv[1] if len(sys.argv) > 1 else "kbos").lower()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "amelia")
MAX_ASSET_BYTES = 200 * 1024 * 1024  # skip anything huge

api = HfApi()
print(f"Listing {REPO} ... (takes ~30s)")
files = [f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
         if getattr(f, "size", None) is not None]
print("Top-level folders:", sorted({f.path.split("/")[0] for f in files}))
ap = [f for f in files if AIRPORT in f.path.lower()]
csvs = [f for f in ap if f.path.endswith(".csv")]
assets = [f for f in ap if not f.path.endswith(".csv")]
print(f"{len(csvs)} hourly CSVs and {len(assets)} other files for {AIRPORT.upper()}")
for f in assets:
    print(f"  asset: {f.path}  ({f.size / 1e6:.1f} MB)")
if not csvs:
    sys.exit("No trajectory CSVs found for this airport -> gate FAILS for this airport.")


def utc_hour(path):
    m = re.search(r"_(\d{9,11})\.csv$", path)
    return datetime.fromtimestamp(int(m.group(1)), tz=timezone.utc).hour if m else None


daytime = [f for f in csvs if utc_hour(f.path) is not None and 14 <= utc_hour(f.path) <= 22]
pick = max(daytime or csvs, key=lambda f: f.size)
print(f"Busiest daytime hour: {pick.path} ({pick.size / 1e6:.1f} MB)")
csv_path = hf_hub_download(REPO, pick.path, repo_type="dataset", local_dir=OUT)

got_assets = []
for f in assets:
    if f.size <= MAX_ASSET_BYTES:
        got_assets.append(hf_hub_download(REPO, f.path, repo_type="dataset", local_dir=OUT))

df = pd.read_csv(csv_path)
print("Columns:", list(df.columns))
print("Rows:", len(df))
df["x_m"], df["y_m"] = df["x"] * 1000, df["y"] * 1000  # x/y are km
types = df.groupby("Type")["ID"].nunique().to_dict()  # 0 aircraft, 1 vehicle, 2 unknown
print("Unique agents by Type (0=aircraft, 1=vehicle, 2=unknown):", types)

ground = df[(df["Altitude"] <= df["Altitude"].min() + 150) & (df["Speed"] < 60)]
gv = ground[ground["Type"] == 1]["ID"].nunique()
ga = ground[ground["Type"] == 0]["ID"].nunique()
print(f"On the ground: {ga} aircraft, {gv} vehicles")

fig, ax = plt.subplots(figsize=(11, 11))
for t, c, lbl in [(0, "tab:blue", "aircraft"), (1, "tab:red", "vehicle"), (2, "tab:gray", "unknown")]:
    g = df[df["Type"] == t]
    ax.scatter(g["x_m"], g["y_m"], s=0.3, c=c, label=f"{lbl} ({g['ID'].nunique()})")
ax.set_aspect("equal")
ax.set_xlim(df["x_m"].quantile(0.01), df["x_m"].quantile(0.99))
ax.set_ylim(df["y_m"].quantile(0.01), df["y_m"].quantile(0.99))
ax.legend(markerscale=20)
ax.set_title(f"{AIRPORT.upper()} - {os.path.basename(pick.path)}")
png = os.path.join(ROOT, "data", "gate_check.png")
fig.savefig(png, dpi=120, bbox_inches="tight")

summary = {
    "airport": AIRPORT, "csv": pick.path, "rows": len(df), "columns": list(df.columns),
    "agents_by_type": {str(k): int(v) for k, v in types.items()},
    "ground_aircraft": int(ga), "ground_vehicles": int(gv),
    "assets_found": [f.path for f in assets], "assets_downloaded": got_assets,
}
with open(os.path.join(ROOT, "data", "gate_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

ok = gv >= 3 and ga >= 10
print("\nGATE:", "PASS" if ok else "CHECK MANUALLY (few ground vehicles/aircraft)")
print("Has map assets:", bool(assets), "| plot saved to", png)
