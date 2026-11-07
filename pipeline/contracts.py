"""Python side of the RunwayZero data contract.

The JSON schemas in contracts/schema/ are the source of truth. Every file the
pipeline writes for the UI goes through `write()`, which validates it first, so
a broken export fails here instead of in the browser.
"""
import json
import os
from functools import lru_cache

from jsonschema import Draft202012Validator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCHEMA_DIR = os.path.join(ROOT, "contracts", "schema")
WEB_DATA_DIR = os.path.join(ROOT, "web", "public", "data")
SCHEMA_VERSION = "1.0"

DETECTORS = ("cv", "kf", "ml")
SCENARIOS = ("normal", "cleared_crossing", "tracking_dropout", "delayed_data")
AGE_BUCKETS = ("0-1s", "1-3s", "3s+")


@lru_cache(maxsize=None)
def _validator(kind):
    with open(os.path.join(SCHEMA_DIR, f"{kind}.schema.json")) as fh:
        return Draft202012Validator(json.load(fh))


def kind_for(filename):
    name = os.path.basename(filename)
    if name.startswith("replay_"):
        return "replay"
    stem = os.path.splitext(name)[0]
    if stem in ("map", "scenarios", "metrics"):
        return stem
    raise ValueError(f"Unknown contract file: {name}")


def validate(kind, obj):
    errors = sorted(_validator(kind).iter_errors(obj), key=lambda e: list(e.path))
    if errors:
        msgs = [f"  {'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:10]]
        raise ValueError(f"{kind} does not match contract v{SCHEMA_VERSION}:\n" + "\n".join(msgs))


def replay_filename(scenario_id, seed, dropout):
    return f"replay_{scenario_id}_s{seed}_d{round(dropout * 100)}.json"


def write(obj, filename, out_dir=WEB_DATA_DIR):
    """Validate, then write compact JSON. Returns the path written."""
    validate(kind_for(filename), obj)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    return path
