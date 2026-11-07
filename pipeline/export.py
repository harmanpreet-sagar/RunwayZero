"""Write real pipeline output into web/public/data (stub, filled in tonight).

Every write goes through pipeline.contracts.write(), which validates against the
schema before touching disk, so the UI never receives a broken file.

    python -m pipeline.export
"""
from pipeline.contracts import replay_filename, write  # noqa: F401


def main():
    raise SystemExit("Not implemented yet: use `python -m pipeline.make_fixtures` + `node scripts/sync-fixtures.mjs` for mock data.")


if __name__ == "__main__":
    main()
