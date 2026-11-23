"""MotherDuck query worker, run as a bare subprocess by src/database.py.

Usage: python md_worker.py <output_dir>

stdin carries {"queries": [{"name", "sql", "params"}, ...]}. Each result is
written to <output_dir>/<name>.parquet, and manifest.json maps every name to
"ok" or its error message. All queries share one connection, so a page load
pays one process spawn and one MotherDuck handshake.

Exit codes: 0 the connection worked (per-query errors are in the manifest),
3 the connection failed, 2 anything else went wrong.

Why a bare script: the previous multiprocessing approach re-imported the
app's __main__ in every child (that is how spawn works), pulling the whole
app stack into each query and eventually crashing the parent. This file
imports only duckdb and pandas. Results go to files because the MotherDuck
client prints notices (like quota warnings) to stdout.
"""

import json
import os
import sys


def main():
    import duckdb

    out_dir = sys.argv[1]
    queries = json.loads(sys.stdin.read())["queries"]
    # GDELT_DB points the worker at a local duckdb file (tests, local runs)
    db = os.getenv("GDELT_DB") or f'md:gdelt_db?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}'
    try:
        con = duckdb.connect(db, read_only=True)
    except Exception as e:
        print(f"connection failed: {e}", file=sys.stderr)
        sys.exit(3)

    manifest = {}
    try:
        for q in queries:
            try:
                params = q.get("params")
                rel = con.execute(q["sql"], params) if params is not None else con.execute(q["sql"])
                rel.df().to_parquet(os.path.join(out_dir, f"{q['name']}.parquet"))
                manifest[q["name"]] = "ok"
            except Exception as e:  # reported per query
                manifest[q["name"]] = str(e)
    finally:
        con.close()
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"worker error: {e}", file=sys.stderr)
        sys.exit(2)
