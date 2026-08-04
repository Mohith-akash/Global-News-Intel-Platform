"""MotherDuck query worker, run as a bare subprocess.

Usage: python md_worker.py <output_path>

Single mode: stdin carries {"sql": ..., "params": ...}; <output_path> is a
file and the result is written to it as parquet. Exit 0 ok, 2 query error.

Batch mode: stdin carries {"queries": [{"name","sql","params"}, ...]};
<output_path> is a DIRECTORY. Each query result is written as
<name>.parquet and a manifest.json maps name -> "ok" or the error message.
The whole batch shares ONE connection, so the caller pays a single spawn
and a single MotherDuck handshake instead of one per query. Exit 0 as long
as the connection worked (per-query errors live in the manifest), 2 if the
connection itself failed.

Runs as its own script ON PURPOSE. The previous multiprocessing approach
re-imported the app's __main__ module in every spawn child (that is how
spawn bootstrapping works), which pulled in the full app stack including
transformers - hundreds of MB and seconds of CPU per query, and enough
memory pressure to evict streamlit caches and eventually crash the parent.
A bare subprocess imports only what this file imports.

The result goes to a FILE, not stdout, ON PURPOSE: the MotherDuck client
prints notices (like the free-tier quota warning) straight to stdout, which
corrupted the parquet byte stream when we piped results through it.
"""

import os
import sys
import json


def _run(conn, sql, params):
    if params is not None:
        return conn.execute(sql, params).df()
    return conn.execute(sql).df()


def main():
    import duckdb

    out_path = sys.argv[1]
    req = json.loads(sys.stdin.read())

    c = duckdb.connect(
        f'md:gdelt_db?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}',
        read_only=True,
    )
    try:
        if "queries" in req:
            manifest = {}
            for q in req["queries"]:
                name = q["name"]
                try:
                    df = _run(c, q["sql"], q.get("params"))
                    df.to_parquet(os.path.join(out_path, f"{name}.parquet"))
                    manifest[name] = "ok"
                except Exception as e:  # noqa: BLE001 - reported per query
                    manifest[name] = str(e)
            with open(os.path.join(out_path, "manifest.json"), "w") as f:
                json.dump(manifest, f)
        else:
            df = _run(c, req["sql"], req.get("params"))
            df.to_parquet(out_path)
    finally:
        c.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - report and exit nonzero
        print(f"query error: {e}", file=sys.stderr)
        sys.exit(2)
