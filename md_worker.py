"""MotherDuck query worker, run as a bare subprocess.

Usage: python md_worker.py
Reads one JSON object {"sql": ..., "params": ...} from stdin, runs the query
against MotherDuck, writes the result as parquet bytes to stdout.
Exit codes: 0 ok, 2 query error (message on stderr).

Runs as its own script ON PURPOSE. The previous multiprocessing approach
re-imported the app's __main__ module in every spawn child (that is how
spawn bootstrapping works), which pulled in the full app stack including
transformers - hundreds of MB and seconds of CPU per query, and enough
memory pressure to evict streamlit caches and eventually crash the parent.
A bare subprocess imports only what this file imports.
"""

import io
import os
import sys
import json


def main():
    import duckdb

    req = json.loads(sys.stdin.read())
    sql = req["sql"]
    params = req.get("params")

    c = duckdb.connect(
        f'md:gdelt_db?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}',
        read_only=True,
    )
    try:
        if params is not None:
            df = c.execute(sql, params).df()
        else:
            df = c.execute(sql).df()
    finally:
        c.close()

    buf = io.BytesIO()
    df.to_parquet(buf)
    sys.stdout.buffer.write(buf.getvalue())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - report and exit nonzero
        print(f"query error: {e}", file=sys.stderr)
        sys.exit(2)
