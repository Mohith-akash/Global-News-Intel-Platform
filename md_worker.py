"""MotherDuck query worker, executed in spawned child processes.

Lives at the repo root ON PURPOSE: src/__init__.py imports the whole app
(llama-index, transformers, every cached query module), so a worker inside
the src package made every spawned child re-import hundreds of MB and burn
seconds of CPU per query. This module imports os and duckdb, nothing else.
"""

import os
import duckdb


def run_query(sql, params=None):
    """Open a fresh read-only MotherDuck connection, run one query,
    return the DataFrame."""
    c = duckdb.connect(
        f'md:gdelt_db?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}',
        read_only=True,
    )
    try:
        if params is not None:
            return c.execute(sql, params).df()
        return c.execute(sql).df()
    finally:
        c.close()
