"""
Database connection and query utilities for GDELT platform.
"""

import os
import logging
import warnings
import concurrent.futures
import pandas as pd
import duckdb
import streamlit as st
from sqlalchemy import create_engine

logger = logging.getLogger("gdelt")

# duckdb-engine doesn't support DOUBLE[] (list) column types during schema
# reflection — suppress the harmless warning it emits on every connection.
warnings.filterwarnings(
    "ignore",
    message="Did not recognize type",
    module="duckdb_engine",
)

_MOTHERDUCK_URL = "md:gdelt_db"


def _new_conn():
    """Open a fresh read-only MotherDuck connection."""
    return duckdb.connect(
        f'{_MOTHERDUCK_URL}?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}',
        read_only=True,
    )


@st.cache_resource(ttl=3600)
def get_db():
    """Return a connection for callers that need a persistent handle (detect_table etc).

    Most query paths go through safe_query() which opens its own fresh connection,
    so this shared handle is only used for one-off metadata calls.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_new_conn)
        try:
            return future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            st.error("⚠️ Database connection timed out — please refresh the page.")
            st.stop()


@st.cache_resource(ttl=3600)
def get_engine():
    """Create SQLAlchemy engine."""
    return create_engine(
        f"duckdb:///{_MOTHERDUCK_URL}?motherduck_token={os.getenv('MOTHERDUCK_TOKEN')}"
    )


@st.cache_data(ttl=3600)
def detect_table(_conn):
    """Find the main events table."""
    df = safe_query(_conn, "SHOW TABLES")
    if not df.empty:
        for name in df.iloc[:, 0].tolist():
            if 'event' in name.lower():
                return name
        return df.iloc[0, 0]
    return 'events_dagster'


def safe_query(conn, sql, timeout=15):  # noqa: ARG001 — conn/timeout kept for call-site compat
    """Execute SQL on a fresh connection.

    Each call opens its own MotherDuck connection, runs the query, then closes it.
    This eliminates two crash modes that hit with the previous shared-connection approach:
      1. Concurrent sessions racing on conn.execute() → NULL dereference segfault
      2. Stale/dropped connection cached in @st.cache_resource → segfault on every
         query after idle, causing a restart loop

    With @st.cache_data TTL=4hr, this function only fires ~6 times per day in
    practice, so the per-call connection overhead is negligible.
    """
    try:
        conn = _new_conn()
        result = conn.execute(sql).df()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return pd.DataFrame()
