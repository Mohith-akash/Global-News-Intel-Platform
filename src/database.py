"""
Database connection and query utilities for GDELT platform.
"""

import os
import logging
import threading
import warnings
import concurrent.futures
import pandas as pd
import duckdb
import streamlit as st
from sqlalchemy import create_engine

logger = logging.getLogger("gdelt")

# DuckDB connections are NOT thread-safe — concurrent conn.execute() calls from
# multiple Streamlit sessions sharing the same @st.cache_resource connection
# cause a NULL dereference and segfault. Serialise all queries through this lock.
_query_lock = threading.Lock()

# duckdb-engine doesn't support DOUBLE[] (list) column types during schema
# reflection — suppress the harmless warning it emits on every connection.
warnings.filterwarnings(
    "ignore",
    message="Did not recognize type",
    module="duckdb_engine",
)


@st.cache_resource(ttl=3600)
def get_db():
    """Connect to MotherDuck with a 30s timeout so a hung handshake doesn't block forever."""
    def _connect():
        return duckdb.connect(
            f'md:gdelt_db?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}',
            read_only=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_connect)
        try:
            return future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            st.error("⚠️ Database connection timed out — please refresh the page.")
            st.stop()


@st.cache_resource(ttl=3600)
def get_engine():
    """Create SQLAlchemy engine."""
    return create_engine(
        f"duckdb:///md:gdelt_db?motherduck_token={os.getenv('MOTHERDUCK_TOKEN')}"
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


def safe_query(conn, sql, timeout=15):  # noqa: ARG001 — timeout kept for call-site compat
    """Execute SQL safely.

    Acquires a module-level lock before every execute() call because DuckDB
    connections are not thread-safe. Multiple Streamlit sessions share the same
    @st.cache_resource connection, so without this lock concurrent requests race
    on conn.execute(), corrupt internal state, and segfault the whole app.
    """
    with _query_lock:
        try:
            return conn.execute(sql).df()
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return pd.DataFrame()
