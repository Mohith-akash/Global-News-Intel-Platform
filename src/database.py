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
    try:
        result = _conn.execute("SHOW TABLES").df()
        if not result.empty:
            for name in result.iloc[:, 0].tolist():
                if 'event' in name.lower():
                    return name
            return result.iloc[0, 0]
    except Exception:
        pass
    return 'events_dagster'


def safe_query(conn, sql, timeout=15):  # noqa: ARG001 — timeout kept for call-site compat
    """Execute SQL with error handling.

    MotherDuck enforces its own server-side query timeouts, so no client-side
    timer is needed here. The previous threading.Timer + conn.interrupt() approach
    caused a NULL dereference when interrupt fired during .df() materialisation,
    leaving the connection in a permanently broken state.
    """
    try:
        return conn.execute(sql).df()
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return pd.DataFrame()
