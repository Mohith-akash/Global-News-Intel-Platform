"""
Database connection and query utilities for GDELT platform.
"""

import os
import logging
import pandas as pd
import duckdb
import streamlit as st
from sqlalchemy import create_engine

logger = logging.getLogger("gdelt")


@st.cache_resource
def get_db():
    """Connect to MotherDuck."""
    return duckdb.connect(
        f'md:gdelt_db?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}',
        read_only=True
    )


@st.cache_resource
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


def safe_query(conn, sql):
    """Execute SQL safely."""
    try:
        return conn.execute(sql).df()
    except Exception as e:
        logger.error(f"Query error: {e}")
        return pd.DataFrame()
