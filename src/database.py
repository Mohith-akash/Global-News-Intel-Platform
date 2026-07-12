"""
Database connection and query utilities for GDELT platform.
"""

import os
import time
import logging
import functools
import concurrent.futures
import pandas as pd
import duckdb
import streamlit as st

logger = logging.getLogger("gdelt")


class WarehouseUnavailable(RuntimeError):
    """MotherDuck is unreachable or out of free-tier quota."""


# After a hung connection, skip MotherDuck entirely for this long. Retrying a
# quota-blocked warehouse every page load just piles up abandoned threads.
_BREAKER_COOLDOWN = 600
_breaker_until = 0.0


def _run_with_timeout(fn, timeout):
    """Run fn in a worker thread with a hard timeout.

    Deliberately avoids `with ThreadPoolExecutor(...)`: the context manager
    calls shutdown(wait=True) on exit, which blocks on the hung thread and
    freezes the main thread anyway - that was the crash mode where the app
    stopped answering health checks and got killed. shutdown(wait=False)
    abandons the stuck thread and lets the app keep serving.
    """
    global _breaker_until
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = ex.submit(fn)
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        _breaker_until = time.time() + _BREAKER_COOLDOWN
        logger.error("MotherDuck call hung >%ss - opening circuit breaker", timeout)
        raise WarehouseUnavailable("warehouse connection timed out") from None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def retry_cache_race(fn):
    """Work around a Streamlit @st.cache_data race at TTL expiry.

    The in-memory cache storage checks `key in cache` then reads `cache[key]`;
    if the entry expires between the two (concurrent sessions), cachetools
    raises a bare KeyError that crashes the whole page. Retrying once hits a
    clean cache miss and recomputes. Apply as the OUTERMOST decorator, above
    @st.cache_data.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError:
            logger.warning("st.cache_data TTL-expiry race hit, retrying once")
            return fn(*args, **kwargs)
    return wrapper

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
    if time.time() < _breaker_until:
        raise WarehouseUnavailable("warehouse circuit breaker open")
    return _run_with_timeout(_new_conn, 30)


@retry_cache_race
@st.cache_data(ttl=86400)
def detect_table(_conn):
    """Find the main events table."""
    df = safe_query(_conn, "SHOW TABLES")
    if not df.empty:
        for name in df.iloc[:, 0].tolist():
            if 'event' in name.lower():
                return name
        return df.iloc[0, 0]
    return 'events_dagster'


def safe_query(conn, sql, params=None):  # noqa: ARG001 — conn kept for call-site compat
    """Execute SQL on a fresh connection.

    Each call opens its own MotherDuck connection, runs the query, then closes it.
    This eliminates two crash modes that hit with the previous shared-connection approach:
      1. Concurrent sessions racing on conn.execute() → NULL dereference segfault
      2. Stale/dropped connection cached in @st.cache_resource → segfault on every
         query after idle, causing a restart loop

    With @st.cache_data TTL=24h, this function only fires a handful of times per day in
    practice, so the per-call connection overhead is negligible.

    Pass `params` (a list) for parameterized queries — the RAG keyword filters
    use this to bind values safely instead of string interpolation.
    """
    if time.time() < _breaker_until:
        raise WarehouseUnavailable("warehouse circuit breaker open")

    def _run():
        c = None
        try:
            c = _new_conn()
            if params is not None:
                return c.execute(sql, params).df()
            return c.execute(sql).df()
        finally:
            if c is not None:
                c.close()

    try:
        return _run_with_timeout(_run, 25)
    except WarehouseUnavailable:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return pd.DataFrame()
