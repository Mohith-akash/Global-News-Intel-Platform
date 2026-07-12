"""
Database connection and query utilities for GDELT platform.

MotherDuck calls run in a child process, not in the streamlit process.
Reason: when the warehouse is quota-blocked, the duckdb native client can
hang while holding the GIL (freezing every thread, so in-process timeouts
never fire) or segfault outright (killing whatever process it runs in).
Both took the whole app down repeatedly. In a child process, a hang gets
killed by the timeout and a segfault only breaks the child - the app keeps
serving and shows a quota banner instead of dying.
"""

import time
import logging
import functools
import multiprocessing
import concurrent.futures
import pandas as pd
import streamlit as st

logger = logging.getLogger("gdelt")


class WarehouseUnavailable(RuntimeError):
    """MotherDuck is unreachable or out of free-tier quota."""


# After a hung or crashed warehouse call, skip MotherDuck entirely for this
# long. Retrying a quota-blocked warehouse on every page load just piles up
# dead child processes.
_BREAKER_COOLDOWN = 600
_breaker_until = 0.0


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


def _query_worker(sql, params):
    """Executed in a child process. Opens its own MotherDuck connection,
    runs one query, returns the DataFrame. Module-level so spawn can pickle it.
    """
    import os
    import duckdb
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


def _open_breaker(reason):
    global _breaker_until
    _breaker_until = time.time() + _BREAKER_COOLDOWN
    logger.error("MotherDuck %s - circuit breaker open for %ss", reason, _BREAKER_COOLDOWN)


def safe_query(conn, sql, params=None):  # noqa: ARG001 — conn kept for call-site compat
    """Execute SQL against MotherDuck inside an isolated child process.

    Crash modes this survives (all observed in production):
      1. Child hangs on a quota-blocked connection -> timeout fires, child is
         terminated, breaker opens.
      2. Child segfaults in the native client -> BrokenProcessPool raised in
         the parent, breaker opens. The app itself never dies.

    With @st.cache_data TTL=24h on the callers, the ~1s process-spawn
    overhead is paid a handful of times per day.

    Pass `params` (a list) for parameterized queries — the RAG keyword filters
    use this to bind values safely instead of string interpolation.
    """
    if time.time() < _breaker_until:
        raise WarehouseUnavailable("warehouse circuit breaker open")

    ctx = multiprocessing.get_context("spawn")
    ex = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    try:
        future = ex.submit(_query_worker, sql, params)
        return future.result(timeout=45)
    except concurrent.futures.TimeoutError:
        _open_breaker("query hung >45s")
        for p in getattr(ex, "_processes", {}).values():
            try:
                p.terminate()
            except Exception:
                pass
        raise WarehouseUnavailable("warehouse connection timed out") from None
    except concurrent.futures.process.BrokenProcessPool:
        _open_breaker("worker process died (native crash)")
        raise WarehouseUnavailable("warehouse client crashed") from None
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def get_db():
    """Kept for call-site compatibility; safe_query ignores the handle and
    every query runs in its own child process, so there is no shared
    connection anymore - and no MotherDuck call on the boot path at all.
    """
    return None


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
