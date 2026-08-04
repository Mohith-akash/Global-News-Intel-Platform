"""
Database connection and query utilities for GDELT platform.

MotherDuck calls run in a bare subprocess (python md_worker.py), not in the
streamlit process and not via multiprocessing.
Reason 1: when the warehouse is quota-blocked, the duckdb native client can
hang while holding the GIL (freezing every thread, so in-process timeouts
never fire) or segfault outright (killing whatever process it runs in).
Reason 2: multiprocessing spawn re-imports the app's __main__ module in the
child, which drags in the entire app stack (llama-index, transformers) -
hundreds of MB per query child. That memory pressure evicted streamlit
caches and eventually crashed the parent with a native allocation failure.
A bare subprocess imports only duckdb and pandas, and subprocess.run kills
it automatically on timeout.
"""

import os
import sys
import json
import time
import shutil
import logging
import tempfile
import functools
import subprocess
import pandas as pd
import streamlit as st

logger = logging.getLogger("gdelt")

_WORKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "md_worker.py")


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


def _open_breaker(reason):
    global _breaker_until
    _breaker_until = time.time() + _BREAKER_COOLDOWN
    logger.error("MotherDuck %s - circuit breaker open for %ss", reason, _BREAKER_COOLDOWN)


def safe_query(conn, sql, params=None):  # noqa: ARG001 — conn kept for call-site compat
    """Execute SQL against MotherDuck in an isolated subprocess.

    Crash modes this survives (all observed in production):
      1. Child hangs on a quota-blocked connection -> subprocess.run kills it
         at the timeout, breaker opens.
      2. Child segfaults in the native client -> nonzero returncode in the
         parent, breaker opens. The app itself never dies.

    The child runs md_worker.py directly, so it imports only duckdb and
    pandas - no multiprocessing bootstrapping, no app-stack re-import.
    With @st.cache_data TTL=24h on the callers, the ~1s subprocess overhead
    is paid a handful of times per day.

    Pass `params` (a list) for parameterized queries — the RAG keyword filters
    use this to bind values safely instead of string interpolation.
    """
    if time.time() < _breaker_until:
        raise WarehouseUnavailable("warehouse circuit breaker open")

    payload = json.dumps({"sql": sql, "params": params})
    fd, out_path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    try:
        try:
            proc = subprocess.run(
                [sys.executable, _WORKER, out_path],
                input=payload.encode(),
                capture_output=True,
                timeout=45,
            )
        except subprocess.TimeoutExpired:
            _open_breaker("query hung >45s")
            raise WarehouseUnavailable("warehouse connection timed out") from None

        if proc.returncode == 0:
            try:
                return pd.read_parquet(out_path)
            except Exception as e:
                logger.error("Worker returned unreadable result: %s", e)
                return pd.DataFrame()

        stderr = proc.stderr.decode(errors="replace").strip()
        if proc.returncode == 2:
            # clean query error reported by the worker (bad SQL, missing table...)
            logger.error("Query error: %s", stderr)
            return pd.DataFrame()

        # any other exit means the native client died (segfault = -11)
        _open_breaker(f"worker died with code {proc.returncode}: {stderr[-200:]}")
        raise WarehouseUnavailable("warehouse client crashed") from None
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def safe_query_batch(queries, timeout=90):
    """Run many queries in ONE worker subprocess.

    `queries` is {name: (sql, params_or_None)}. Returns {name: DataFrame},
    with an empty frame (and a log line) for any query that errored — same
    per-query contract as safe_query.

    This exists because a cold dashboard load runs ~17 cached queries; one
    subprocess each meant ~17 python spawns and ~17 MotherDuck handshakes,
    rendering the page piece by piece for 30-60s. One worker doing the whole
    batch pays the spawn + handshake once.

    Raises WarehouseUnavailable (and opens the breaker) if the connection
    itself hangs, dies, or is quota-blocked — identical to safe_query.
    """
    if time.time() < _breaker_until:
        raise WarehouseUnavailable("warehouse circuit breaker open")

    payload = json.dumps({"queries": [
        {"name": name, "sql": sql, "params": params}
        for name, (sql, params) in queries.items()
    ]})
    out_dir = tempfile.mkdtemp()
    try:
        try:
            proc = subprocess.run(
                [sys.executable, _WORKER, out_dir],
                input=payload.encode(),
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            _open_breaker(f"batch hung >{timeout}s")
            raise WarehouseUnavailable("warehouse connection timed out") from None

        stderr = proc.stderr.decode(errors="replace").strip()
        if proc.returncode != 0:
            # batch mode reports per-query errors in the manifest, so any
            # nonzero exit means the connection itself failed or the native
            # client died
            _open_breaker(f"batch worker died with code {proc.returncode}: {stderr[-200:]}")
            raise WarehouseUnavailable("warehouse client crashed") from None

        try:
            with open(os.path.join(out_dir, "manifest.json")) as f:
                manifest = json.load(f)
        except Exception:
            _open_breaker("batch worker wrote no manifest")
            raise WarehouseUnavailable("warehouse client crashed") from None

        results = {}
        for name in queries:
            status = manifest.get(name)
            if status == "ok":
                try:
                    results[name] = pd.read_parquet(os.path.join(out_dir, f"{name}.parquet"))
                    continue
                except Exception as e:
                    logger.error("Unreadable batch result %s: %s", name, e)
            else:
                logger.error("Batch query %s failed: %s", name, status)
            results[name] = pd.DataFrame()
        return results
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


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
