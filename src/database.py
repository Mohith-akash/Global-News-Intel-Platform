"""
MotherDuck access for the app.

Every query runs in a bare subprocess (md_worker.py). When the warehouse is
quota-blocked the duckdb client can hang while holding the GIL, so in-process
timeouts never fire, or segfault and take its process down. In a child
process both become a killed or failed subprocess, and a circuit breaker
stops the app retrying a dead warehouse on every page load.
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


def _call_worker(queries, timeout):
    """Run {name: (sql, params)} in one md_worker subprocess.

    Returns (exit_code, stderr, {name: DataFrame or error message}). A hang
    is the one failure handled here: the child is killed at the timeout and
    the breaker opens. Callers decide what a nonzero exit means.
    """
    if time.time() < _breaker_until:
        raise WarehouseUnavailable("warehouse circuit breaker open")

    payload = json.dumps({"queries": [
        {"name": name, "sql": sql, "params": params} for name, (sql, params) in queries.items()
    ]})
    out_dir = tempfile.mkdtemp()
    try:
        try:
            proc = subprocess.run([sys.executable, _WORKER, out_dir], input=payload.encode(),
                                  capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            _open_breaker(f"query hung >{timeout}s")
            raise WarehouseUnavailable("warehouse connection timed out") from None

        stderr = proc.stderr.decode(errors="replace").strip()
        if proc.returncode != 0:
            return proc.returncode, stderr, {}
        try:
            with open(os.path.join(out_dir, "manifest.json")) as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            _open_breaker("worker wrote no manifest")
            raise WarehouseUnavailable("warehouse client crashed") from None

        results = {}
        for name in queries:
            status = manifest.get(name, "no result")
            if status == "ok":
                try:
                    results[name] = pd.read_parquet(os.path.join(out_dir, f"{name}.parquet"))
                    continue
                except Exception as e:
                    status = f"unreadable result: {e}"
            results[name] = status
        return 0, stderr, results
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _frames(results):
    """Error messages become empty frames (logged), like a query with no rows."""
    out = {}
    for name, r in results.items():
        if isinstance(r, str):
            logger.error("Query %s failed: %s", name, r)
            r = pd.DataFrame()
        out[name] = r
    return out


def safe_query_batch(queries, timeout=45):
    """Run {name: (sql, params_or_None)} in ONE worker -> {name: DataFrame}.

    A cold dashboard load runs ~17 queries; one subprocess each meant ~17
    spawns and MotherDuck handshakes and a page that rendered piece by piece
    for 30-60s. A query that errors comes back as an empty frame. If the
    connection itself fails, hangs or crashes, the breaker opens and
    WarehouseUnavailable is raised.

    The 45s timeout stays under Streamlit's 60s script health check; a
    normal batch takes ~3s.
    """
    code, stderr, results = _call_worker(queries, timeout)
    if code != 0:
        _open_breaker(f"worker died with code {code}: {stderr[-200:]}")
        raise WarehouseUnavailable("warehouse client crashed") from None
    return _frames(results)


def safe_queries(queries, timeout=45):
    """Several chat queries in one worker, with softer failure rules.

    Unlike a dashboard batch, a refused connection only empties these results
    (the chat says "no results") instead of opening the breaker for the whole
    dashboard. A hang or a native crash still opens it.
    """
    code, stderr, results = _call_worker(queries, timeout)
    if code in (2, 3):
        logger.error("Query error: %s", stderr)
        return {name: pd.DataFrame() for name in queries}
    if code != 0:  # the native client died (segfault = -11)
        _open_breaker(f"worker died with code {code}: {stderr[-200:]}")
        raise WarehouseUnavailable("warehouse client crashed") from None
    return _frames(results)


def safe_query(sql, params=None, timeout=45):
    """One query; pass `params` (a list) to bind values."""
    return safe_queries({"q": (sql, params)}, timeout)["q"]


@retry_cache_race
@st.cache_data(ttl=86400)
def detect_table():
    """Find the main events table."""
    df = safe_query("SHOW TABLES")
    if not df.empty:
        for name in df.iloc[:, 0].tolist():
            if 'event' in name.lower():
                return name
        return df.iloc[0, 0]
    return 'events_dagster'
