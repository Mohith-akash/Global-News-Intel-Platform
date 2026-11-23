"""Snapshot tests: pin the app's current behaviour so refactors can't change it.

Inputs are fixed (frozen clock, a small local duckdb warehouse, a fake LLM and
a fake embedding API); outputs are compared to tests/snapshots/*.json.
Regenerate after an intended behaviour change with UPDATE_SNAPSHOTS=1.
"""

import datetime as _dt
import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import streamlit as st

# import the app before the clock is patched
import src.database, src.queries, src.rag_engine, src.data_processing, src.sql_templates  # noqa: E401,F401
import components.ai_chat  # noqa: F401

SNAP_DIR = Path(__file__).parent / "snapshots"
UPDATE = os.getenv("UPDATE_SNAPSHOTS") == "1"
NOW = _dt.datetime(2026, 11, 23, 10, 0)  # a Monday


class _FrozenDateTime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(_dt, "datetime", _FrozenDateTime)


def check(name, value):
    path = SNAP_DIR / f"{name}.json"
    data = json.loads(json.dumps(value, default=str))
    if UPDATE or not path.exists():
        SNAP_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    assert data == json.loads(path.read_text(encoding="utf-8")), f"snapshot {name} changed"


PROMPTS = [
    "What happened in Germany today?", "crisis in Ukraine and Russia",
    "which countries have a crisis this week", "top 7 countries by events",
    "how many events this month", "major events this month", "news from France and Germany",
    "severe events", "total events in Germany", "latest news", "events last week",
    "what happened this week in India", "events on oct 30", "news 15 november",
    "events in october", "summarize the news", "show me 3 events from Japan",
    "how many events this year", "all time total", "conflicts in the middle east",
    "trending stories in the united states", "what is happening in south korea",
    "march events", "top 20 events", "may 5 news", "events in december",
    "biggest stories last week", "crisis countries this month", "news today in Brazil",
    "tell me about the war",
]


def _date_bounds(qi):
    """Whatever function the chat uses to turn a question into date bounds."""
    import components.ai_chat as chat
    fn = getattr(chat, "resolve_date_range", None)
    if fn is None:
        from src.sql_templates import date_bounds as fn
    from src.utils import get_dates
    return list(fn(qi, get_dates()))


def test_query_understanding():
    from src.utils import detect_query_type, get_dates
    from src.sql_templates import build_sql
    out = {}
    for p in PROMPTS:
        qi = detect_query_type(p)
        plan = build_sql(p, qi, get_dates())
        out[p] = {
            "qi": {k: v for k, v in sorted(qi.items())},
            "bounds": _date_bounds(qi),
            "sql": plan.sql, "params": plan.params, "limit": plan.limit,
            "country_agg": plan.is_country_aggregate, "count_agg": plan.is_count_aggregate,
            "country_name": plan.country_filter_name, "display": plan.display_sql(),
        }
    check("query_understanding", out)


def test_labels():
    from src import utils
    scores = [None, -9, -8, -7, -6, -5, -4.5, -4, -3, -2, -1.5, -1, 0, 0.5, 1, 2, 3, 4, 5, 6, 8]
    out = {}
    out["get_intensity_label"] = [utils.get_intensity_label(s) for s in scores]
    out["country"] = [utils.get_country(c) for c in ["DEU", "US", "AFR", "XX", "", None, "usa", "ZZZ"]]
    out["code"] = [utils.get_country_code(w) for w in ["germany", "uk", "the", "korea", "india", "12", "us", "may", "fra"]]
    check("labels", out)


FEED_ROWS = [
    # DATE, NEWS_LINK, HEADLINE, HEADLINE_AI, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT
    ("20261122", "https://news.example.com/world/2026/11/22/berlin-talks-on-energy-prices-continue", "Berlin talks on energy prices continue", None, "GERMANY", "DEU", -5.0, 25),
    ("20261121", "https://example.org/senate-passes-budget-bill-after-long-debate", "senate passes budget bill after long debate", "Senate Passes Budget Bill After Long Debate", "CONGRESS", "USA", 2.0, 30),
    ("20261120", "https://example.net/abc123", "abc123", None, "UKRAINE", "UKR", -7.0, 10),
    ("20261119", "https://site.fr/rail-strike-spreads-across-france", "Rail strike spreads across France", None, "UNION", "FRA", -3.0, 22),
    ("20261118", "https://site.fr/rail-strikes-spread-across-france-today", "Rail strikes spread across France today", None, "UNION", "FRA", -2.0, 12),
    ("20261117", "https://cn.example/trade-figures-released-for-october", "Trade figures released for October", None, "CHINA", "CHN", 3.5, 50),
    ("20261116", None, None, None, "NOBODY", None, 0.0, 1),
]
FEED_COLS = ["DATE", "NEWS_LINK", "HEADLINE", "HEADLINE_AI", "MAIN_ACTOR", "ACTOR_COUNTRY_CODE", "IMPACT_SCORE", "ARTICLE_COUNT"]


def test_feed_processing():
    from src.data_processing import process_df
    df = process_df(pd.DataFrame(FEED_ROWS, columns=FEED_COLS))
    check("feed_processing", df.to_dict(orient="records"))


# ---------- a small local warehouse for the worker-backed paths ----------

DIM = 1024


def _vec(seed):
    return [((i * 7 + seed * 13) % 17) / 17.0 for i in range(DIM)]


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    db = tmp_path_factory.mktemp("wh") / "gdelt.duckdb"
    c = duckdb.connect(str(db))
    c.execute("""CREATE TABLE events_dagster (EVENT_ID VARCHAR, DATE VARCHAR, MAIN_ACTOR VARCHAR,
        ACTOR_COUNTRY_CODE VARCHAR, EVENT_CATEGORY_CODE VARCHAR, IMPACT_SCORE DOUBLE, ARTICLE_COUNT BIGINT,
        SENTIMENT_SCORE DOUBLE, NEWS_LINK VARCHAR, HEADLINE VARCHAR, HEADLINE_AI VARCHAR, EMBEDDING DOUBLE[])""")
    for i, (d, link, h, hai, actor, cc, imp, n) in enumerate(FEED_ROWS * 3):
        day = (NOW - _dt.timedelta(days=i % 12)).strftime("%Y%m%d")
        c.execute("INSERT INTO events_dagster VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  [f"E{i}", day, actor, cc, "01", imp, n, 0.0, link, h, hai, _vec(i)])
    c.close()
    os.environ["GDELT_DB"] = str(db)
    yield db
    os.environ.pop("GDELT_DB", None)


def _frame(df):
    return df.round(6).astype(object).where(df.notna(), None).to_dict(orient="records")


def test_dashboard_queries(warehouse):
    from src import queries
    st.cache_data.clear()
    out = {"metrics": queries.get_metrics("events_dagster")}
    for name in ["alerts", "trending", "feed", "countries", "timeseries", "sentiment", "actors", "distribution"]:
        # sorted: several of these ORDER BY a count, and tied rows may come back in any order
        out[name] = sorted(_frame(getattr(queries, f"get_{name}")("events_dagster")), key=str)
    check("dashboard_queries", out)


def test_safe_query_errors(warehouse):
    from src.database import safe_query
    assert safe_query("SELECT 41 + 1 AS x").iloc[0]["x"] == 42
    assert safe_query("SELECT ? AS y", [7]).iloc[0]["y"] == 7
    assert safe_query("SELECT * FROM no_such_table").empty  # query error -> empty frame, no breaker
    assert safe_query("SELECT 1 AS ok").iloc[0]["ok"] == 1  # breaker still closed


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return "ANSWER"


def test_rag_pipeline(warehouse, monkeypatch):
    from src import rag_engine
    monkeypatch.setattr(rag_engine, "get_embedding", lambda text: _vec(1))
    llm = FakeLLM()
    res = rag_engine.rag_query("energy prices and strikes in europe", llm, top_k=15,
                               min_date="20261101", table_name="events_dagster")
    check("rag_pipeline", {"answer": res["answer"], "prompt": llm.prompts,
                           "headlines": _frame(res["headlines"].drop(columns=["similarity"], errors="ignore"))})
    # keyword-only fallback when the embedding API is down
    monkeypatch.setattr(rag_engine, "get_embedding", lambda text: None)
    fb = rag_engine.search_similar_headlines("rail strike france", top_k=5, table_name="events_dagster")
    check("rag_fallback", _frame(fb))
