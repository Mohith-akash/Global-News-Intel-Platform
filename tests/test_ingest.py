"""Ingest helpers: GCAM emotion parsing and the duplicate-safe loader."""

import duckdb
import polars as pl
import pytest

from etl.pipeline_polars import GCAM_EMOTIONS, _insert_new, parse_gcam_field


def test_gcam_scores_are_wordnet_affect_per_1000_words():
    # anger (c15.15) 2 words, fear (c15.198) 1 word, a Roget code that must be ignored
    scores = parse_gcam_field("wc:500,c9.1:40,c15.15:2,c15.198:1,v10.1:0.5")
    assert scores["anger"] == pytest.approx(4.0)
    assert scores["fear"] == pytest.approx(2.0)
    assert scores["joy"] == 0.0
    assert set(scores) == set(GCAM_EMOTIONS.values())
    assert all(code.startswith("c15.") for code in GCAM_EMOTIONS)


@pytest.mark.parametrize("field", [None, "", "c15.15:2", "wc:0,c15.15:2", "wc:abc,c15.15:2"])
def test_gcam_without_a_word_count_scores_zero(field):
    assert set(parse_gcam_field(field).values()) == {0.0}


def _events(ids, date="20261123"):
    return pl.DataFrame({"EVENT_ID": ids, "DATE": [date] * len(ids), "HEADLINE": [f"h{i}" for i in ids]})


def test_insert_new_creates_then_skips_duplicates():
    con = duckdb.connect()
    assert _insert_new(con, "events", _events(["1", "2"]), "EVENT_ID") == 2  # first run creates the table
    con.execute("ALTER TABLE events ADD COLUMN HEADLINE_AI VARCHAR")       # columns the batch doesn't have
    assert _insert_new(con, "events", _events(["2", "3"]), "EVENT_ID") == 1
    rows = con.execute("SELECT EVENT_ID, HEADLINE_AI FROM events ORDER BY 1").fetchall()
    assert rows == [("1", None), ("2", None), ("3", None)]


def test_insert_new_with_numeric_dates():
    con = duckdb.connect()
    gkg = pl.DataFrame({"GKG_ID": ["a", "b"], "DATE": [20261123090000, 20261123091500]})
    assert _insert_new(con, "gkg", gkg, "GKG_ID") == 2
    assert _insert_new(con, "gkg", gkg, "GKG_ID") == 0
