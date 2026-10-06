"""SQL chat templates: bound parameters, same rows as the readable SQL, no injection."""

import duckdb
import pytest

from src.sql_templates import build_sql
from src.utils import detect_query_type

DATES = {
    "today": "20261006",
    "week_ago": "20260929",
    "month_ago": "20260906",
    "three_months_ago": "20260706",
}
ROWS = [
    ("1", "20261006", "DEU", "Berlin talks on energy prices", "GERMANY", -5.0, 25.0, "https://a"),
    ("2", "20261005", "USA", "Senate passes budget bill", "CONGRESS", 2.0, 30.0, "https://b"),
    ("3", "20261001", "UKR", "Shelling hits power station", "UKRAINE", -6.0, 10.0, "https://c"),
    ("4", "20261003", "FRA", "Rail strike across France", "UNION", -4.0, 22.0, "https://d"),
    ("5", "20260601", "CHN", "Trade figures released", "CHINA", 1.0, 50.0, "https://e"),
]
PROMPTS = [
    "What happened in Germany today?",
    "crisis in Ukraine and Russia",
    "which countries have a crisis this week",
    "top 7 countries by events",
    "how many events this month",
    "major events this month",
    "news from France and Germany",
    "severe events",
    "total events in Germany",
    "latest news",
]


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE events_dagster (EVENT_ID VARCHAR, DATE VARCHAR, ACTOR_COUNTRY_CODE VARCHAR, "
        "HEADLINE VARCHAR, MAIN_ACTOR VARCHAR, IMPACT_SCORE DOUBLE, ARTICLE_COUNT DOUBLE, NEWS_LINK VARCHAR)"
    )
    c.executemany("INSERT INTO events_dagster VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ROWS)
    yield c
    c.close()


def plan_for(prompt):
    return build_sql(prompt, detect_query_type(prompt), DATES)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_values_are_bound_not_inlined(prompt):
    plan = plan_for(prompt)
    assert plan.sql.count("?") == len(plan.params)
    assert "'" not in plan.sql


@pytest.mark.parametrize("prompt", PROMPTS)
def test_bound_query_matches_readable_query(con, prompt):
    plan = plan_for(prompt)
    bound = con.execute(plan.sql, plan.params).fetchall()
    shown = con.execute(plan.display_sql()).fetchall()
    # ties in COUNT(*) ordering can come back in either order
    assert sorted(bound) == sorted(shown)


def test_count_for_one_country(con):
    plan = plan_for("total events in Germany")
    assert plan.is_count_aggregate
    assert plan.country_filter_name == "Germany"
    assert con.execute(plan.sql, plan.params).fetchone()[0] == 1


def test_top_n_is_capped_at_ten():
    assert plan_for("top 50 countries by events").limit == 10


def test_quotes_in_the_question_cannot_change_the_sql(con):
    plan = plan_for("events in Germany'; DROP TABLE events_dagster; --")
    assert "DROP" not in plan.sql
    con.execute(plan.sql, plan.params).fetchall()
    assert con.execute("SELECT COUNT(*) FROM events_dagster").fetchone()[0] == len(ROWS)
