"""
SQL for the AI chat's SQL mode.

Questions are routed by keywords to a handful of tested query templates.
Dates and country codes are bound as parameters (?). The only values put
into the SQL text directly are the table name (a constant) and LIMITs,
which are ints parsed and capped in result_limit().
"""

import re
from dataclasses import dataclass, field

from src.utils import get_country, get_country_code

EVENT_COLS = "DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK"
BASE_WHERE = "MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL"
MULTI_WORD_REGIONS = [
    'middle east', 'united states', 'united kingdom', 'great britain',
    'south korea', 'north korea', 'saudi arabia', 'south africa',
    'new zealand',
]
# Require a few articles so headlines are real stories, not noise
ARTICLE_THRESHOLD = 3


@dataclass
class SqlPlan:
    sql: str
    params: list = field(default_factory=list)
    limit: int = 5
    is_country_aggregate: bool = False
    is_count_aggregate: bool = False
    country_filter_name: str | None = None

    def display_sql(self) -> str:
        """The query with its parameters written in, for showing in the chat."""
        out = self.sql
        for p in self.params:
            out = out.replace("?", f"'{p}'", 1)
        return out


def date_filter(qi: dict, dates: dict) -> tuple[str, list]:
    """WHERE clause for the time window the question asks about."""
    if qi['is_specific_date'] and qi['specific_date']:
        return "DATE = ?", [qi['specific_date']]
    if qi.get('is_week_range') and qi.get('week_start') and qi.get('week_end'):
        # "last week" or "this week"
        return "DATE >= ? AND DATE <= ?", [qi['week_start'], qi['week_end']]
    if qi.get('is_month_range') and qi.get('month_start') and qi.get('month_end'):
        # month-only questions like "events in october"
        return "DATE >= ? AND DATE <= ?", [qi['month_start'], qi['month_end']]
    if qi['time_period'] == 'all' or qi['is_aggregate']:
        return "DATE >= ?", [dates['three_months_ago']]
    if qi['time_period'] == 'month':
        return "DATE >= ?", [dates['month_ago']]
    if qi['time_period'] == 'day':
        return "DATE = ?", [dates['today']]
    return "DATE >= ?", [dates['week_ago']]


def result_limit(prompt: str, qi: dict) -> int:
    """Rows to show: default 5, '10 events' / 'top 7' up to 10."""
    limit = 5
    p = prompt.lower()
    m = re.search(r'(\d+)\s*(events?|results?|items?)', p)
    if m:
        limit = min(int(m.group(1)), 10)
    m2 = re.search(r'top\s+(\d+)', p)
    if m2:
        limit = min(int(m2.group(1)), 10)
    # month-range answers stay short
    if qi.get('is_month_range'):
        limit = min(limit, 5)
    return limit


def country_codes_from_prompt(text: str) -> list[str]:
    """3-letter country codes mentioned in the question, regions expanded."""
    codes = []
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())

    # Region aliases first ("middle east" -> several codes)
    try:
        from src.config import REGION_ALIASES
        for region, region_codes in REGION_ALIASES.items():
            if region in clean_text:
                codes.extend(region_codes)
                return codes
    except ImportError:
        pass

    for phrase in MULTI_WORD_REGIONS:
        if phrase in clean_text:
            code = get_country_code(phrase)
            if code and code not in codes:
                codes.append(code)

    for w in clean_text.split():
        if len(w) >= 2:
            code = get_country_code(w)
            if code and code not in codes:
                codes.append(code)
    return codes


def country_filter(codes: list[str]) -> tuple[str, list]:
    if len(codes) == 1:
        return "ACTOR_COUNTRY_CODE = ?", [codes[0]]
    return f"ACTOR_COUNTRY_CODE IN ({', '.join(['?'] * len(codes))})", list(codes)


def build_sql(prompt: str, qi: dict, dates: dict, tbl: str = "events_dagster") -> SqlPlan:
    """Pick the query template for a question. Order of the checks matters."""
    limit = result_limit(prompt, qi)
    # fetch more rows than shown, headline filtering and dedup drop some
    fetch_limit = min(limit * 50, 500)
    df_sql, df_params = date_filter(qi, dates)

    p = prompt.lower()
    has_crisis = 'crisis' in p or 'severe' in p
    has_country_word = 'countr' in p
    has_major = any(w in p for w in ('major', 'important', 'significant', 'biggest', 'trending'))

    # 1. countries with crisis events (before plain crisis)
    if has_crisis and has_country_word:
        return SqlPlan(
            f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as EVENT_COUNT FROM {tbl} WHERE {BASE_WHERE} "
            f"AND IMPACT_SCORE < -3 AND {df_sql} GROUP BY ACTOR_COUNTRY_CODE "
            f"ORDER BY EVENT_COUNT DESC LIMIT {limit}",
            df_params, limit, is_country_aggregate=True,
        )

    # 2. crisis events, optionally for given countries
    if has_crisis:
        codes = country_codes_from_prompt(prompt)
        cf_sql, cf_params = country_filter(codes) if codes else ("", [])
        cf = f"{cf_sql} AND " if cf_sql else ""
        return SqlPlan(
            f"SELECT {EVENT_COLS} FROM {tbl} WHERE {BASE_WHERE} AND {cf}ARTICLE_COUNT >= 3 "
            f"AND IMPACT_SCORE < -3 AND {df_sql} "
            f"ORDER BY ARTICLE_COUNT DESC, IMPACT_SCORE ASC LIMIT {fetch_limit}",
            cf_params + df_params, limit,
        )

    # 3. major / trending stories
    if has_major:
        return SqlPlan(
            f"SELECT {EVENT_COLS} FROM {tbl} WHERE {BASE_WHERE} AND ARTICLE_COUNT > 20 "
            f"AND {df_sql} ORDER BY ARTICLE_COUNT DESC LIMIT {fetch_limit}",
            df_params, limit,
        )

    # 4. top countries (before the generic aggregate)
    if 'top' in p and has_country_word:
        return SqlPlan(
            f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as EVENT_COUNT FROM {tbl} WHERE {BASE_WHERE} "
            f"AND {df_sql} GROUP BY ACTOR_COUNTRY_CODE ORDER BY EVENT_COUNT DESC LIMIT {limit}",
            df_params, limit, is_country_aggregate=True,
        )

    # 5. counts (how many, total), optionally for one country
    if qi['is_aggregate']:
        codes = country_codes_from_prompt(prompt)
        if codes:
            return SqlPlan(
                f"SELECT COUNT(*) as TOTAL_EVENTS FROM {tbl} WHERE {BASE_WHERE} "
                f"AND ACTOR_COUNTRY_CODE = ? AND {df_sql} LIMIT {limit}",
                [codes[0]] + df_params, limit, is_count_aggregate=True,
                country_filter_name=get_country(codes[0]) or codes[0],
            )
        return SqlPlan(
            f"SELECT COUNT(*) as TOTAL_EVENTS FROM {tbl} WHERE {BASE_WHERE} AND {df_sql} LIMIT {limit}",
            df_params, limit, is_count_aggregate=True,
        )

    # 6. default: events, optionally for given countries
    codes = country_codes_from_prompt(prompt)
    if codes:
        cf_sql, cf_params = country_filter(codes)
        return SqlPlan(
            f"SELECT {EVENT_COLS} FROM {tbl} WHERE {BASE_WHERE} AND {cf_sql} "
            f"AND ARTICLE_COUNT > {ARTICLE_THRESHOLD} AND {df_sql} "
            f"ORDER BY ARTICLE_COUNT DESC, DATE DESC LIMIT {fetch_limit}",
            cf_params + df_params, limit,
        )
    return SqlPlan(
        f"SELECT {EVENT_COLS} FROM {tbl} WHERE {BASE_WHERE} AND ARTICLE_COUNT > {ARTICLE_THRESHOLD} "
        f"AND {df_sql} ORDER BY ARTICLE_COUNT DESC LIMIT {fetch_limit}",
        df_params, limit,
    )
