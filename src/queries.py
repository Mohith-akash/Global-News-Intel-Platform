"""
Database query functions for GDELT platform.

SQL safety note: f-strings here interpolate only internally-generated values
(dates from datetime.now() via get_dates(), and a trusted table name `t`
supplied by the caller). No user-controlled input reaches these queries, so
f-string formatting is safe. If user input is ever added, switch to DuckDB
parameter binding ($1, $2, ...) instead.
"""

import datetime
import streamlit as st

from src.database import safe_query, retry_cache_race
from src.utils import get_dates


@retry_cache_race
@st.cache_data(ttl=14400)
def _get_total_count(_c, t):
    """Read row count from catalog stats — instant, no full scan."""
    table_name = t.split('.')[-1]
    df = safe_query(_c, f"""
        SELECT estimated_size
        FROM duckdb_tables()
        WHERE table_name = '{table_name}'
        LIMIT 1
    """)
    if not df.empty and df.iloc[0]['estimated_size']:
        return int(df.iloc[0]['estimated_size'])
    # Fallback: skip the slow COUNT(*) and return a sentinel so UI shows cached value
    return None


@retry_cache_race
@st.cache_data(ttl=14400)
def _get_weekly_metrics(_c, t):
    """Week-filtered queries only — date pushdown keeps these fast."""
    dates = get_dates()
    df = safe_query(_c, f"""
        SELECT
            COUNT(*) as recent,
            SUM(CASE WHEN ABS(IMPACT_SCORE) > 6 THEN 1 ELSE 0 END) as critical
        FROM {t}
        WHERE DATE >= '{dates['week_ago']}'
    """)
    hs = safe_query(_c, f"""
        SELECT ACTOR_COUNTRY_CODE, COUNT(*) as c FROM {t}
        WHERE DATE >= '{dates['week_ago']}' AND ACTOR_COUNTRY_CODE IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC LIMIT 1
    """)
    return {
        'recent': int(df.iloc[0]['recent'] or 0) if not df.empty else 0,
        'critical': int(df.iloc[0]['critical'] or 0) if not df.empty else 0,
        'hotspot': hs.iloc[0]['ACTOR_COUNTRY_CODE'] if not hs.empty else None,
    }


def get_metrics(_c, t):
    total = _get_total_count(_c, t)
    weekly = _get_weekly_metrics(_c, t)
    return {'total': total, **weekly}


@retry_cache_race
@st.cache_data(ttl=14400)
def get_alerts(_c, t):
    three_days = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime('%Y%m%d')
    return safe_query(_c, f"""
        SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t} 
        WHERE DATE >= '{three_days}' AND IMPACT_SCORE < -4 AND MAIN_ACTOR IS NOT NULL 
        ORDER BY IMPACT_SCORE LIMIT 15
    """)



@retry_cache_race
@st.cache_data(ttl=14400)
def get_trending(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT DATE, NEWS_LINK, HEADLINE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT
        FROM {t}
        WHERE DATE >= '{dates['week_ago']}'
          AND ARTICLE_COUNT > 3
          AND NEWS_LINK IS NOT NULL
          AND ACTOR_COUNTRY_CODE IS NOT NULL
          AND HEADLINE IS NOT NULL
          AND LENGTH(HEADLINE) > 20
        ORDER BY ARTICLE_COUNT DESC, DATE DESC
        LIMIT 500
    """)


@retry_cache_race
@st.cache_data(ttl=14400)
def get_feed(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT DATE, NEWS_LINK, HEADLINE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT
        FROM {t}
        WHERE DATE >= '{dates['week_ago']}'
          AND NEWS_LINK IS NOT NULL
          AND ACTOR_COUNTRY_CODE IS NOT NULL
          AND HEADLINE IS NOT NULL
          AND LENGTH(HEADLINE) > 20
        ORDER BY DATE DESC, ARTICLE_COUNT DESC
        LIMIT 500
    """)


@retry_cache_race
@st.cache_data(ttl=14400)
def get_countries(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT ACTOR_COUNTRY_CODE as country, COUNT(*) as events FROM {t} 
        WHERE DATE >= '{dates['month_ago']}' AND ACTOR_COUNTRY_CODE IS NOT NULL 
        GROUP BY 1 ORDER BY 2 DESC
    """)


@retry_cache_race
@st.cache_data(ttl=14400)
def get_timeseries(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT DATE, COUNT(*) as events, 
            SUM(CASE WHEN IMPACT_SCORE < -2 THEN 1 ELSE 0 END) as negative, 
            SUM(CASE WHEN IMPACT_SCORE > 2 THEN 1 ELSE 0 END) as positive 
        FROM {t} WHERE DATE >= '{dates['month_ago']}' GROUP BY 1 ORDER BY 1
    """)


@retry_cache_race
@st.cache_data(ttl=14400)
def get_sentiment(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT AVG(IMPACT_SCORE) as avg, 
            SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as neg, 
            SUM(CASE WHEN IMPACT_SCORE > 3 THEN 1 ELSE 0 END) as pos, 
            COUNT(*) as total 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND IMPACT_SCORE IS NOT NULL
    """)


@retry_cache_race
@st.cache_data(ttl=14400)
def get_actors(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, COUNT(*) as events, AVG(IMPACT_SCORE) as avg_impact 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND MAIN_ACTOR IS NOT NULL AND LENGTH(MAIN_ACTOR) > 3 
        GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10
    """)


@retry_cache_race
@st.cache_data(ttl=14400)
def get_distribution(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT CASE 
            WHEN IMPACT_SCORE < -5 THEN 'Crisis' 
            WHEN IMPACT_SCORE < -2 THEN 'Negative' 
            WHEN IMPACT_SCORE < 2 THEN 'Neutral' 
            WHEN IMPACT_SCORE < 5 THEN 'Positive' 
            ELSE 'Very Positive' END as cat, COUNT(*) as cnt 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND IMPACT_SCORE IS NOT NULL GROUP BY 1
    """)
