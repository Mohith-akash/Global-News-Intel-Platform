"""
Database query functions for GDELT platform.

All dashboard data is fetched by ONE cached bundle (_dashboard_bundle) that
runs every query in a single worker subprocess — one python spawn and one
MotherDuck handshake per cache refresh instead of one per query, which is
what made cold page loads render piece by piece for 30-60s. The public
get_* functions just pick their slice out of the bundle, keeping the old
call sites unchanged.

SQL safety note: f-strings here interpolate only internally-generated values
(dates from datetime.now() via get_dates(), and a trusted table name `t`
supplied by the caller). No user-controlled input reaches these queries, so
f-string formatting is safe. If user input is ever added, switch to DuckDB
parameter binding ($1, $2, ...) instead.
"""

import datetime
import streamlit as st

from src.database import safe_query_batch, retry_cache_race
from src.utils import get_dates


@retry_cache_race
@st.cache_data(ttl=14400)
def _dashboard_bundle(t):
    """Every dashboard query, one worker call. Cached 4h."""
    dates = get_dates()
    three_days = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime('%Y%m%d')

    return safe_query_batch({
        # A real COUNT(*), not duckdb_tables().estimated_size. The estimate was
        # meant to dodge a slow full scan, but it is a lazily-refreshed catalog
        # statistic: it sat frozen for a day at a time (so the headline number
        # looked stuck) and read 2,920 rows high. COUNT(*) on a columnar store
        # comes off column metadata - measured at 0.03s against 24M rows,
        # actually faster than the catalog lookup it replaced.
        'total': (f"SELECT COUNT(*) AS total FROM {t}", None),
        'weekly': (f"""
            SELECT
                COUNT(*) as recent,
                SUM(CASE WHEN ABS(IMPACT_SCORE) > 6 THEN 1 ELSE 0 END) as critical
            FROM {t}
            WHERE DATE >= '{dates['week_ago']}'
        """, None),
        'hotspot': (f"""
            SELECT ACTOR_COUNTRY_CODE, COUNT(*) as c FROM {t}
            WHERE DATE >= '{dates['week_ago']}' AND ACTOR_COUNTRY_CODE IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC LIMIT 1
        """, None),
        'alerts': (f"""
            SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t}
            WHERE DATE >= '{three_days}' AND IMPACT_SCORE < -4 AND MAIN_ACTOR IS NOT NULL
            ORDER BY IMPACT_SCORE LIMIT 15
        """, None),
        'trending': (f"""
            SELECT DATE, NEWS_LINK, HEADLINE, HEADLINE_AI, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT
            FROM {t}
            WHERE DATE >= '{dates['week_ago']}'
              AND ARTICLE_COUNT > 3
              AND NEWS_LINK IS NOT NULL
              AND ACTOR_COUNTRY_CODE IS NOT NULL
              AND HEADLINE IS NOT NULL
              AND LENGTH(HEADLINE) > 20
            ORDER BY ARTICLE_COUNT DESC, DATE DESC
            LIMIT 500
        """, None),
        'feed': (f"""
            SELECT DATE, NEWS_LINK, HEADLINE, HEADLINE_AI, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT
            FROM {t}
            WHERE DATE >= '{dates['week_ago']}'
              AND NEWS_LINK IS NOT NULL
              AND ACTOR_COUNTRY_CODE IS NOT NULL
              AND HEADLINE IS NOT NULL
              AND LENGTH(HEADLINE) > 20
            ORDER BY DATE DESC, ARTICLE_COUNT DESC
            LIMIT 500
        """, None),
        'countries': (f"""
            SELECT ACTOR_COUNTRY_CODE as country, COUNT(*) as events FROM {t}
            WHERE DATE >= '{dates['month_ago']}' AND ACTOR_COUNTRY_CODE IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC
        """, None),
        'timeseries': (f"""
            SELECT DATE, COUNT(*) as events,
                SUM(CASE WHEN IMPACT_SCORE < -2 THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN IMPACT_SCORE > 2 THEN 1 ELSE 0 END) as positive
            FROM {t} WHERE DATE >= '{dates['month_ago']}' GROUP BY 1 ORDER BY 1
        """, None),
        'sentiment': (f"""
            SELECT AVG(IMPACT_SCORE) as avg,
                SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as neg,
                SUM(CASE WHEN IMPACT_SCORE > 3 THEN 1 ELSE 0 END) as pos,
                COUNT(*) as total
            FROM {t} WHERE DATE >= '{dates['week_ago']}' AND IMPACT_SCORE IS NOT NULL
        """, None),
        'actors': (f"""
            SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, COUNT(*) as events, AVG(IMPACT_SCORE) as avg_impact
            FROM {t} WHERE DATE >= '{dates['week_ago']}' AND MAIN_ACTOR IS NOT NULL AND LENGTH(MAIN_ACTOR) > 3
            GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10
        """, None),
        'distribution': (f"""
            SELECT CASE
                WHEN IMPACT_SCORE < -5 THEN 'Crisis'
                WHEN IMPACT_SCORE < -2 THEN 'Negative'
                WHEN IMPACT_SCORE < 2 THEN 'Neutral'
                WHEN IMPACT_SCORE < 5 THEN 'Positive'
                ELSE 'Very Positive' END as cat, COUNT(*) as cnt
            FROM {t} WHERE DATE >= '{dates['week_ago']}' AND IMPACT_SCORE IS NOT NULL GROUP BY 1
        """, None),
    })


def get_metrics(_c, t):
    b = _dashboard_bundle(t)
    total_df, weekly, hs = b['total'], b['weekly'], b['hotspot']
    total = None
    if not total_df.empty and total_df.iloc[0]['total']:
        total = int(total_df.iloc[0]['total'])
    return {
        'total': total,
        'recent': int(weekly.iloc[0]['recent'] or 0) if not weekly.empty else 0,
        'critical': int(weekly.iloc[0]['critical'] or 0) if not weekly.empty else 0,
        'hotspot': hs.iloc[0]['ACTOR_COUNTRY_CODE'] if not hs.empty else None,
    }


def get_alerts(_c, t):
    return _dashboard_bundle(t)['alerts']


def get_trending(_c, t):
    return _dashboard_bundle(t)['trending']


def get_feed(_c, t):
    return _dashboard_bundle(t)['feed']


def get_countries(_c, t):
    return _dashboard_bundle(t)['countries']


def get_timeseries(_c, t):
    return _dashboard_bundle(t)['timeseries']


def get_sentiment(_c, t):
    return _dashboard_bundle(t)['sentiment']


def get_actors(_c, t):
    return _dashboard_bundle(t)['actors']


def get_distribution(_c, t):
    return _dashboard_bundle(t)['distribution']
