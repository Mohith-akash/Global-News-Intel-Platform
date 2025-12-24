"""
Database query functions for GDELT platform.
"""

import datetime
import streamlit as st

from src.database import safe_query
from src.utils import get_dates


@st.cache_data(ttl=300)
def get_metrics(_c, t):
    dates = get_dates()
    df = safe_query(_c, f"""
        SELECT COUNT(*) as total,
            SUM(CASE WHEN DATE >= '{dates['week_ago']}' THEN 1 ELSE 0 END) as recent,
            SUM(CASE WHEN ABS(IMPACT_SCORE) > 6 AND DATE >= '{dates['week_ago']}' THEN 1 ELSE 0 END) as critical
        FROM {t}
    """)
    hs = safe_query(_c, f"""
        SELECT ACTOR_COUNTRY_CODE, COUNT(*) as c FROM {t} 
        WHERE DATE >= '{dates['week_ago']}' AND ACTOR_COUNTRY_CODE IS NOT NULL 
        GROUP BY 1 ORDER BY 2 DESC LIMIT 1
    """)
    return {
        'total': df.iloc[0]['total'] if not df.empty else 0,
        'recent': df.iloc[0]['recent'] if not df.empty else 0,
        'critical': df.iloc[0]['critical'] if not df.empty else 0,
        'hotspot': hs.iloc[0]['ACTOR_COUNTRY_CODE'] if not hs.empty else None
    }


@st.cache_data(ttl=300)
def get_alerts(_c, t):
    three_days = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime('%Y%m%d')
    return safe_query(_c, f"""
        SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t} 
        WHERE DATE >= '{three_days}' AND IMPACT_SCORE < -4 AND MAIN_ACTOR IS NOT NULL 
        ORDER BY IMPACT_SCORE LIMIT 15
    """)



@st.cache_data(ttl=300)
def get_trending(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT DATE, NEWS_LINK, HEADLINE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND ARTICLE_COUNT > 3 AND NEWS_LINK IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL
        ORDER BY ARTICLE_COUNT DESC LIMIT 500
    """)


@st.cache_data(ttl=300)
def get_feed(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT DATE, NEWS_LINK, HEADLINE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND NEWS_LINK IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL
        ORDER BY DATE DESC LIMIT 500
    """)


@st.cache_data(ttl=300)
def get_countries(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT ACTOR_COUNTRY_CODE as country, COUNT(*) as events FROM {t} 
        WHERE DATE >= '{dates['month_ago']}' AND ACTOR_COUNTRY_CODE IS NOT NULL 
        GROUP BY 1 ORDER BY 2 DESC
    """)


@st.cache_data(ttl=300)
def get_timeseries(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT DATE, COUNT(*) as events, 
            SUM(CASE WHEN IMPACT_SCORE < -2 THEN 1 ELSE 0 END) as negative, 
            SUM(CASE WHEN IMPACT_SCORE > 2 THEN 1 ELSE 0 END) as positive 
        FROM {t} WHERE DATE >= '{dates['month_ago']}' GROUP BY 1 ORDER BY 1
    """)


@st.cache_data(ttl=300)
def get_sentiment(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT AVG(IMPACT_SCORE) as avg, 
            SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as neg, 
            SUM(CASE WHEN IMPACT_SCORE > 3 THEN 1 ELSE 0 END) as pos, 
            COUNT(*) as total 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND IMPACT_SCORE IS NOT NULL
    """)


@st.cache_data(ttl=300)
def get_actors(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, COUNT(*) as events, AVG(IMPACT_SCORE) as avg_impact 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND MAIN_ACTOR IS NOT NULL AND LENGTH(MAIN_ACTOR) > 3 
        GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10
    """)


@st.cache_data(ttl=300)
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
