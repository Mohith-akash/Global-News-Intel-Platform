"""
Core modules for GDELT platform.
"""

from src.config import REQUIRED_ENVS, CEREBRAS_MODEL, COUNTRY_ALIASES
from src.database import get_db, get_engine, detect_table, safe_query
from src.ai_engine import get_ai_engine, get_query_engine, get_cerebras_llm
from src.styles import inject_css
from src.utils import (
    get_country_code, get_dates, detect_query_type, 
    get_country, get_impact_label, get_intensity_label
)
from src.queries import (
    get_metrics, get_alerts, get_headlines, get_trending,
    get_feed, get_countries, get_timeseries, get_sentiment,
    get_actors, get_distribution
)
from src.data_processing import (
    clean_headline, enhance_headline, extract_headline, process_df
)
