"""
Components package for GDELT dashboard.
"""

from components.render import (
    render_header,
    render_metrics,
    render_ticker,
    render_sentiment,
    render_actors,
    render_distribution,
    render_countries,
    render_trending,
    render_feed,
    render_timeseries
)
from components.ai_chat import render_ai_chat
from components.about import render_about
from components.emotions import render_emotions_tab

__all__ = [
    'render_header',
    'render_metrics',
    'render_ticker',
    'render_sentiment',
    'render_actors',
    'render_distribution',
    'render_countries',
    'render_trending',
    'render_feed',
    'render_timeseries',
    'render_ai_chat',
    'render_about',
    'render_emotions_tab'
]

