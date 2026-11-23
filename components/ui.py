"""Small building blocks shared by the tabs: card headers, metrics, charts."""

import logging

import streamlit as st

from src.database import WarehouseUnavailable

GRID = 'rgba(30,58,95,0.3)'
MUTED = '#64748b'
TEXT = '#94a3b8'


def card_header(title, note=None):
    note_html = f'<span class="card-note">{note}</span>' if note else ''
    st.markdown(f'<div class="card-hdr"><span class="card-title">{title}</span>{note_html}</div>',
                unsafe_allow_html=True)


def metric(label, value, hint):
    st.metric(label, value)
    st.markdown(f'<div class="metric-hint">{hint}</div>', unsafe_allow_html=True)


def metric_row(cards):
    """cards: [(label, value, hint), ...] in equal columns."""
    for col, card in zip(st.columns(len(cards)), cards):
        with col:
            metric(*card)


def show_chart(fig, key, height, **layout):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **layout)
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch', key=key)


def section(render, *args):
    """Render one tab; if it fails, say so there and let the other tabs load."""
    try:
        render(*args)
    except WarehouseUnavailable:
        st.warning("The data warehouse is not reachable right now. Try again in a few minutes.")
    except Exception:
        logging.getLogger("gdelt").exception("%s failed", getattr(render, "__name__", "section"))
        st.error("This section couldn't load. The other tabs still work.")
