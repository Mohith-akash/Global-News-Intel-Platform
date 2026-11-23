"""Dashboard CSS, kept in assets/style.css."""
from pathlib import Path

import streamlit as st

_CSS = (Path(__file__).resolve().parent.parent / "assets" / "style.css").read_text(encoding="utf-8")


def inject_css():
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
