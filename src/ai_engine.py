"""
AI/LLM setup and query engine for GDELT platform.
Imports are deferred to avoid memory issues on Streamlit Cloud startup.
"""

import os
import logging
import streamlit as st

from src.config import CEREBRAS_MODEL
from src.database import get_db, detect_table

logger = logging.getLogger("gdelt")


@st.cache_resource
def get_ai_engine(_engine):
    """Set up AI query engine."""
    # Import inside function to avoid memory issues on startup
    from llama_index.llms.cerebras import Cerebras
    from llama_index.core import SQLDatabase, Settings
    try:
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            return None
        
        llm = Cerebras(api_key=api_key, model=CEREBRAS_MODEL, temperature=0.1)
        Settings.llm = llm
        
        conn = get_db()
        main_table = detect_table(conn)
        sql_db = SQLDatabase(_engine, include_tables=[main_table], sample_rows_in_table_info=0)
        
        return sql_db
    except Exception as e:
        logger.error(f"AI initialization failed: {e}")
        return None


@st.cache_resource
def get_query_engine(_sql_db):
    """Create AI query engine."""
    # Import inside function to avoid memory issues on startup
    from llama_index.core.query_engine import NLSQLTableQueryEngine
    if not _sql_db:
        return None
    try:
        tables = list(_sql_db.get_usable_table_names())
        target = next((t for t in tables if 'event' in t.lower()), tables[0] if tables else None)
        if target:
            return NLSQLTableQueryEngine(sql_database=_sql_db, tables=[target])
        return NLSQLTableQueryEngine(sql_database=_sql_db)
    except Exception:
        return None


@st.cache_resource
def get_cerebras_llm():
    """Initialize Cerebras LLM."""
    # Import inside function to avoid memory issues on startup
    from llama_index.llms.cerebras import Cerebras
    try:
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            return None
        return Cerebras(api_key=api_key, model=CEREBRAS_MODEL, temperature=0.1)
    except:
        return None
