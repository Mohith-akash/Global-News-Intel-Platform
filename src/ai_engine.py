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

# Global flag to track AI availability
AI_AVAILABLE = False

# Try to import llama-index components at module level
# This provides early detection of import issues
try:
    from llama_index.llms.cerebras import Cerebras
    from llama_index.core import SQLDatabase, Settings
    from llama_index.core.query_engine import NLSQLTableQueryEngine
    AI_AVAILABLE = True
    logger.info("AI features initialized successfully")
except ImportError as e:
    logger.warning(f"AI features unavailable - llama-index import failed: {e}")
    Cerebras = None
    SQLDatabase = None
    Settings = None
    NLSQLTableQueryEngine = None
except Exception as e:
    logger.error(f"Unexpected error loading AI dependencies: {e}")
    Cerebras = None
    SQLDatabase = None
    Settings = None
    NLSQLTableQueryEngine = None


@st.cache_resource
def get_ai_engine(_engine):
    """Set up AI query engine."""
    if not AI_AVAILABLE:
        logger.warning("AI features unavailable - cannot initialize engine")
        return None
    
    try:
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            logger.warning("CEREBRAS_API_KEY not found")
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
    if not AI_AVAILABLE:
        logger.warning("AI features unavailable - cannot create query engine")
        return None
    
    if not _sql_db:
        return None
    
    try:
        tables = list(_sql_db.get_usable_table_names())
        target = next((t for t in tables if 'event' in t.lower()), tables[0] if tables else None)
        if target:
            return NLSQLTableQueryEngine(sql_database=_sql_db, tables=[target])
        return NLSQLTableQueryEngine(sql_database=_sql_db)
    except Exception as e:
        logger.error(f"Query engine creation failed: {e}")
        return None


@st.cache_resource
def get_cerebras_llm():
    """Initialize Cerebras LLM."""
    if not AI_AVAILABLE:
        logger.warning("AI features unavailable - cannot initialize Cerebras LLM")
        return None
    
    try:
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            logger.warning("CEREBRAS_API_KEY not found")
            return None
        return Cerebras(api_key=api_key, model=CEREBRAS_MODEL, temperature=0.1)
    except Exception as e:
        logger.error(f"Cerebras LLM initialization failed: {e}")
        return None
