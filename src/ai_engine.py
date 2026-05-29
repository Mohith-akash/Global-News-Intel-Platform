"""
AI/LLM setup for GDELT platform.
Imports are deferred to avoid memory issues on Streamlit Cloud startup.
"""

import os
import logging
import streamlit as st

from src.config import CEREBRAS_MODEL

logger = logging.getLogger("gdelt")

# Global flag to track AI availability
AI_AVAILABLE = False

# Try to import the Cerebras LLM client at module level for early failure detection.
try:
    from llama_index.llms.cerebras import Cerebras
    AI_AVAILABLE = True
    logger.info("AI features initialized successfully")
except ImportError as e:
    logger.warning(f"AI features unavailable - cerebras client import failed: {e}")
except Exception as e:
    logger.error(f"Unexpected error loading AI dependencies: {e}", exc_info=True)


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
