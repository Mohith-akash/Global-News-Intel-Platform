"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🌐 GLOBAL NEWS INTELLIGENCE PLATFORM                       ║
║                                                                              ║
║  Real-Time Analytics Dashboard for GDELT                                     ║
║  (Global Database of Events, Language & Tone)                                ║
║                                                                              ║
║  Author: Mohith Akash                                                        ║
║  Purpose: Portfolio Project for AI/Data Engineering Roles                    ║
║                                                                              ║
║  WHAT IS GDELT?                                                              ║
║  GDELT monitors news media worldwide (TV, print, web) in 100+ languages,    ║
║  translating and processing articles to identify events, people, emotions,   ║
║  and themes. It updates every 15 minutes with global coverage.              ║
║                                                                              ║
║  ARCHITECTURE:                                                               ║
║  GDELT API → GitHub Actions (every 30 min) → MotherDuck (DuckDB cloud)      ║
║           → Gemini AI (natural language queries) → Streamlit (this app)     ║
║                                                                              ║
║  KEY TECHNOLOGIES:                                                           ║
║  - Dagster: Data pipeline orchestration                                      ║
║  - dbt: Data transformation                                                  ║
║  - MotherDuck: Serverless DuckDB (migrated from Snowflake for cost savings) ║
║  - LlamaIndex: LLM framework for text-to-SQL                                ║
║  - Google Gemini: LLM for natural language understanding                    ║
║  - Streamlit: Interactive web dashboard                                     ║
║  - Plotly: Data visualization                                               ║
║                                                                              ║
║  COST OPTIMIZATION:                                                          ║
║  This project demonstrates building production-grade analytics on $0/month  ║
║  using free tiers: MotherDuck free tier + Gemini API free tier              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
# These are all the Python libraries we need for the application

import streamlit as st                    # Web framework for the dashboard
import os                                 # Access environment variables
import pandas as pd                       # Data manipulation
import plotly.express as px              # Quick charts
import plotly.graph_objects as go        # Detailed chart customization
from plotly.subplots import make_subplots # Multiple charts in one figure
from dotenv import load_dotenv           # Load .env file for local development
from llama_index.llms.gemini import Gemini           # Google's AI model
from llama_index.embeddings.gemini import GeminiEmbedding  # Text embeddings
from llama_index.core import SQLDatabase, Settings   # Database connections
from llama_index.core.query_engine import NLSQLTableQueryEngine  # Text-to-SQL
from sqlalchemy import create_engine, inspect        # Database engine
import datetime                          # Date/time handling
import pycountry                         # Country code lookups
import logging                           # Error logging
import re                                # Regular expressions for text parsing
from urllib.parse import urlparse, unquote  # URL parsing for headlines
import duckdb                            # Direct database connection

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: APP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Set up the Streamlit page settings - this MUST be the first Streamlit command

st.set_page_config(
    page_title="Global News Intelligence | GDELT Analytics",  # Browser tab title
    page_icon="🌐",                      # Browser tab icon
    layout="wide",                        # Use full screen width
    initial_sidebar_state="collapsed",    # Hide sidebar by default
    menu_items={
        'Get Help': 'https://github.com/Mohith-akash/global-news-intel-platform',
        'Report a bug': 'https://github.com/Mohith-akash/global-news-intel-platform/issues',
        'About': "Real-time global news analytics powered by GDELT, MotherDuck & Gemini AI"
    }
)

# Load environment variables from .env file (for local development)
# In production (Streamlit Cloud), these come from the Secrets manager
load_dotenv()

# Set up logging - helps debug issues in production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gdelt_platform")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ENVIRONMENT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
# Check that all required API keys and tokens are configured

REQUIRED_ENVS = ["MOTHERDUCK_TOKEN", "GOOGLE_API_KEY"]
missing = [k for k in REQUIRED_ENVS if not os.getenv(k)]

if missing:
    # Show error and stop the app if credentials are missing
    st.error(f"❌ SYSTEM CRITICAL: Missing environment variables: {', '.join(missing)}")
    st.info("Please configure your secrets in the Streamlit Cloud dashboard.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CONSTANTS & DATE CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Define reusable constants for the entire application

# AI Model configuration - using Google's Gemini models
GEMINI_MODEL = "models/gemini-2.5-flash-preview-09-2025"  # For text generation
GEMINI_EMBED_MODEL = "models/embedding-001"               # For text embeddings

# Pre-calculate common date filters
# GDELT uses YYYYMMDD format as strings, not actual date objects
NOW = datetime.datetime.now()
TODAY = f"'{NOW.strftime('%Y%m%d')}'"
YESTERDAY = f"'{(NOW - datetime.timedelta(days=1)).strftime('%Y%m%d')}'"
TWO_DAYS_AGO = f"'{(NOW - datetime.timedelta(days=2)).strftime('%Y%m%d')}'"
WEEK_AGO = f"'{(NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')}'"
MONTH_AGO = f"'{(NOW - datetime.timedelta(days=30)).strftime('%Y%m%d')}'"

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════
# This function injects custom CSS to transform the default Streamlit look
# into a professional "intelligence dashboard" aesthetic

def inject_custom_css():
    """
    Inject custom CSS for the dark theme intelligence dashboard design.
    
    Design Philosophy:
    - Dark theme reduces eye strain and looks professional
    - Cyan/teal accents provide visual hierarchy
    - JetBrains Mono font for data/code (monospace = easy to read numbers)
    - Inter font for body text (clean, modern, readable)
    - Subtle animations add polish without being distracting
    """
    st.markdown("""
    <style>
        /* ═══════════════ FONT IMPORTS ═══════════════ */
        /* JetBrains Mono: Great for code, numbers, technical data */
        /* Inter: Clean, modern font for body text */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* ═══════════════ CSS VARIABLES ═══════════════ */
        /* Define colors once, use everywhere - makes theming easy */
        :root {
            /* Background colors - dark blues */
            --bg-primary: #0a0e17;      /* Darkest - main background */
            --bg-secondary: #0d1320;    /* Slightly lighter */
            --bg-card: #111827;         /* Card backgrounds */
            --bg-elevated: #1a2332;     /* Hover states, elevated elements */
            
            /* Border colors */
            --border-color: #1e3a5f;    /* Default borders */
            --border-glow: #0ea5e9;     /* Glowing/active borders */
            
            /* Text colors */
            --text-primary: #e2e8f0;    /* Main text - almost white */
            --text-secondary: #94a3b8;  /* Secondary text - gray */
            --text-muted: #64748b;      /* Subtle text - darker gray */
            
            /* Accent colors - used for highlights, buttons, links */
            --accent-blue: #0ea5e9;     /* Primary accent */
            --accent-cyan: #06b6d4;     /* Secondary accent */
            --accent-emerald: #10b981;  /* Success/positive */
            --accent-amber: #f59e0b;    /* Warning */
            --accent-red: #ef4444;      /* Danger/negative */
            --accent-purple: #8b5cf6;   /* AI/special features */
            
            /* Gradients for visual interest */
            --gradient-blue: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
            --gradient-danger: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            --gradient-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }
        
        /* ═══════════════ GLOBAL STYLES ═══════════════ */
        /* Apply dark theme to entire app with subtle gradient background */
        .stApp {
            background: var(--bg-primary);
            background-image: 
                radial-gradient(ellipse at top, rgba(14, 165, 233, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(6, 182, 212, 0.03) 0%, transparent 50%);
        }
        
        /* Hide Streamlit's default UI elements for cleaner look */
        header[data-testid="stHeader"] { display: none; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stDeployButton { display: none; }
        div[data-testid="stToolbar"] { display: none; }
        div[data-testid="stDecoration"] { display: none; }
        
        /* ═══════════════ TYPOGRAPHY ═══════════════ */
        /* Set default fonts for all text */
        html, body, .stApp, .stMarkdown, p, span, div {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-primary);
        }
        
        /* Headers use monospace font for techy look */
        h1, h2, h3, .header-title {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        
        /* Code blocks use monospace */
        code, pre, .mono {
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* ═══════════════ CONTAINER LAYOUT ═══════════════ */
        .block-container {
            padding: 1.5rem 2rem 3rem 2rem;
            max-width: 100%;
        }
        
        /* ═══════════════ HEADER SECTION ═══════════════ */
        /* The main title bar at top of page */
        .command-header {
            background: linear-gradient(180deg, var(--bg-secondary) 0%, transparent 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 0 1.5rem 0;
            margin-bottom: 1.5rem;
        }
        
        .header-grid {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 1.5rem;
        }
        
        .logo-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .logo-icon {
            font-size: 2.5rem;
            filter: drop-shadow(0 0 10px rgba(14, 165, 233, 0.5));
        }
        
        .logo-text {
            display: flex;
            flex-direction: column;
        }
        
        .logo-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin: 0;
            line-height: 1.2;
        }
        
        .logo-subtitle {
            font-size: 0.7rem;
            color: var(--accent-cyan);
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }
        
        /* Live status indicator badge */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
        }
        
        /* Animated pulsing dot to show "live" status */
        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-emerald);
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
            box-shadow: 0 0 10px var(--accent-emerald);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(0.85); }
        }
        
        /* ═══════════════ METRIC CARDS ═══════════════ */
        /* These are the KPI boxes at top of dashboard */
        div[data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        /* Gradient accent bar at top of each metric card */
        div[data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-blue);
        }
        
        /* Hover effect - lift card up slightly */
        div[data-testid="stMetric"]:hover {
            border-color: var(--border-glow);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(14, 165, 233, 0.15);
        }
        
        /* Metric label styling */
        div[data-testid="stMetric"] label {
            color: var(--text-secondary);
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        
        /* Metric value styling - big bold number */
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: var(--text-primary);
            font-size: 1.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Metric delta (change indicator) */
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* ═══════════════ CUSTOM CARDS ═══════════════ */
        /* Reusable card component styling */
        .intel-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .intel-card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 4px 20px rgba(14, 165, 233, 0.1);
        }
        
        /* Card header with icon and title */
        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .card-icon {
            font-size: 1.25rem;
        }
        
        .card-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0;
        }
        
        /* ═══════════════ TABS - MOBILE OPTIMIZED ═══════════════ */
        /* Navigation tabs at top of main content */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 4px;
            border: 1px solid var(--border-color);
            overflow-x: auto;                    /* Enable horizontal scroll on mobile */
            -webkit-overflow-scrolling: touch;  /* Smooth scrolling on iOS */
            scrollbar-width: none;              /* Hide scrollbar (Firefox) */
        }
        
        /* Hide scrollbar (Chrome, Safari) */
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }
        
        /* Individual tab styling */
        .stTabs [data-baseweb="tab"] {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-secondary);
            background: transparent;
            border-radius: 6px;
            padding: 0.5rem 0.9rem;
            letter-spacing: 0.02em;
            white-space: nowrap;    /* Don't wrap tab text */
            flex-shrink: 0;         /* Don't shrink tabs */
            transition: all 0.2s ease;
        }
        
        /* Active/selected tab */
        .stTabs [aria-selected="true"] {
            background: var(--bg-elevated);
            color: var(--accent-cyan);
            border: 1px solid var(--border-color);
        }
        
        /* Hide default tab underline */
        .stTabs [data-baseweb="tab-highlight"] { display: none; }
        .stTabs [data-baseweb="tab-border"] { display: none; }
        
        /* ═══════════════ CHAT INTERFACE ═══════════════ */
        /* AI chat message bubbles */
        div[data-testid="stChatMessage"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
        }
        
        div[data-testid="stChatMessageContent"] p {
            color: var(--text-primary);
        }
        
        /* Chat input box */
        .stChatInput > div {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }
        
        .stChatInput input {
            background: transparent;
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }
        
        .stChatInput input::placeholder {
            color: var(--text-muted);
        }
        
        /* ═══════════════ DATAFRAMES ═══════════════ */
        /* Table styling with scroll indicator */
        div[data-testid="stDataFrame"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        
        /* "Scroll for more" hint at bottom right */
        div[data-testid="stDataFrame"]::after {
            content: '↓ scroll';
            position: absolute;
            bottom: 6px;
            right: 10px;
            font-size: 0.65rem;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
            background: rgba(17, 24, 39, 0.9);
            padding: 2px 8px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            pointer-events: none;
            z-index: 10;
        }
        
        div[data-testid="stDataFrame"] table {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
        }
        
        /* Table header styling */
        div[data-testid="stDataFrame"] th {
            background: var(--bg-elevated) !important;
            color: var(--text-secondary) !important;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* ═══════════════ BUTTONS ═══════════════ */
        .stButton > button {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            background: var(--bg-elevated);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
        }
        
        .stButton > button[kind="primary"] {
            background: var(--gradient-blue);
            border: none;
            color: white;
        }
        
        /* ═══════════════ EXPANDERS ═══════════════ */
        .streamlit-expanderHeader {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        .streamlit-expanderContent {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-top: none;
            border-radius: 0 0 8px 8px;
        }
        
        /* ═══════════════ SELECTBOX & INPUTS ═══════════════ */
        .stSelectbox > div > div {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        .stSelectbox [data-baseweb="select"] {
            background: var(--bg-secondary);
        }
        
        /* ═══════════════ PLOTLY CHARTS ═══════════════ */
        .js-plotly-plot {
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* ═══════════════ SCROLLBAR ═══════════════ */
        /* Custom scrollbar that matches theme */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-blue);
        }
        
        /* ═══════════════ DIVIDERS ═══════════════ */
        hr {
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 1.5rem 0;
        }
        
        /* ═══════════════ SPECIAL ELEMENTS ═══════════════ */
        /* Large highlighted stat number with gradient */
        .stat-highlight {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--gradient-blue);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Technology badge pills */
        .tech-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 0.4rem 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin: 0.25rem;
            transition: all 0.2s ease;
        }
        
        .tech-badge:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.2);
        }
        
        /* ═══════════════ MOBILE RESPONSIVE ═══════════════ */
        /* Adjust layout for smaller screens */
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 0.75rem 2rem 0.75rem;
            }
            
            .logo-title {
                font-size: 1rem !important;
            }
            
            .logo-subtitle {
                display: none;  /* Hide subtitle on mobile */
            }
            
            .card-title {
                font-size: 0.7rem !important;
            }
            
            /* Smaller metrics on mobile */
            div[data-testid="stMetric"] {
                padding: 0.6rem !important;
            }
            
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
                font-size: 1.1rem !important;
            }
            
            div[data-testid="stMetric"] label {
                font-size: 0.6rem !important;
            }
            
            .header-grid {
                gap: 0.5rem !important;
            }
            
            .status-badge {
                padding: 0.25rem 0.5rem !important;
                font-size: 0.65rem !important;
            }
        }
        
        /* ═══════════════ ANIMATED NEWS TICKER ═══════════════ */
        /* This creates the scrolling alert bar */
        .ticker-container {
            background: linear-gradient(90deg, 
                rgba(239, 68, 68, 0.15) 0%, 
                rgba(239, 68, 68, 0.08) 50%,
                rgba(239, 68, 68, 0.15) 100%);
            border-left: 4px solid #ef4444;
            border-radius: 0 8px 8px 0;
            padding: 0.6rem 0;
            overflow: hidden;
            position: relative;
            margin: 0.5rem 0;
        }
        
        /* Fixed "LIVE" label on left side */
        .ticker-label {
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            background: linear-gradient(90deg, 
                rgba(127, 29, 29, 0.98) 0%, 
                rgba(127, 29, 29, 0.95) 60%, 
                transparent 100%);
            padding: 0.6rem 1.25rem 0.6rem 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 600;
            color: #ef4444;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            z-index: 2;
            min-width: 85px;
        }
        
        /* Animated pulsing red dot */
        .ticker-dot {
            width: 7px;
            height: 7px;
            background: #ef4444;
            border-radius: 50%;
            animation: ticker-pulse 1s ease-in-out infinite;
            box-shadow: 0 0 8px #ef4444;
        }
        
        @keyframes ticker-pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.7); }
        }
        
        /* Scrolling text content */
        .ticker-content {
            display: inline-block;
            white-space: nowrap;
            padding-left: 95px;  /* Leave space for LIVE label */
            animation: ticker-scroll 40s linear infinite;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #fca5a5;
        }
        
        /* Pause animation on hover so user can read */
        .ticker-content:hover {
            animation-play-state: paused;
        }
        
        /* Scroll animation - starts visible, scrolls left */
        @keyframes ticker-scroll {
            0% { transform: translateX(0%); }
            100% { transform: translateX(-50%); }
        }
        
        /* ═══════════════ MOBILE TAB SWIPE HINT ═══════════════ */
        .swipe-hint {
            text-align: center;
            padding: 0.4rem;
            color: var(--accent-cyan);
            font-size: 0.7rem;
            font-family: 'JetBrains Mono', monospace;
            display: none;  /* Hidden by default */
            animation: hint-bounce 2s ease-in-out infinite;
        }
        
        /* Only show on mobile */
        @media (max-width: 768px) {
            .swipe-hint {
                display: block;
            }
        }
        
        @keyframes hint-bounce {
            0%, 100% { transform: translateX(0); opacity: 0.7; }
            50% { transform: translateX(5px); opacity: 1; }
        }
        
        /* ═══════════════ GLOW EFFECTS ═══════════════ */
        /* Subtle glow on important elements */
        .glow-cyan {
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
        }
        
        .glow-emerald {
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
        }
        
        .glow-red {
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════
# Functions to connect to MotherDuck (cloud DuckDB)

@st.cache_resource  # Cache the connection so we don't reconnect on every page refresh
def get_db_connection():
    """
    Create a direct DuckDB connection to MotherDuck.
    
    MotherDuck is a serverless cloud version of DuckDB.
    We migrated from Snowflake to MotherDuck to save costs
    (MotherDuck has a generous free tier).
    
    Returns:
        duckdb.Connection: A read-only connection to the database
    """
    token = os.getenv("MOTHERDUCK_TOKEN")
    return duckdb.connect(f'md:gdelt_db?motherduck_token={token}', read_only=True)


@st.cache_resource  # Cache this too - SQLAlchemy engine is expensive to create
def get_sql_engine():
    """
    Create a SQLAlchemy engine for LlamaIndex.
    
    LlamaIndex needs a SQLAlchemy engine to introspect the database
    schema and generate SQL queries from natural language.
    
    Returns:
        sqlalchemy.Engine: Database engine for LlamaIndex
    """
    token = os.getenv("MOTHERDUCK_TOKEN")
    connection_string = f"duckdb:///md:gdelt_db?motherduck_token={token}"
    return create_engine(connection_string)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: AI/LLM SETUP (LLAMAINDEX + GEMINI)
# ═══════════════════════════════════════════════════════════════════════════════
# Configure the AI components for natural language to SQL conversion

@st.cache_resource  # Cache AI model initialization
def get_ai_engine(_sql_engine):
    """
    Initialize LlamaIndex with Gemini for text-to-SQL capabilities.
    
    This sets up:
    1. Gemini LLM - For understanding queries and generating SQL
    2. Gemini Embeddings - For semantic understanding (not heavily used here)
    3. SQL Database - Wrapper around our MotherDuck connection
    
    Args:
        _sql_engine: SQLAlchemy engine (underscore prefix tells Streamlit not to hash it)
    
    Returns:
        SQLDatabase: LlamaIndex database object, or None if setup fails
    """
    try:
        # Initialize Google Gemini as our LLM
        llm = Gemini(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model=GEMINI_MODEL,
            temperature=0.1  # Low temperature = more focused, less creative responses
        )
        
        # Initialize embeddings (for semantic search if needed)
        embed = GeminiEmbedding(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model_name=GEMINI_EMBED_MODEL
        )
        
        # Set these as global defaults for LlamaIndex
        Settings.llm = llm
        Settings.embed_model = embed
        
        # Create SQL Database wrapper
        # We only expose the EVENTS_DAGSTER table to the AI
        # This prevents the AI from accessing other tables or system tables
        return SQLDatabase(_sql_engine, include_tables=["EVENTS_DAGSTER"])
        
    except Exception as e:
        logger.error(f"Failed to initialize AI engine: {e}")
        return None


@st.cache_resource  # Cache the query engine
def get_query_engine(_engine):
    """
    Create the natural language to SQL query engine.
    
    This is the main component that converts user questions like
    "What conflicts happened in Europe this week?" into SQL queries.
    
    Args:
        _engine: LlamaIndex SQLDatabase object
    
    Returns:
        NLSQLTableQueryEngine: Query engine, or None if setup fails
    """
    if _engine is None:
        return None
    
    try:
        return NLSQLTableQueryEngine(
            sql_database=_engine,
            tables=["EVENTS_DAGSTER"],  # Only allow queries against this table
        )
    except Exception as e:
        logger.error(f"Failed to create query engine: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions used throughout the application

def safe_query(conn, sql):
    """
    Execute a SQL query safely with error handling.
    
    This wrapper ensures that if a query fails, we return an empty
    DataFrame instead of crashing the entire app.
    
    Args:
        conn: Database connection
        sql: SQL query string
    
    Returns:
        pd.DataFrame: Query results, or empty DataFrame on error
    """
    try:
        return conn.execute(sql).df()
    except Exception as e:
        logger.error(f"Query failed: {e}\nSQL: {sql[:100]}...")  # Log first 100 chars
        return pd.DataFrame()  # Return empty DataFrame on error


def get_country_name(code):
    """
    Convert a 2-letter country code to full country name.
    
    Uses pycountry library for lookups.
    Example: 'US' -> 'United States', 'GB' -> 'United Kingdom'
    
    Args:
        code: 2-letter ISO country code
    
    Returns:
        str: Full country name, or original code if lookup fails
    """
    if not code:
        return "Unknown"
    
    code = str(code)
    
    if len(code) != 2:
        return code
        
    try:
        country = pycountry.countries.get(alpha_2=code)
        return country.name if country else code
    except:
        return code


def format_headline(url, actor=None):
    """
    Extract a readable headline from a news URL.
    
    News URLs often contain the headline in the path, like:
    https://example.com/news/2024/russia-ukraine-peace-talks
    
    This function extracts "Russia Ukraine Peace Talks" from that URL.
    
    Args:
        url: Full URL string
        actor: Optional actor name to use if URL parsing fails
    
    Returns:
        str: Extracted headline or fallback text
    """
    if not url:
        return actor if actor else "News Event"
    
    try:
        # Parse the URL and get the path
        parsed = urlparse(url)
        path = unquote(parsed.path)  # Decode URL-encoded characters
        
        # Get the last meaningful segment of the path
        segments = [s for s in path.split('/') if s and len(s) > 3]
        
        if segments:
            # Take the last segment (usually contains the headline)
            headline = segments[-1]
            
            # Clean up: remove file extensions, replace separators with spaces
            headline = re.sub(r'\.(html?|php|aspx?)$', '', headline)
            headline = re.sub(r'[-_]', ' ', headline)
            
            # Title case and limit length
            headline = headline.title()[:80]
            
            if len(headline) > 10:
                return headline
        
        # Fallback: use domain name
        return parsed.netloc.replace('www.', '').split('.')[0].title()
        
    except:
        return actor if actor else "News Event"


def format_number(num):
    """
    Format large numbers with K/M suffixes for display.
    
    Examples:
        1234 -> "1.2K"
        1234567 -> "1.2M"
        123 -> "123"
    
    Args:
        num: Number to format
    
    Returns:
        str: Formatted number string
    """
    if num is None:
        return "0"
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    if num >= 1000:
        return f"{num/1000:.1f}K"
    return str(int(num))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: DATA FETCHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
# These functions fetch data from the database for different dashboard components
# All use @st.cache_data to cache results and reduce database load

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_dashboard_metrics(_conn):
    """
    Fetch the main KPI metrics for the dashboard header.
    
    Metrics:
    - Total events in database
    - Events from last 7 days
    - Critical alerts (high impact scores)
    - Most active country (hotspot)
    
    Args:
        _conn: Database connection (underscore = don't hash for caching)
    
    Returns:
        dict: Dictionary of metric values
    """
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    # Count all events and recent events
    df = safe_query(_conn, f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN DATE >= '{week_ago}' THEN 1 ELSE 0 END) as recent,
            SUM(CASE WHEN ABS(IMPACT_SCORE) > 6 AND DATE >= '{week_ago}' THEN 1 ELSE 0 END) as critical
        FROM EVENTS_DAGSTER
    """)
    
    # Find the most active country this week
    hotspot_df = safe_query(_conn, f"""
        SELECT ACTOR_COUNTRY_CODE, COUNT(*) as cnt
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{week_ago}'
        AND ACTOR_COUNTRY_CODE IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC LIMIT 1
    """)
    
    return {
        'total': df.iloc[0]['total'] if not df.empty else 0,
        'recent': df.iloc[0]['recent'] if not df.empty else 0,
        'critical': df.iloc[0]['critical'] if not df.empty else 0,
        'hotspot': hotspot_df.iloc[0]['ACTOR_COUNTRY_CODE'] if not hotspot_df.empty else 'N/A'
    }


@st.cache_data(ttl=600)
def get_alert_events(_conn):
    """
    Fetch high-impact events for the alert ticker.
    
    These are events with high negative impact scores,
    indicating conflicts, crises, or significant negative events.
    
    Returns events from last 3 days with impact score < -4
    """
    three_days = (NOW - datetime.timedelta(days=3)).strftime('%Y%m%d')
    
    return safe_query(_conn, f"""
        SELECT 
            DATE,
            MAIN_ACTOR,
            ACTOR_COUNTRY_CODE,
            IMPACT_SCORE,
            NEWS_LINK
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{three_days}'
        AND IMPACT_SCORE < -4
        AND MAIN_ACTOR IS NOT NULL
        ORDER BY IMPACT_SCORE ASC, DATE DESC
        LIMIT 15
    """)


@st.cache_data(ttl=600)
def get_country_data(_conn):
    """
    Get aggregated event counts by country for the map/charts.
    
    Returns country code, event count, and average impact score
    for each country with events in the last 30 days.
    """
    month_ago = (NOW - datetime.timedelta(days=30)).strftime('%Y%m%d')
    
    return safe_query(_conn, f"""
        SELECT 
            ACTOR_COUNTRY_CODE as country,
            COUNT(*) as events,
            AVG(IMPACT_SCORE) as avg_impact
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{month_ago}'
        AND ACTOR_COUNTRY_CODE IS NOT NULL
        GROUP BY 1
        ORDER BY events DESC
    """)


@st.cache_data(ttl=600)
def get_time_series(_conn):
    """
    Get daily event counts for the time series chart.
    
    Returns daily totals for events, conflicts (negative impact),
    and positive events for the last 30 days.
    """
    month_ago = (NOW - datetime.timedelta(days=30)).strftime('%Y%m%d')
    
    return safe_query(_conn, f"""
        SELECT 
            DATE,
            COUNT(*) as events,
            SUM(CASE WHEN IMPACT_SCORE < -2 THEN 1 ELSE 0 END) as negative,
            SUM(CASE WHEN IMPACT_SCORE > 2 THEN 1 ELSE 0 END) as positive
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{month_ago}'
        GROUP BY 1
        ORDER BY 1
    """)


@st.cache_data(ttl=600)
def get_trending_news(_conn):
    """
    Get trending stories based on media coverage (article count).
    
    High article count = many news outlets covering the same event,
    which indicates a trending/important story.
    """
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    return safe_query(_conn, f"""
        SELECT 
            DATE,
            NEWS_LINK,
            MAIN_ACTOR,
            ACTOR_COUNTRY_CODE,
            IMPACT_SCORE,
            ARTICLE_COUNT
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{week_ago}'
        AND ARTICLE_COUNT > 3
        AND NEWS_LINK IS NOT NULL
        ORDER BY ARTICLE_COUNT DESC, DATE DESC
        LIMIT 50
    """)


@st.cache_data(ttl=600)
def get_recent_feed(_conn):
    """
    Get the most recent events for the live feed.
    
    Simply returns the latest 50 events by date,
    regardless of impact or coverage.
    """
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    return safe_query(_conn, f"""
        SELECT 
            DATE,
            NEWS_LINK,
            MAIN_ACTOR,
            ACTOR_COUNTRY_CODE,
            IMPACT_SCORE,
            ARTICLE_COUNT
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{week_ago}'
        AND NEWS_LINK IS NOT NULL
        ORDER BY DATE DESC
        LIMIT 50
    """)


@st.cache_data(ttl=600)  
def get_actor_network(_conn):
    """
    Get top actors (people, organizations, countries) by event count.
    
    Used for the "Most Mentioned" chart showing who's making news.
    """
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    return safe_query(_conn, f"""
        SELECT 
            MAIN_ACTOR,
            ACTOR_COUNTRY_CODE,
            COUNT(*) as events,
            AVG(IMPACT_SCORE) as avg_impact,
            SUM(ARTICLE_COUNT) as total_coverage
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{week_ago}'
        AND MAIN_ACTOR IS NOT NULL
        AND LENGTH(MAIN_ACTOR) > 2
        GROUP BY 1, 2
        ORDER BY events DESC
        LIMIT 15
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: UI RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
# These functions render different parts of the dashboard UI

def render_header():
    """
    Render the main header/title bar at the top of the page.
    
    Shows:
    - Logo and app title
    - "LIVE DATA" status indicator
    """
    st.markdown("""
    <div class="command-header">
        <div class="header-grid">
            <div class="logo-container">
                <span class="logo-icon">🌐</span>
                <div class="logo-text">
                    <span class="logo-title">Global News Intelligence</span>
                    <span class="logo-subtitle">Powered by GDELT • Real-Time Analytics</span>
                </div>
            </div>
            <div></div>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>LIVE DATA</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(conn):
    """
    Render the KPI metric cards at the top of the dashboard.
    
    Shows 5 key metrics:
    - Total events (all time)
    - Recent events (last 7 days)
    - Critical alerts (high impact)
    - Top hotspot country
    - Last update time
    """
    metrics = get_dashboard_metrics(conn)
    
    # Create 5 equal columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📡 TOTAL EVENTS",
            value=format_number(metrics['total']),
            delta="All time"
        )
    
    with col2:
        st.metric(
            label="⚡ LAST 7 DAYS",
            value=format_number(metrics['recent']),
            delta="Recent activity"
        )
    
    with col3:
        st.metric(
            label="🚨 CRITICAL",
            value=format_number(metrics['critical']),
            delta="High impact"
        )
    
    with col4:
        # Show full country name for hotspot
        hotspot_name = get_country_name(metrics['hotspot'])
        st.metric(
            label="🔥 TOP HOTSPOT",
            value=hotspot_name[:12],  # Truncate long names
            delta=metrics['hotspot']   # Show country code as delta
        )
    
    with col5:
        st.metric(
            label="📅 UPDATED",
            value=NOW.strftime("%H:%M"),
            delta=NOW.strftime("%d %b")
        )


def render_alert_ticker(conn):
    """
    Render the animated scrolling news ticker.
    
    Shows high-impact events scrolling across the screen
    with a pulsing "LIVE" indicator.
    
    The ticker text is duplicated (shown twice) to create
    a seamless infinite scroll effect.
    """
    df = get_alert_events(conn)
    
    # Build ticker text from alert events
    if df.empty:
        # Fallback text when no alerts
        ticker_text = "⚡ Monitoring global news feeds for significant events... │ Platform powered by GDELT real-time data │ AI analysis by Google Gemini │ "
    else:
        items = []
        for _, row in df.iterrows():
            # Format each alert item
            actor = str(row['MAIN_ACTOR'])[:25] if row['MAIN_ACTOR'] else 'Unknown'
            country = get_country_name(row['ACTOR_COUNTRY_CODE'])[:15]
            impact = row['IMPACT_SCORE']
            items.append(f"⚠️ {actor} ({country}) • Impact: {impact:.1f}")
        ticker_text = " │ ".join(items) + " │ "
    
    # Duplicate text for seamless loop animation
    # When the first copy scrolls off-screen, the second copy is right behind it
    double_text = ticker_text + ticker_text
    
    # Render the ticker HTML
    st.markdown(f"""
    <div class="ticker-container">
        <div class="ticker-label">
            <span class="ticker-dot"></span>
            LIVE
        </div>
        <div class="ticker-content">{double_text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_quick_briefing(conn):
    """
    Render the "Latest Headlines" table on the dashboard.
    
    Shows recent news events with:
    - Tone indicator (emoji based on impact score)
    - Date
    - Headline (extracted from URL)
    - Country
    - Clickable link to original article
    """
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    # Fetch headlines with high article counts (more coverage = more important)
    df = safe_query(conn, f"""
        SELECT 
            DATE,
            NEWS_LINK,
            MAIN_ACTOR,
            ACTOR_COUNTRY_CODE,
            IMPACT_SCORE,
            ARTICLE_COUNT
        FROM EVENTS_DAGSTER 
        WHERE NEWS_LINK IS NOT NULL 
        AND ARTICLE_COUNT > 5
        AND DATE >= '{week_ago}'
        ORDER BY DATE DESC, ARTICLE_COUNT DESC
        LIMIT 25
    """)
    
    if df.empty:
        st.info("📰 Loading headlines...")
        return
    
    # Standardize column names to uppercase
    df.columns = [c.upper() for c in df.columns]
    
    # Extract headlines from URLs
    df['HEADLINE'] = df.apply(
        lambda x: format_headline(x.get('NEWS_LINK', ''), x.get('MAIN_ACTOR', '')), 
        axis=1
    )
    
    # Get full country names
    df['COUNTRY'] = df['ACTOR_COUNTRY_CODE'].apply(get_country_name)
    
    # Remove duplicate headlines
    df = df.drop_duplicates(subset=['HEADLINE']).head(12)
    
    # Format dates nicely
    try:
        df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
    except:
        df['DATE_FMT'] = df['DATE']
    
    # Add tone indicator emoji based on impact score
    def get_tone_emoji(score):
        if score < -4:
            return "🔴"  # Very negative (conflict, crisis)
        elif score < -1:
            return "🟡"  # Somewhat negative
        elif score > 2:
            return "🟢"  # Positive
        else:
            return "⚪"  # Neutral
    
    df['TONE'] = df['IMPACT_SCORE'].apply(get_tone_emoji)
    
    # Display as interactive table
    st.dataframe(
        df[['TONE', 'DATE_FMT', 'HEADLINE', 'COUNTRY', 'NEWS_LINK']],
        hide_index=True,
        height=350,
        column_config={
            "TONE": st.column_config.TextColumn("", width="small"),
            "DATE_FMT": st.column_config.TextColumn("Date", width="small"),
            "HEADLINE": st.column_config.TextColumn("Headline", width="large"),
            "COUNTRY": st.column_config.TextColumn("Region", width="small"),
            "NEWS_LINK": st.column_config.LinkColumn("Link", display_text="🔗")
        },
        use_container_width=True
    )


def render_conflict_gauge(conn):
    """
    Render the "News Sentiment This Week" indicator.
    
    Shows:
    - Overall sentiment status (ELEVATED, MODERATE, STABLE, POSITIVE)
    - Average tone score
    - Breakdown of conflicts vs positive events
    
    This replaces the heavy animated gauge with lightweight HTML/CSS
    for better performance.
    """
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    # Calculate sentiment metrics
    df = safe_query(conn, f"""
        SELECT 
            AVG(IMPACT_SCORE) as avg_impact,
            SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as conflicts,
            SUM(CASE WHEN IMPACT_SCORE > 3 THEN 1 ELSE 0 END) as cooperations,
            COUNT(*) as total
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{week_ago}'
        AND IMPACT_SCORE IS NOT NULL
    """)
    
    if df.empty:
        st.info("Loading sentiment data...")
        return
    
    # Extract values
    avg_impact = df.iloc[0]['avg_impact'] or 0
    conflicts = int(df.iloc[0]['conflicts'] or 0)
    cooperations = int(df.iloc[0]['cooperations'] or 0)
    total = int(df.iloc[0]['total'] or 1)
    
    # Calculate percentages
    conflict_pct = (conflicts / total * 100) if total > 0 else 0
    coop_pct = (cooperations / total * 100) if total > 0 else 0
    
    # Determine overall status based on average impact
    if avg_impact < -2:
        status = "⚠️ ELEVATED TENSIONS"
        status_color = "#ef4444"  # Red
    elif avg_impact < 0:
        status = "🟡 MODERATE ACTIVITY"
        status_color = "#f59e0b"  # Amber
    elif avg_impact < 2:
        status = "🟢 STABLE"
        status_color = "#10b981"  # Green
    else:
        status = "✨ POSITIVE TREND"
        status_color = "#06b6d4"  # Cyan
    
    # Render the main status card
    st.markdown(f"""
    <div style="
        text-align: center; 
        padding: 1.25rem; 
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%); 
        border-radius: 12px; 
        border: 1px solid #1e3a5f; 
        margin-bottom: 1rem;
    ">
        <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
            Weekly News Sentiment
        </div>
        <div style="font-size: 1.75rem; font-weight: 700; color: {status_color}; font-family: 'JetBrains Mono', monospace;">
            {status}
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">
            Avg Tone: <span style="color: {status_color}; font-weight: 600;">{avg_impact:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render breakdown stats in 3 columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">
            <div style="font-size: 1.25rem; font-weight: 700; color: #ef4444; font-family: 'JetBrains Mono';">{conflicts:,}</div>
            <div style="font-size: 0.6rem; color: #94a3b8; text-transform: uppercase;">Negative ({conflict_pct:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.75rem; background: rgba(107, 114, 128, 0.1); border-radius: 8px; border: 1px solid rgba(107, 114, 128, 0.3);">
            <div style="font-size: 1.25rem; font-weight: 700; color: #9ca3af; font-family: 'JetBrains Mono';">{total:,}</div>
            <div style="font-size: 0.6rem; color: #94a3b8; text-transform: uppercase;">Total Events</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.75rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);">
            <div style="font-size: 1.25rem; font-weight: 700; color: #10b981; font-family: 'JetBrains Mono';">{cooperations:,}</div>
            <div style="font-size: 0.6rem; color: #94a3b8; text-transform: uppercase;">Positive ({coop_pct:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)


def render_time_series_chart(conn):
    """
    Render the 30-day activity trend chart.
    
    Shows:
    - Total events per day (area chart)
    - Negative events (red line)
    - Positive events (green line)
    
    This helps visualize trends over time and spot
    days with unusual activity.
    """
    df = get_time_series(conn)
    
    if df.empty:
        st.info("📈 Loading trend data...")
        return
    
    # Parse date strings into actual dates
    df['date_parsed'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Total events - area chart (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['date_parsed'],
            y=df['events'],
            fill='tozeroy',
            fillcolor='rgba(6, 182, 212, 0.15)',
            line=dict(color='#06b6d4', width=2),
            name='Total Events',
            hovertemplate='%{x|%b %d}: %{y:,} events<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Negative events - red line (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['date_parsed'],
            y=df['negative'],
            line=dict(color='#ef4444', width=2),
            name='Negative',
            hovertemplate='%{x|%b %d}: %{y:,} negative<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Positive events - green line (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['date_parsed'],
            y=df['positive'],
            line=dict(color='#10b981', width=2),
            name='Positive',
            hovertemplate='%{x|%b %d}: %{y:,} positive<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update layout with dark theme
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=11, color='#94a3b8'),
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(30, 58, 95, 0.3)',
            tickfont=dict(color='#64748b', size=10),
            tickformat='%d %b'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(30, 58, 95, 0.3)',
            tickfont=dict(color='#64748b', size=10),
            title=None
        ),
        yaxis2=dict(
            showgrid=False,
            tickfont=dict(color='#64748b', size=10),
            title=None
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_top_actors(conn):
    """
    Render the "Most Mentioned in News" horizontal bar chart.
    
    Shows the top 10 actors (people, organizations, countries)
    by event count this week, color-coded by average sentiment.
    """
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    # Get actor data
    df = safe_query(conn, f"""
        SELECT 
            MAIN_ACTOR,
            ACTOR_COUNTRY_CODE,
            COUNT(*) as events,
            AVG(IMPACT_SCORE) as avg_impact,
            SUM(ARTICLE_COUNT) as media_coverage
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{week_ago}'
        AND MAIN_ACTOR IS NOT NULL
        AND LENGTH(MAIN_ACTOR) > 2
        GROUP BY 1, 2
        ORDER BY events DESC
        LIMIT 10
    """)
    
    if df.empty:
        st.info("🎯 Loading actor data...")
        return
    
    # Create display labels (Actor Name + Country Code)
    df['label'] = df.apply(
        lambda x: f"{str(x['MAIN_ACTOR'])[:18]} ({x['ACTOR_COUNTRY_CODE']})" 
                  if x['ACTOR_COUNTRY_CODE'] else str(x['MAIN_ACTOR'])[:20],
        axis=1
    )
    
    # Assign colors based on average sentiment
    def get_bar_color(impact):
        if impact < -3:
            return '#ef4444'  # Red for negative
        elif impact < 0:
            return '#f59e0b'  # Amber for slightly negative
        elif impact > 3:
            return '#10b981'  # Green for positive
        else:
            return '#06b6d4'  # Cyan for neutral
    
    df['color'] = df['avg_impact'].apply(get_bar_color)
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=df['events'],
        y=df['label'],
        orientation='h',
        marker_color=df['color'],
        text=df['events'].apply(lambda x: f'{x:,}'),
        textposition='outside',
        textfont=dict(color='#94a3b8', size=10),
        hovertemplate='<b>%{y}</b><br>Events: %{x:,}<br>Avg Impact: %{customdata:.2f}<extra></extra>',
        customdata=df['avg_impact']
    ))
    
    # Update layout
    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=50, t=10, b=0),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(30, 58, 95, 0.3)',
            tickfont=dict(color='#64748b', size=10),
            title=None
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color='#e2e8f0', size=11, family='JetBrains Mono'),
            autorange='reversed',  # Highest at top
            title=None
        ),
        bargap=0.3
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_impact_distribution(conn):
    """
    Render the "News Tone Breakdown" donut chart.
    
    Shows the distribution of events by impact category:
    - Crisis (very negative)
    - Negative
    - Neutral
    - Positive
    - Very Positive
    """
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    # Get impact distribution
    df = safe_query(conn, f"""
        SELECT 
            CASE 
                WHEN IMPACT_SCORE < -5 THEN 'Crisis'
                WHEN IMPACT_SCORE < -2 THEN 'Negative'
                WHEN IMPACT_SCORE < 2 THEN 'Neutral'
                WHEN IMPACT_SCORE < 5 THEN 'Positive'
                ELSE 'Very Positive'
            END as category,
            COUNT(*) as count
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{week_ago}'
        AND IMPACT_SCORE IS NOT NULL
        GROUP BY 1
    """)
    
    if df.empty:
        st.info("📊 Loading distribution data...")
        return
    
    # Define colors for each category
    color_map = {
        'Crisis': '#ef4444',
        'Negative': '#f59e0b',
        'Neutral': '#64748b',
        'Positive': '#10b981',
        'Very Positive': '#06b6d4'
    }
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=df['category'],
        values=df['count'],
        hole=0.6,  # Donut hole size
        marker_colors=[color_map.get(c, '#64748b') for c in df['category']],
        textinfo='percent',
        textfont=dict(size=11, color='#e2e8f0'),
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>',
        sort=False
    )])
    
    fig.update_layout(
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(size=10, color='#94a3b8')
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_country_bar_chart(conn):
    """
    Render the "Top Countries by Coverage" bar chart.
    
    Shows the top 8 countries by event count,
    with bars colored by average sentiment.
    """
    df = get_country_data(conn)
    
    if df.empty:
        st.info("🏆 Loading country data...")
        return
    
    # Get top 8 countries
    df = df.head(8)
    
    # Get full country names
    df['country_name'] = df['country'].apply(get_country_name)
    
    # Create vertical bar chart
    fig = go.Figure(go.Bar(
        x=df['country_name'],
        y=df['events'],
        marker_color='#06b6d4',
        text=df['events'].apply(format_number),
        textposition='outside',
        textfont=dict(color='#94a3b8', size=10),
        hovertemplate='<b>%{x}</b><br>Events: %{y:,}<extra></extra>'
    ))
    
    fig.update_layout(
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color='#94a3b8', size=9),
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(30, 58, 95, 0.3)',
            showticklabels=False
        ),
        bargap=0.4
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_trending_table(conn):
    """
    Render the trending stories table.
    
    Shows stories with the highest media coverage (article count),
    indicating stories that many news outlets are reporting on.
    """
    df = get_trending_news(conn)
    
    if df.empty:
        st.info("🔥 Loading trending stories...")
        return
    
    # Standardize column names
    df.columns = [c.upper() for c in df.columns]
    
    # Extract headlines from URLs
    df['HEADLINE'] = df.apply(
        lambda x: format_headline(x.get('NEWS_LINK', ''), x.get('MAIN_ACTOR', '')),
        axis=1
    )
    
    # Get country names
    df['COUNTRY'] = df['ACTOR_COUNTRY_CODE'].apply(get_country_name)
    
    # Remove duplicates and take top 15
    df = df.drop_duplicates(subset=['HEADLINE']).head(15)
    
    # Format dates
    try:
        df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
    except:
        df['DATE_FMT'] = df['DATE']
    
    # Display table
    st.dataframe(
        df[['DATE_FMT', 'HEADLINE', 'COUNTRY', 'ARTICLE_COUNT', 'NEWS_LINK']],
        hide_index=True,
        height=400,
        column_config={
            "DATE_FMT": st.column_config.TextColumn("Date", width="small"),
            "HEADLINE": st.column_config.TextColumn("Story", width="large"),
            "COUNTRY": st.column_config.TextColumn("Region", width="small"),
            "ARTICLE_COUNT": st.column_config.NumberColumn("📰", width="small", help="Number of articles covering this story"),
            "NEWS_LINK": st.column_config.LinkColumn("🔗", width="small")
        },
        use_container_width=True
    )


def render_feed_table(conn):
    """
    Render the recent events feed table.
    
    Shows the most recent events in chronological order,
    regardless of impact or coverage.
    """
    df = get_recent_feed(conn)
    
    if df.empty:
        st.info("📋 Loading recent events...")
        return
    
    # Standardize column names
    df.columns = [c.upper() for c in df.columns]
    
    # Extract headlines
    df['HEADLINE'] = df.apply(
        lambda x: format_headline(x.get('NEWS_LINK', ''), x.get('MAIN_ACTOR', '')),
        axis=1
    )
    
    # Get country names
    df['COUNTRY'] = df['ACTOR_COUNTRY_CODE'].apply(get_country_name)
    
    # Format dates
    try:
        df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
    except:
        df['DATE_FMT'] = df['DATE']
    
    # Add tone indicator
    def get_tone_emoji(score):
        if score is None:
            return "⚪"
        if score < -3:
            return "🔴"
        elif score < 0:
            return "🟡"
        elif score > 2:
            return "🟢"
        return "⚪"
    
    df['TONE'] = df['IMPACT_SCORE'].apply(get_tone_emoji)
    
    # Remove duplicates and take top 15
    df = df.drop_duplicates(subset=['HEADLINE']).head(15)
    
    # Display table
    st.dataframe(
        df[['TONE', 'DATE_FMT', 'HEADLINE', 'COUNTRY', 'NEWS_LINK']],
        hide_index=True,
        height=400,
        column_config={
            "TONE": st.column_config.TextColumn("", width="small"),
            "DATE_FMT": st.column_config.TextColumn("Date", width="small"),
            "HEADLINE": st.column_config.TextColumn("Event", width="large"),
            "COUNTRY": st.column_config.TextColumn("Region", width="small"),
            "NEWS_LINK": st.column_config.LinkColumn("🔗", width="small")
        },
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: AI CHAT FUNCTIONALITY
# ═══════════════════════════════════════════════════════════════════════════════
# Functions for the natural language query interface

def execute_ai_query(query_engine, question, conn):
    """
    Execute a natural language query against the database.
    
    This function:
    1. Sends the question to the LlamaIndex query engine
    2. The query engine uses Gemini to convert natural language to SQL
    3. Executes the SQL and returns results
    
    Args:
        query_engine: LlamaIndex NLSQLTableQueryEngine
        question: User's natural language question
        conn: Database connection (for fallback queries)
    
    Returns:
        dict: Contains success status, response text, SQL query, and data
    """
    try:
        # Execute the natural language query
        response = query_engine.query(question)
        
        # Extract the generated SQL from metadata
        sql_query = response.metadata.get('sql_query', None)
        
        # Try to get the result data
        result_data = None
        if sql_query:
            result_data = safe_query(conn, sql_query)
        
        return {
            'success': True,
            'response': str(response),
            'sql': sql_query,
            'data': result_data
        }
        
    except Exception as e:
        logger.error(f"AI query failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'response': None,
            'sql': None,
            'data': None
        }


def render_ai_chat(conn, engine):
    """
    Render the AI chat interface.
    
    Features:
    - Example queries to help users get started
    - Chat input at the top for accessibility
    - Message history display
    - SQL query reveal option
    - Results displayed as interactive tables
    """
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "🌐 **GDELT Analyst Ready**\n\nAsk me questions about global news events! I'll convert your question to SQL and query the database."
        }]
    
    # Example queries hint
    st.markdown("""
    <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem;">
        <span style="color: #64748b; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;">💡 TRY:</span>
        <span style="color: #94a3b8; font-size: 0.75rem;"> "Show crisis events this week" • "What's happening in Russia?" • "Top 5 countries by activity"</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat input at the top for easy access
    prompt = st.chat_input("Ask about global news events...")
    
    # Display chat history (last 10 messages to keep it manageable)
    for msg in st.session_state.messages[-10:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Process new user input
    if prompt:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Querying GDELT database..."):
                # Get the query engine
                qe = get_query_engine(engine)
                
                if qe:
                    # Execute the query
                    result = execute_ai_query(qe, prompt, conn)
                    
                    if result['success']:
                        response = result['response']
                        st.markdown(response)
                        
                        # Display result data if available
                        if result['data'] is not None and not result['data'].empty:
                            df = result['data']
                            df.columns = [c.upper() for c in df.columns]
                            
                            # Format dates if present
                            if 'DATE' in df.columns:
                                try:
                                    df['DATE'] = pd.to_datetime(
                                        df['DATE'].astype(str), 
                                        format='%Y%m%d'
                                    ).dt.strftime('%d %b %Y')
                                except:
                                    pass
                            
                            # Add headlines if news links present
                            if 'NEWS_LINK' in df.columns:
                                df['HEADLINE'] = df.apply(
                                    lambda x: format_headline(x.get('NEWS_LINK', '')),
                                    axis=1
                                )
                                cols = [c for c in ['DATE', 'HEADLINE', 'ACTOR_COUNTRY_CODE', 'IMPACT_SCORE', 'NEWS_LINK'] if c in df.columns]
                                st.dataframe(
                                    df[cols],
                                    hide_index=True,
                                    column_config={
                                        "NEWS_LINK": st.column_config.LinkColumn("🔗"),
                                        "HEADLINE": st.column_config.TextColumn("Event", width="large")
                                    }
                                )
                            else:
                                st.dataframe(df, hide_index=True)
                        
                        # Show SQL query in expandable section
                        if result['sql']:
                            with st.expander("🔍 View Generated SQL"):
                                st.code(result['sql'], language='sql')
                        
                        # Save response to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                    else:
                        # Query failed
                        st.error(f"❌ {result.get('error', 'Query failed')}")
                        st.info("💡 Try a different question like: 'Show recent high-impact events'")
                else:
                    st.error("AI Engine unavailable. Please check your API configuration.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: ARCHITECTURE & ABOUT PAGES
# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio/documentation pages showing technical details

def render_architecture():
    """
    Render the "How It Works" / Architecture documentation page.
    
    This page is specifically designed for recruiters and technical reviewers
    to understand the technical decisions and skills demonstrated in this project.
    """
    # Page title
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="font-family: 'JetBrains Mono', monospace; color: #e2e8f0; margin-bottom: 0.5rem;">
            🏗️ System Architecture
        </h2>
        <p style="color: #64748b; font-size: 0.9rem;">
            End-to-end data pipeline: Ingesting GDELT news data with AI-powered analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture flow diagram
    st.markdown("""
    <div style="
        background: #111827; 
        border: 1px solid #1e3a5f; 
        border-radius: 12px; 
        padding: 2rem; 
        text-align: center;
        margin-bottom: 2rem;
    ">
        <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
            <span class="arch-node">📰 GDELT API</span>
            <span class="arch-arrow">→</span>
            <span class="arch-node">⚡ GitHub Actions</span>
            <span class="arch-arrow">→</span>
            <span class="arch-node">🦆 MotherDuck</span>
            <span class="arch-arrow">→</span>
            <span class="arch-node">🤖 Gemini AI</span>
            <span class="arch-arrow">→</span>
            <span class="arch-node">🎨 Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed component cards
    col1, col2 = st.columns(2)
    
    with col1:
        # Data Source card
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; height: 100%; margin-bottom: 1rem;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #06b6d4; font-size: 0.9rem; margin-bottom: 1rem;">
                📥 DATA SOURCE: GDELT
            </h4>
            <p style="color: #94a3b8; font-size: 0.8rem; line-height: 1.6; margin-bottom: 0.75rem;">
                <strong>GDELT</strong> (Global Database of Events, Language & Tone) monitors news worldwide in 100+ languages, 
                identifying events, people, organizations, and emotions.
            </p>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>Updates:</strong> Every 15 minutes</li>
                <li><strong>Pipeline:</strong> Dagster orchestration</li>
                <li><strong>Automation:</strong> GitHub Actions (30-min)</li>
                <li><strong>Volume:</strong> ~10M+ events processed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Layer card
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; height: 100%;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #8b5cf6; font-size: 0.9rem; margin-bottom: 1rem;">
                🤖 GENERATIVE AI LAYER
            </h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>LLM:</strong> Google Gemini 2.5 Flash</li>
                <li><strong>Framework:</strong> LlamaIndex (RAG)</li>
                <li><strong>Feature:</strong> Text-to-SQL Generation</li>
                <li><strong>Embeddings:</strong> Gemini Embedding-001</li>
                <li><strong>Cost:</strong> Free Tier API</li>
            </ul>
            <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(139, 92, 246, 0.1); border-radius: 6px; border-left: 3px solid #8b5cf6;">
                <span style="color: #8b5cf6; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">🧠 Natural Language → SQL</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Storage card - highlighting the Snowflake to MotherDuck migration
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; height: 100%; margin-bottom: 1rem;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #10b981; font-size: 0.9rem; margin-bottom: 1rem;">
                🗄️ DATA STORAGE
            </h4>
            <p style="color: #94a3b8; font-size: 0.8rem; line-height: 1.6; margin-bottom: 0.5rem;">
                <strong>Originally:</strong> Built on Snowflake
            </p>
            <p style="color: #94a3b8; font-size: 0.8rem; line-height: 1.6; margin-bottom: 0.75rem;">
                <strong>Migration:</strong> Moved to MotherDuck for cost optimization
            </p>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>Database:</strong> DuckDB (columnar)</li>
                <li><strong>Cloud:</strong> MotherDuck (serverless)</li>
                <li><strong>Performance:</strong> Sub-second queries</li>
            </ul>
            <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(16, 185, 129, 0.1); border-radius: 6px; border-left: 3px solid #10b981;">
                <span style="color: #10b981; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">💡 COST: Snowflake → Free Tier</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization card
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; height: 100%;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #f59e0b; font-size: 0.9rem; margin-bottom: 1rem;">
                📊 VISUALIZATION
            </h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>Framework:</strong> Streamlit</li>
                <li><strong>Charts:</strong> Plotly (interactive)</li>
                <li><strong>Styling:</strong> Custom CSS</li>
                <li><strong>Hosting:</strong> Streamlit Cloud</li>
            </ul>
            <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(245, 158, 11, 0.1); border-radius: 6px; border-left: 3px solid #f59e0b;">
                <span style="color: #f59e0b; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">🚀 Deployed on Streamlit Cloud</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tech stack badges
    st.markdown("---")
    st.markdown("""
    <h3 style="font-family: 'JetBrains Mono', monospace; color: #e2e8f0; text-align: center; margin: 1.5rem 0 1rem 0;">
        🛠️ Tech Stack
    </h3>
    <div style="text-align: center; padding: 1rem;">
        <span class="tech-badge">🐍 Python</span>
        <span class="tech-badge">❄️ Snowflake</span>
        <span class="tech-badge">🦆 DuckDB</span>
        <span class="tech-badge">☁️ MotherDuck</span>
        <span class="tech-badge">⚙️ Dagster</span>
        <span class="tech-badge">🔧 dbt</span>
        <span class="tech-badge">🤖 Gen AI</span>
        <span class="tech-badge">🦙 LlamaIndex</span>
        <span class="tech-badge">✨ Gemini API</span>
        <span class="tech-badge">📊 Plotly</span>
        <span class="tech-badge">🎨 Streamlit</span>
        <span class="tech-badge">🔄 CI/CD</span>
        <span class="tech-badge">⚡ GitHub Actions</span>
        <span class="tech-badge">🐼 Pandas</span>
        <span class="tech-badge">🗃️ SQL</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Key features
    st.markdown("---")
    st.markdown("""
    <h3 style="font-family: 'JetBrains Mono', monospace; color: #e2e8f0; text-align: center; margin: 2rem 0 1.5rem 0;">
        ✨ Key Features
    </h3>
    """, unsafe_allow_html=True)
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <span style="font-size: 2rem;">🔄</span>
            <h4 style="color: #e2e8f0; font-size: 1rem; margin: 0.75rem 0 0.5rem 0;">Automated Pipeline</h4>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">
                Data refreshed every 30 mins via GitHub Actions + Dagster
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <span style="font-size: 2rem;">🧠</span>
            <h4 style="color: #e2e8f0; font-size: 1rem; margin: 0.75rem 0 0.5rem 0;">Natural Language Queries</h4>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">
                Ask questions in English, get SQL results via Gemini AI
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <span style="font-size: 2rem;">💰</span>
            <h4 style="color: #e2e8f0; font-size: 1rem; margin: 0.75rem 0 0.5rem 0;">Zero-Cost Infrastructure</h4>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">
                MotherDuck free tier + Gemini free API = $0/month
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_about():
    """
    Render the About page with personal information and contact links.
    
    This page helps recruiters understand the project goals and
    provides ways to contact the developer.
    """
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h2 style="font-family: 'JetBrains Mono', monospace; color: #e2e8f0; margin-bottom: 1rem;">
            👋 About This Project
        </h2>
        <p style="color: #94a3b8; font-size: 1rem; max-width: 750px; margin: 0 auto 1.5rem auto; line-height: 1.7;">
            This platform analyzes data from <strong>GDELT</strong> (Global Database of Events, Language & Tone) — 
            the world's largest open database of human society, monitoring news media from every country in 100+ languages.
        </p>
        <p style="color: #64748b; font-size: 0.9rem; max-width: 700px; margin: 0 auto 2rem auto; line-height: 1.7;">
            Originally built on <strong>Snowflake</strong>, then strategically migrated to <strong>MotherDuck</strong> 
            for cost optimization. Uses <strong>Gemini AI free tier</strong> for natural language queries — 
            demonstrating how to build production-grade analytics on a $0 budget.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Goals and Skills sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #06b6d4; font-size: 0.9rem; margin-bottom: 1rem;">
                🎯 PROJECT GOALS
            </h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li>Demonstrate end-to-end data engineering</li>
                <li>Showcase AI/LLM integration skills</li>
                <li>Build production-grade dashboards</li>
                <li>Optimize costs (Snowflake → MotherDuck)</li>
                <li>Leverage free tiers effectively</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #10b981; font-size: 0.9rem; margin-bottom: 1rem;">
                🛠️ SKILLS DEMONSTRATED
            </h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li>Python, SQL, Data Engineering</li>
                <li>ETL/ELT Pipelines (Dagster, dbt)</li>
                <li>Cloud Platforms (Snowflake → MotherDuck)</li>
                <li>Generative AI / LLM Integration</li>
                <li>CI/CD (GitHub Actions automation)</li>
                <li>Data Visualization & Dashboards</li>
                <li>Cost Optimization & Cloud Migration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Contact links
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h4 style="font-family: 'JetBrains Mono', monospace; color: #e2e8f0; margin-bottom: 1rem;">
            📬 GET IN TOUCH
        </h4>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <a href="https://github.com/Mohith-akash" target="_blank" style="
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: #111827;
                border: 1px solid #1e3a5f;
                border-radius: 8px;
                padding: 0.75rem 1.25rem;
                color: #e2e8f0;
                text-decoration: none;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                transition: all 0.3s ease;
            ">
                ⭐ GitHub
            </a>
            <a href="https://www.linkedin.com/in/mohith-akash/" target="_blank" style="
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: #111827;
                border: 1px solid #1e3a5f;
                border-radius: 8px;
                padding: 0.75rem 1.25rem;
                color: #e2e8f0;
                text-decoration: none;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                transition: all 0.3s ease;
            ">
                💼 LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
# This is the entry point that assembles all the components together

def main():
    """
    Main function that runs the Streamlit application.
    
    This function:
    1. Injects custom CSS styling
    2. Initializes database and AI connections
    3. Renders the header
    4. Creates the navigation tabs
    5. Renders content for each tab
    6. Renders the footer
    """
    # Apply custom styling
    inject_custom_css()
    
    # Initialize connections (cached, so only runs once)
    conn = get_db_connection()
    engine = get_ai_engine(get_sql_engine())
    
    # Render the header
    render_header()
    
    # Create the main navigation tabs (short names for mobile)
    tab_dashboard, tab_analytics, tab_ai, tab_arch, tab_about = st.tabs([
        "📊 HOME",
        "📈 TRENDS", 
        "🤖 AI",
        "🏗️ TECH",
        "👤 ABOUT"
    ])
    
    # Mobile swipe hint (only visible on small screens)
    st.markdown('<div class="swipe-hint">👆 Swipe for more tabs →</div>', unsafe_allow_html=True)
    
    # ═══════════════ DASHBOARD TAB ═══════════════
    with tab_dashboard:
        # KPI Metrics row at the top
        render_metrics(conn)
        
        # Animated alert ticker
        render_alert_ticker(conn)
        
        st.markdown("---")
        
        # Main dashboard content - Headlines and Sentiment
        col_briefing, col_sentiment = st.columns([6, 4])
        
        with col_briefing:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">📰</span>
                <span class="card-title">Latest Headlines</span>
            </div>
            """, unsafe_allow_html=True)
            render_quick_briefing(conn)
        
        with col_sentiment:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">⚡</span>
                <span class="card-title">News Sentiment This Week</span>
            </div>
            """, unsafe_allow_html=True)
            render_conflict_gauge(conn)
        
        st.markdown("---")
        
        # Charts row - Actors and Distribution
        col_actors, col_dist = st.columns([6, 4])
        
        with col_actors:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">🎯</span>
                <span class="card-title">Most Mentioned in News</span>
            </div>
            """, unsafe_allow_html=True)
            render_top_actors(conn)
        
        with col_dist:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">📊</span>
                <span class="card-title">News Tone Breakdown</span>
            </div>
            """, unsafe_allow_html=True)
            render_impact_distribution(conn)
            
            st.markdown("""
            <div class="card-header" style="margin-top: 1rem;">
                <span class="card-icon">🏆</span>
                <span class="card-title">Top Countries by Coverage</span>
            </div>
            """, unsafe_allow_html=True)
            render_country_bar_chart(conn)
    
    # ═══════════════ ANALYTICS TAB ═══════════════
    with tab_analytics:
        # Tables at top (as user requested)
        col_trend, col_feed = st.columns(2)
        
        with col_trend:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">🔥</span>
                <span class="card-title">Trending Stories (by Coverage)</span>
            </div>
            """, unsafe_allow_html=True)
            render_trending_table(conn)
        
        with col_feed:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">📋</span>
                <span class="card-title">Recent Events Feed</span>
            </div>
            """, unsafe_allow_html=True)
            render_feed_table(conn)
        
        st.markdown("---")
        
        # Time series chart at bottom
        st.markdown("""
        <div class="card-header">
            <span class="card-icon">📈</span>
            <span class="card-title">30-Day Activity Trend</span>
        </div>
        """, unsafe_allow_html=True)
        render_time_series_chart(conn)
    
    # ═══════════════ AI CHAT TAB ═══════════════
    with tab_ai:
        col_chat, col_info = st.columns([7, 3])
        
        with col_chat:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">🤖</span>
                <span class="card-title">Ask Questions in Plain English</span>
            </div>
            """, unsafe_allow_html=True)
            render_ai_chat(conn, engine)
        
        with col_info:
            # How it works explanation
            st.markdown("""
            <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.25rem;">
                <h4 style="font-family: 'JetBrains Mono', monospace; color: #06b6d4; font-size: 0.85rem; margin-bottom: 1rem;">
                    ℹ️ HOW IT WORKS
                </h4>
                <p style="color: #94a3b8; font-size: 0.8rem; line-height: 1.6;">
                    Your question is converted to SQL using <strong>Gemini AI</strong>, 
                    then executed against the GDELT database.
                </p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #1e3a5f;">
                    <p style="color: #64748b; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; margin-bottom: 0.5rem;">
                        DATA AVAILABLE
                    </p>
                    <p style="font-size: 0.75rem; color: #94a3b8; line-height: 1.5;">
                        📅 Event dates<br>
                        👤 Actors & countries<br>
                        📊 Impact scores (-10 to +10)<br>
                        📰 Media coverage count<br>
                        🔗 Source news links
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Query tips
            st.markdown("""
            <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.25rem; margin-top: 1rem;">
                <h4 style="font-family: 'JetBrains Mono', monospace; color: #f59e0b; font-size: 0.85rem; margin-bottom: 0.75rem;">
                    ⚡ TIPS
                </h4>
                <p style="color: #94a3b8; font-size: 0.8rem; line-height: 1.6;">
                    • Use country names (Russia, China)<br>
                    • Specify time (today, this week)<br>
                    • Ask for "crisis" or "conflict"<br>
                    • Request "trending" stories
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # ═══════════════ ARCHITECTURE TAB ═══════════════
    with tab_arch:
        render_architecture()
    
    # ═══════════════ ABOUT TAB ═══════════════
    with tab_about:
        render_about()
    
    # ═══════════════ FOOTER ═══════════════
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; border-top: 1px solid #1e3a5f; margin-top: 2rem;">
        <p style="color: #64748b; font-size: 0.8rem; margin-bottom: 0.5rem;">
            <strong>GDELT</strong> (Global Database of Events, Language & Tone) monitors worldwide news media, 
            identifying events, emotions, and themes in real-time.
        </p>
        <p style="color: #475569; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
            Built by <a href="https://www.linkedin.com/in/mohith-akash/" target="_blank" style="color: #06b6d4; text-decoration: none;">Mohith Akash</a> | 
            Snowflake → MotherDuck | Gemini AI Free Tier
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14: ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
# This runs when the script is executed

if __name__ == "__main__":
    main()