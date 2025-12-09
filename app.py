"""
🌐 GLOBAL NEWS INTELLIGENCE PLATFORM
Real-Time Analytics Dashboard for GDELT Global News Database

This application monitors worldwide news events in real-time and displays them
in an easy-to-understand dashboard with charts, tables, and AI-powered search.

Author: Mohith Akash | Portfolio Project
Tech Stack: Python, Streamlit, DuckDB, Cerebras AI, Plotly
"""

# ============================================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================================

import streamlit as st              # Creates the web interface
import os                           # Accesses environment variables
import pandas as pd                 # Handles data tables
import plotly.graph_objects as go   # Creates interactive charts
from plotly.subplots import make_subplots  # Allows multiple charts in one
from dotenv import load_dotenv      # Loads secret keys from .env file
from llama_index.llms.cerebras import Cerebras  # Cerebras AI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding  # AI text understanding
from llama_index.core import SQLDatabase, Settings  # Database wrapper for AI
from llama_index.core.query_engine import NLSQLTableQueryEngine  # Converts English to SQL
from sqlalchemy import create_engine  # Connects to databases
import datetime                     # Handles dates and times
import pycountry                    # Converts country codes
import logging                      # Tracks errors and info messages
import re                           # Pattern matching in text
from urllib.parse import urlparse, unquote  # Extracts info from web links
import duckdb                       # Fast database engine

# ============================================================================
# SECTION 2: INITIAL SETUP
# ============================================================================

# Configure the web page appearance
st.set_page_config(
    page_title="Global News Intelligence",  # Browser tab title
    page_icon="🌐",                          # Browser tab icon
    layout="wide",                           # Use full screen width
    initial_sidebar_state="collapsed"        # Hide sidebar by default
)

# Load environment variables from .env file
load_dotenv()

# Set up logging to track what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gdelt")

# ============================================================================
# SECTION 3: SECURITY & API KEYS
# ============================================================================

def get_secret(key):
    """
    Get secret API keys from environment variables or Streamlit secrets.
    
    Think of this like getting a password from a secure vault.
    We check two places: .env file (local) and Streamlit Cloud (deployment).
    
    Args:
        key: The name of the secret we're looking for (e.g., "CEREBRAS_API_KEY")
    
    Returns:
        The secret value if found, None if not found
    """
    # First, check environment variables
    val = os.getenv(key)
    if val: 
        return val
    
    # If not found, check Streamlit's secret storage
    try: 
        return st.secrets.get(key)
    except: 
        return None

# List of required API keys - the app won't work without these
REQUIRED_ENVS = [
    "MOTHERDUCK_TOKEN",
    "CEREBRAS_API_KEY"
]

# Check if any required keys are missing
missing = [k for k in REQUIRED_ENVS if not get_secret(k)]
if missing:
    # If keys are missing, show error and stop the app
    st.error(f"❌ Missing required API keys: {', '.join(missing)}")
    st.stop()  # Stop execution - can't proceed without keys

# Set the API keys in environment variables so other parts of code can use them
for key in REQUIRED_ENVS:
    val = get_secret(key)
    if val: 
        os.environ[key] = val

# ============================================================================
# SECTION 4: GLOBAL CONSTANTS
# ============================================================================

GEMINI_MODEL = "llama3.1-8b"  # Cerebras model name

def get_dates():
    """
    Get current date and calculated date ranges.
    
    IMPORTANT: This function is called every time to ensure dates are always current.
    If we calculated these once at module level, they would become stale as the 
    server stays running for days/weeks.
    
    Returns:
        dict with: now, week_ago, month_ago (all as strings in YYYYMMDD format)
    """
    now = datetime.datetime.now()
    week_ago = (now - datetime.timedelta(days=7)).strftime('%Y%m%d')
    month_ago = (now - datetime.timedelta(days=30)).strftime('%Y%m%d')
    return {
        'now': now,
        'week_ago': week_ago,
        'month_ago': month_ago
    }

# ============================================================================
# SECTION 5: STYLING
# ============================================================================

def inject_css():
    """
    Add custom CSS styling to make the dashboard look professional.
    
    CSS is like the "paint and decoration" of a website - it controls colors,
    fonts, spacing, animations, etc. We use a dark theme with blue accents.
    """
    st.markdown("""
    <style>
        /* Import modern fonts from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
        
        /* Define color palette as variables for easy reuse */
        :root { 
            --bg:#0a0e17;           /* Dark background */
            --card:#111827;         /* Slightly lighter for cards */
            --border:#1e3a5f;       /* Blue border color */
            --text:#e2e8f0;         /* Light text */
            --muted:#94a3b8;        /* Dimmed text */
            --cyan:#06b6d4;         /* Bright cyan accent */
            --green:#10b981;        /* Green for positive */
            --red:#ef4444;          /* Red for negative */
            --amber:#f59e0b;        /* Orange for warnings */
        }
        
        /* Main app background */
        .stApp { background: var(--bg); }
        
        /* Hide Streamlit's default header, menu, and footer */
        header[data-testid="stHeader"], #MainMenu, footer, .stDeployButton { 
            display: none !important; 
        }
        
        /* Set default fonts for different elements */
        html, body, p, span, div { 
            font-family: 'Inter', sans-serif; 
            color: var(--text); 
        }
        h1, h2, h3, code { 
            font-family: 'JetBrains Mono', monospace; 
        }
        
        /* Main content area spacing */
        .block-container { 
            padding: 1.5rem 2rem; 
            max-width: 100%; 
        }
        
        /* Header section styling */
        .header { 
            border-bottom: 1px solid var(--border); 
            padding: 1rem 0 1.5rem; 
            margin-bottom: 1.5rem; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
        }
        
        /* Logo and branding */
        .logo { 
            display: flex; 
            align-items: center; 
            gap: 0.75rem; 
        }
        .logo-icon { 
            font-size: 2.5rem; 
        }
        .logo-title { 
            font-family: 'JetBrains Mono'; 
            font-size: 1.4rem; 
            font-weight: 700; 
            text-transform: uppercase; 
        }
        .logo-sub { 
            font-size: 0.7rem; 
            color: var(--cyan); 
        }
        
        /* "LIVE DATA" badge in header */
        .live-badge { 
            display: flex; 
            align-items: center; 
            gap: 0.5rem; 
            background: rgba(16,185,129,0.15); 
            border: 1px solid rgba(16,185,129,0.4); 
            padding: 0.4rem 0.8rem; 
            border-radius: 20px; 
            font-size: 0.75rem; 
        }
        
        /* Pulsing green dot animation */
        .live-dot { 
            width: 8px; 
            height: 8px; 
            background: var(--green); 
            border-radius: 50%; 
            animation: pulse 2s infinite; 
        }
        @keyframes pulse { 
            0%,100% { opacity:1; } 
            50% { opacity:0.5; } 
        }
        
        /* Metric cards (the boxes showing numbers like "Total Events") */
        div[data-testid="stMetric"] { 
            background: var(--card); 
            border: 1px solid var(--border); 
            border-radius: 12px; 
            padding: 1rem; 
        }
        div[data-testid="stMetric"] label { 
            color: var(--muted); 
            font-size: 0.7rem; 
            font-family: 'JetBrains Mono'; 
            text-transform: uppercase; 
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { 
            font-size: 1.5rem; 
            font-weight: 700; 
            font-family: 'JetBrains Mono'; 
        }
        
        /* Card headers (titles above charts/tables) */
        .card-hdr { 
            display: flex; 
            align-items: center; 
            gap: 0.75rem; 
            margin-bottom: 1rem; 
            padding-bottom: 0.75rem; 
            border-bottom: 1px solid var(--border); 
        }
        .card-title { 
            font-family: 'JetBrains Mono'; 
            font-size: 0.85rem; 
            font-weight: 600; 
            text-transform: uppercase; 
        }
        
        /* Tab navigation styling */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 0; 
            background: #0d1320; 
            border-radius: 8px; 
            padding: 4px; 
            border: 1px solid var(--border); 
            overflow-x: auto; 
        }
        .stTabs [data-baseweb="tab"] { 
            font-family: 'JetBrains Mono'; 
            font-size: 0.75rem; 
            color: var(--muted); 
            padding: 0.5rem 0.9rem; 
            white-space: nowrap; 
        }
        .stTabs [aria-selected="true"] { 
            background: #1a2332; 
            color: var(--cyan); 
            border-radius: 6px; 
        }
        .stTabs [data-baseweb="tab-highlight"], 
        .stTabs [data-baseweb="tab-border"] { 
            display: none; 
        }
        
        /* Data table styling */
        div[data-testid="stDataFrame"] { 
            background: var(--card); 
            border: 1px solid var(--border); 
            border-radius: 12px; 
        }
        div[data-testid="stDataFrame"] th { 
            background: #1a2332 !important; 
            color: var(--muted) !important; 
            font-size: 0.75rem; 
            text-transform: uppercase; 
        }
        
        /* Live ticker (scrolling alert bar) */
        .ticker { 
            background: linear-gradient(90deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05)); 
            border-left: 4px solid var(--red); 
            border-radius: 0 8px 8px 0; 
            padding: 0.6rem 0; 
            overflow: hidden; 
            position: relative; 
            margin: 0.5rem 0; 
        }
        .ticker-label { 
            position: absolute; 
            left: 0; 
            top: 0; 
            bottom: 0; 
            background: linear-gradient(90deg, rgba(127,29,29,0.98), transparent); 
            padding: 0.6rem 1.25rem 0.6rem 0.75rem; 
            font-size: 0.7rem; 
            font-weight: 600; 
            color: var(--red); 
            display: flex; 
            align-items: center; 
            gap: 0.5rem; 
            z-index: 2; 
        }
        
        /* Blinking dot in ticker */
        .ticker-dot { 
            width: 7px; 
            height: 7px; 
            background: var(--red); 
            border-radius: 50%; 
            animation: blink 1s infinite; 
        }
        @keyframes blink { 
            0%,100% { opacity:1; } 
            50% { opacity:0.3; } 
        }
        
        /* Scrolling text animation */
        .ticker-text { 
            display: inline-block; 
            white-space: nowrap; 
            padding-left: 95px; 
            animation: scroll 40s linear infinite; 
            font-size: 0.8rem; 
            color: #fca5a5; 
        }
        @keyframes scroll { 
            0% { transform: translateX(0); } 
            100% { transform: translateX(-50%); } 
        }
        
        /* Technology badges */
        .tech-badge { 
            display: inline-flex; 
            background: #1a2332; 
            border: 1px solid var(--border); 
            border-radius: 20px; 
            padding: 0.4rem 0.8rem; 
            font-size: 0.75rem; 
            color: var(--muted); 
            margin: 0.25rem; 
        }
        
        /* Horizontal divider line */
        hr { 
            border: none; 
            border-top: 1px solid var(--border); 
            margin: 1.5rem 0; 
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SECTION 6: DATABASE CONNECTION
# ============================================================================

@st.cache_resource  # Cache this so we only connect once, not on every page reload
def get_db():
    """
    Connect to MotherDuck cloud database.
    
    MotherDuck is like Google Drive for databases - it stores all our news data
    in the cloud. We use DuckDB (a fast database engine) to access it.
    
    Returns:
        Database connection object
    """
    return duckdb.connect(
        f'md:gdelt_db?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}',
        read_only=True  # Read-only mode - we won't modify the data
    )

@st.cache_resource  # Cache this connection too
def get_engine():
    """
    Create a SQLAlchemy engine for AI to use.
    
    SQLAlchemy is a tool that helps the AI understand our database structure
    so it can write SQL queries automatically.
    
    Returns:
        SQLAlchemy engine object
    """
    return create_engine(
        f"duckdb:///md:gdelt_db?motherduck_token={os.getenv('MOTHERDUCK_TOKEN')}"
    )

# ============================================================================
# SECTION 7: DATA RETRIEVAL HELPERS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def detect_table(_conn):
    """
    Automatically find the main events table in the database.
    
    The table might be named differently in different setups, so we search
    for any table with "event" in the name. It's like searching for a file
    when you're not sure of the exact filename.
    
    Args:
        _conn: Database connection (underscore means "don't cache this parameter")
    
    Returns:
        Name of the main events table (usually "events_dagster")
    """
    try:
        # Get list of all tables in database
        result = _conn.execute("SHOW TABLES").df()
        
        if not result.empty:
            # Loop through table names and find one with "event" in it
            for table_name in result.iloc[:, 0].tolist():
                if 'event' in table_name.lower():
                    return table_name
            
            # If no "event" table found, just use the first table
            return result.iloc[0, 0]
    except:
        pass
    
    # If anything goes wrong, assume default table name
    return 'events_dagster'

def safe_query(conn, sql):
    """
    Execute a SQL query safely with error handling.
    
    This is like a safety wrapper - if something goes wrong with the query,
    we return an empty table instead of crashing the whole app.
    
    Args:
        conn: Database connection
        sql: SQL query to execute
    
    Returns:
        Pandas DataFrame with results, or empty DataFrame if error
    """
    try:
        return conn.execute(sql).df()
    except Exception as e:
        # Log the error so we can debug it later
        logger.error(f"Query error: {e}")
        # Return empty table instead of crashing
        return pd.DataFrame()

# ============================================================================
# SECTION 8: AI SETUP
# ============================================================================

@st.cache_resource  # Only set up AI once
def get_ai_engine(_engine):
    """
    Set up Cerebras AI to understand our database.
    
    This is like teaching the AI about our data so it can answer questions
    in plain English. The AI learns the table structure and can write SQL
    queries automatically.
    
    Args:
        _engine: SQLAlchemy engine for database access
    
    Returns:
        SQL database wrapper for AI, or None if setup fails
    """
    try:
        # Get API key for Cerebras AI service
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key: 
            return None
        
        # Initialize Cerebras LLM
        llm = Cerebras(
            api_key=api_key, 
            model=GEMINI_MODEL,  # Use llama3.1-8b
            temperature=0.1
        )
        
        # Initialize embedding model
        embed = GoogleGenAIEmbedding(
            api_key=os.getenv("GOOGLE_API_KEY") or api_key, 
            model_name="text-embedding-004"
        )
        
        # Set these as global defaults for LlamaIndex
        Settings.llm = llm
        Settings.embed_model = embed
        
        # Get database connection and find main table
        conn = get_db()
        main_table = detect_table(conn)
        
        # Wrap database so AI can understand it
        sql_db = SQLDatabase(_engine, include_tables=[main_table])
        
        return sql_db
    
    except Exception as e:
        logger.error(f"AI initialization failed: {e}")
        return None

@st.cache_resource  # Cache the query engine
def get_query_engine(_sql_db):
    """
    Create an AI query engine that converts English to SQL.
    
    This is the "brain" that takes your questions like "Show me crisis events"
    and converts them to SQL like "SELECT * FROM events WHERE impact_score < -3"
    
    Args:
        _sql_db: SQL database wrapper from get_ai_engine()
    
    Returns:
        Natural language SQL query engine, or None if setup fails
    """
    if not _sql_db: 
        return None
    
    try:
        # Get list of available tables
        tables = list(_sql_db.get_usable_table_names())
        
        # Find the events table
        target = next(
            (t for t in tables if 'event' in t.lower()), 
            tables[0] if tables else None
        )
        
        if target:
            # Create query engine for specific table
            return NLSQLTableQueryEngine(
                sql_database=_sql_db, 
                tables=[target]
            )
        
        # Fallback: create engine for all tables
        return NLSQLTableQueryEngine(sql_database=_sql_db)
    
    except:
        return None

@st.cache_resource  # Cache the Cerebras LLM
def get_cerebras_llm():
    """
    Initialize Cerebras LLM for generating answers from query results.
    
    Returns:
        Cerebras LLM instance, or None if API key missing
    """
    try:
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            logger.warning("CEREBRAS_API_KEY not found")
            return None
        
        cerebras_llm = Cerebras(
            api_key=api_key,
            model="llama3.1-8b",  # Cerebras Llama model
            temperature=0.1       # Low temp for factual responses
        )
        
        logger.info("Cerebras LLM initialized successfully")
        return cerebras_llm
    
    except Exception as e:
        logger.error(f"Failed to initialize Cerebras: {e}")
        return None

# ============================================================================
# SECTION 9: DATA TRANSFORMATION HELPERS
# ============================================================================

def get_country(code):
    """
    Convert country codes to full country names.
    
    Converts codes like "USA" → "United States", "GBR" → "United Kingdom"
    Uses the pycountry library which has a database of all country codes.
    
    Args:
        code: 2-letter (US) or 3-letter (USA) country code
    
    Returns:
        Full country name, or None if code is invalid
    """
    # Validate input
    if not code or not isinstance(code, str): 
        return None
    
    code = code.strip().upper()
    
    if len(code) < 2: 
        return None
    
    try:
        # Try 2-letter code first
        if len(code) == 2:
            country = pycountry.countries.get(alpha_2=code)
            if country: 
                return country.name
        
        # Try 3-letter code
        if len(code) == 3:
            country = pycountry.countries.get(alpha_3=code)
            if country: 
                return country.name
        
        return None
    
    except:
        return None

def get_impact_label(score):
    """
    Convert numeric impact scores to human-readable labels.
    
    Impact scores range from -10 (extreme conflict) to +10 (major agreement).
    This makes them easier to understand at a glance.
    
    Args:
        score: Impact score number (-10 to +10)
    
    Returns:
        Descriptive label with emoji (e.g., "🔴 Major Conflict")
    """
    if score is None: 
        return "Neutral"
    
    score = float(score)
    
    # Negative events
    if score <= -8: return "🔴 Severe Crisis"
    if score <= -5: return "🔴 Major Conflict"
    if score <= -3: return "🟠 Rising Tensions"
    if score <= -1: return "🟡 Minor Dispute"
    
    # Neutral events
    if score < 1: return "⚪ Neutral"
    
    # Positive events
    if score < 3: return "🟢 Cooperation"
    if score < 5: return "🟢 Partnership"
    
    return "✨ Major Agreement"

def get_intensity_label(score):
    """
    Get more detailed intensity description for events.
    
    Similar to get_impact_label but with more specific descriptions
    that are easier for general audience to understand.
    
    Args:
        score: Impact score number (-10 to +10)
    
    Returns:
        Descriptive intensity label (e.g., "⚔️ Armed Conflict")
    """
    if score is None: 
        return "⚪ Neutral Event"
    
    score = float(score)
    
    # Negative intensities
    if score <= -8: return "⚔️ Armed Conflict"      # Military action
    if score <= -6: return "🔴 Major Crisis"        # Severe situation
    if score <= -4: return "🟠 Serious Tension"     # High stakes
    if score <= -2: return "🟡 Verbal Dispute"      # Disagreements
    
    # Neutral
    if score < 2: return "⚪ Neutral Event"         # Standard news
    
    # Positive intensities
    if score < 4: return "🟢 Diplomatic Talk"       # Cooperation
    if score < 6: return "🤝 Active Partnership"    # Joint efforts
    
    return "✨ Peace Agreement"                     # Major accord

def clean_headline(text):
    """
    Remove garbage patterns and dates from headlines.
    
    URLs often contain weird codes, dates, and IDs that aren't useful as headlines.
    This function aggressively cleans them out to leave just the readable text.
    
    Examples:
        "20241206 Gaza conflict continues" → "Gaza conflict continues"
        "article-a3f8d9 Breaking news" → "Breaking news"
        "abc123def456ghi789" → None (rejected as garbage)
    
    Args:
        text: Raw headline text extracted from URL or database
    
    Returns:
        Cleaned headline text, or None if text is garbage/too short
    """
    if not text: 
        return None
    
    text = str(text).strip()
    
    # Reject patterns that indicate this is garbage data
    reject_patterns = [
        r'^[a-f0-9]{8}[-\s][a-f0-9]{4}',  # UUIDs like "a3f8d9b2-1c4e"
        r'^[a-f0-9\s\-]{20,}$',            # Long hex strings
        r'^(article|post|item|id)[\s\-_]*[a-f0-9]{8}',  # Article IDs
    ]
    
    for pattern in reject_patterns:
        if re.match(pattern, text.lower()): 
            return None
    
    # Remove date patterns
    for _ in range(5):
        text = re.sub(r'^\d{4}\s+\d{1,2}\s+\d{1,2}\s+', '', text)  # "2024 12 06 "
        text = re.sub(r'^\d{1,2}\s+\d{1,2}\s+', '', text)          # "12 06 "
        text = re.sub(r'^\d{1,2}[/\-\.]\d{1,2}\s+', '', text)      # "12/06 "
        text = re.sub(r'^\d{4}\s+', '', text)                      # "2024 "
        text = re.sub(r'^\d{8}\s*', '', text)                      # "20241206"
        text = re.sub(r'^\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\s*', '', text)  # "2024/12/06"
    
    # Remove other garbage patterns
    text = re.sub(r'\s+\d{1,2}\.\d{5,}', ' ', text)                # Long decimal numbers
    text = re.sub(r'\s+\d{5,}', ' ', text)                         # Long integers
    text = re.sub(r'\s+[a-z]{3,5}\d[a-z\d]{4,}', ' ', text, flags=re.I)  # IDs like "abc1d2e3f4"
    text = re.sub(r'\s+[a-z0-9]{12,}(?=\s|$)', ' ', text, flags=re.I)   # Very long alphanumeric
    text = re.sub(r'[\s,]+\d{1,3}$', '', text)                     # Trailing numbers
    text = re.sub(r'\.(html?|php|aspx?|jsp|shtml)$', '', text, flags=re.I)  # File extensions
    text = re.sub(r'[-_]+', ' ', text)                             # Convert dashes/underscores to spaces
    text = ' '.join(text.split())                                  # Normalize whitespace
    
    # Quality checks - reject if too short or too many numbers
    if len(text) < 10: 
        return None
    
    text_no_spaces = text.replace(' ', '')
    if text_no_spaces:
        # Reject if more than 20% numbers
        num_count = sum(c.isdigit() for c in text_no_spaces)
        if num_count > len(text_no_spaces) * 0.2: 
            return None
    
    # Reject if more than 35% hex characters
    hex_count = sum(c in '0123456789abcdefABCDEF' for c in text_no_spaces)
    if hex_count > len(text_no_spaces) * 0.35: 
        return None
    
    # Reject single-word "headlines"
    if ' ' not in text: 
        return None
    
    # Reject if fewer than 3 words
    words = text.split()
    if len(words) < 3: 
        return None
    
    # Truncate to 100 characters for display
    return text[:100]

def enhance_headline(text, impact_score=None, actor=None):
    """
    Make headlines more engaging with proper capitalization.
    
    Capitalizes important words like "President", "Military", "Crisis" to
    make headlines more professional and easier to read.
    
    Args:
        text: Cleaned headline text
        impact_score: Event impact score (not currently used)
        actor: Main actor in event (not currently used)
    
    Returns:
        Enhanced headline with better capitalization
    """
    if not text: 
        return None
    
    words = text.split()
    capitalized = []
    
    # List of important words that should always be capitalized
    important_words = {
        'president', 'minister', 'government', 'military', 'congress', 'senate',
        'crisis', 'attack', 'strike', 'protest', 'emergency', 'war', 'peace',
        'agreement', 'deal', 'summit', 'meeting', 'vote', 'election', 'law',
        'court', 'judge', 'police', 'fire', 'flood', 'earthquake', 'storm'
    }
    
    for i, word in enumerate(words):
        word_lower = word.lower()
        
        # Always capitalize first word
        if i == 0:
            capitalized.append(word.capitalize())
        # Capitalize important words
        elif word_lower in important_words:
            capitalized.append(word.capitalize())
        # Convert ALL CAPS to Title Case
        elif word.isupper() and len(word) > 2:
            capitalized.append(word.title())
        else:
            capitalized.append(word)
    
    return ' '.join(capitalized)

def extract_headline(url, actor=None, impact_score=None):
    """
    Extract a readable headline from a news article URL.
    
    URLs often contain the article title in the path, like:
    https://example.com/news/2024/gaza-conflict-continues
    
    This function extracts "gaza-conflict-continues" and cleans it up.
    
    Args:
        url: Full URL to news article
        actor: Main actor (used as fallback if URL extraction fails)
        impact_score: Event impact score (passed to enhance_headline)
    
    Returns:
        Clean, readable headline text
    """
    # If no URL but we have an actor name, use that
    if not url and actor:
        cleaned = clean_headline(actor)
        return enhance_headline(cleaned, impact_score, actor) if cleaned else None
    
    if not url: 
        return None
    
    try:
        # Parse the URL into components
        parsed = urlparse(str(url))
        path = unquote(parsed.path)  # Decode URL encoding
        
        # Split path into segments and find promising ones
        segments = [s for s in path.split('/') if s and len(s) > 8]
        
        # Try segments in reverse order
        for seg in reversed(segments):
            cleaned = clean_headline(seg)
            if cleaned and len(cleaned) > 20:
                return enhance_headline(cleaned, impact_score, actor)
        
        # If URL extraction failed, use actor as fallback
        if actor:
            cleaned = clean_headline(actor)
            return enhance_headline(cleaned, impact_score, actor) if cleaned else None
        
        return None
    
    except:
        # If any error occurs, try using actor as fallback
        if actor:
            cleaned = clean_headline(actor)
            return enhance_headline(cleaned, impact_score, actor) if cleaned else None
        return None

def process_df(df):
    """
    Process raw database results into display-ready format.
    
    This is the master processing function that:
    1. Extracts headlines from URLs
    2. Converts country codes to names
    3. Formats dates
    4. Adds emoji indicators
    5. Adds intensity labels
    6. Removes duplicates
    
    Args:
        df: Raw pandas DataFrame from database query
    
    Returns:
        Processed DataFrame ready for display in tables
    """
    if df.empty: 
        return df
    
    # Create copy to avoid modifying original
    df = df.copy()
    
    # Ensure all column names are uppercase for consistency
    df.columns = [c.upper() for c in df.columns]
    
    # Extract headlines from URLs
    headlines = []
    for _, row in df.iterrows():
        headline = extract_headline(
            row.get('NEWS_LINK', ''), 
            row.get('MAIN_ACTOR', ''), 
            row.get('IMPACT_SCORE', None)
        )
        headlines.append(headline if headline else None)
    
    df['HEADLINE'] = headlines
    
    # Remove rows where we couldn't extract a headline
    df = df[df['HEADLINE'].notna()]
    
    # Convert country codes to full names
    df['REGION'] = df['ACTOR_COUNTRY_CODE'].apply(
        lambda x: get_country(x) or x if x else 'Global'
    )
    
    # Format dates from YYYYMMDD to DD/MM for easier reading
    try:
        df['DATE_FMT'] = pd.to_datetime(
            df['DATE'].astype(str), 
            format='%Y%m%d'
        ).dt.strftime('%d/%m')
    except:
        df['DATE_FMT'] = df['DATE']
    
    # Add emoji tone indicators
    df['TONE'] = df['IMPACT_SCORE'].apply(
        lambda x: "🔴" if x and x < -4 else (
            "🟡" if x and x < -1 else (
                "🟢" if x and x > 2 else "⚪"
            )
        )
    )
    
    # Add detailed intensity labels
    df['INTENSITY'] = df['IMPACT_SCORE'].apply(get_intensity_label)
    
    # Remove duplicate headlines
    df = df.drop_duplicates(subset=['HEADLINE'])
    
    return df

# ============================================================================
# SECTION 10: DATA QUERIES
# ============================================================================
# Each function retrieves specific data needed for different parts of the dashboard

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_metrics(_c, t):
    """
    Get key metrics for the dashboard header.
    
    Calculates:
    - Total events in database
    - Events from past 7 days
    - Critical events (high impact) this week
    - Hotspot country (most active this week)
    
    Args:
        _c: Database connection
        t: Table name
    
    Returns:
        Dictionary with metrics: {total, recent, critical, hotspot}
    """
    dates = get_dates()  # Get current dates
    
    # Query for counts
    df = safe_query(_c, f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN DATE >= '{dates['week_ago']}' THEN 1 ELSE 0 END) as recent,
            SUM(CASE WHEN ABS(IMPACT_SCORE) > 6 AND DATE >= '{dates['week_ago']}' THEN 1 ELSE 0 END) as critical
        FROM {t}
    """)
    
    # Query for hotspot country
    hs = safe_query(_c, f"""
        SELECT ACTOR_COUNTRY_CODE, COUNT(*) as c 
        FROM {t} 
        WHERE DATE >= '{dates['week_ago']}' 
            AND ACTOR_COUNTRY_CODE IS NOT NULL 
        GROUP BY 1 
        ORDER BY 2 DESC 
        LIMIT 1
    """)
    
    return {
        'total': df.iloc[0]['total'] if not df.empty else 0,
        'recent': df.iloc[0]['recent'] if not df.empty else 0,
        'critical': df.iloc[0]['critical'] if not df.empty else 0,
        'hotspot': hs.iloc[0]['ACTOR_COUNTRY_CODE'] if not hs.empty else None
    }

@st.cache_data(ttl=300)
def get_alerts(_c, t):
    """
    Get high-impact alerts for the live ticker.
    
    Fetches recent events with very negative impact scores (< -4)
    to display in the scrolling alert banner.
    
    Args:
        _c: Database connection
        t: Table name
    
    Returns:
        DataFrame with: MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE
    """
    # Get events from past 3 days with severe negative impact
    now = datetime.datetime.now()  # Get current time
    three_days_ago = (now - datetime.timedelta(days=3)).strftime('%Y%m%d')
    
    return safe_query(_c, f"""
        SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE 
        FROM {t} 
        WHERE DATE >= '{three_days_ago}' 
            AND IMPACT_SCORE < -4 
            AND MAIN_ACTOR IS NOT NULL 
        ORDER BY IMPACT_SCORE 
        LIMIT 15
    """)

@st.cache_data(ttl=300)
def get_headlines(_c, t):
    """
    Get latest headlines with news links.
    
    Retrieves recent events that have:
    - Valid news links
    - Multiple articles (ARTICLE_COUNT > 5) = more important
    - From past week
    
    Args:
        _c: Database connection
        t: Table name
    
    Returns:
        DataFrame with: DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE
    """
    dates = get_dates()  # Get current dates
    return safe_query(_c, f"""
        SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE 
        FROM {t} 
        WHERE NEWS_LINK IS NOT NULL 
            AND ARTICLE_COUNT > 5 
            AND DATE >= '{dates['week_ago']}' 
        ORDER BY DATE DESC, ARTICLE_COUNT DESC 
        LIMIT 60
    """)

@st.cache_data(ttl=300)
def get_trending(_c, t):
    """
    Get trending stories (most talked-about events).
    
    "Trending" means high ARTICLE_COUNT - lots of news sources
    covering the same event indicates it's important.
    
    Args:
        _c: Database connection
        t: Table name
    
    Returns:
        DataFrame with: DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, 
                        IMPACT_SCORE, ARTICLE_COUNT
    """
    dates = get_dates()  # Get current dates
    return safe_query(_c, f"""
        SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT 
        FROM {t} 
        WHERE DATE >= '{dates['week_ago']}' 
            AND ARTICLE_COUNT > 3 
            AND NEWS_LINK IS NOT NULL 
        ORDER BY ARTICLE_COUNT DESC 
        LIMIT 60
    """)

@st.cache_data(ttl=300)
def get_feed(_c, t):
    """
    Get chronological feed of recent events.
    
    Simple chronological list of what happened recently,
    sorted by date (newest first).
    
    Args:
        _c: Database connection
        t: Table name
    
    Returns:
        DataFrame with: DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE
    """
    dates = get_dates()  # Get current dates
    return safe_query(_c, f"""
        SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE 
        FROM {t} 
        WHERE DATE >= '{dates['week_ago']}' 
            AND NEWS_LINK IS NOT NULL 
        ORDER BY DATE DESC 
        LIMIT 60
    """)

@st.cache_data(ttl=300)
def get_countries(_c, t):
    """
    Get event counts by country.
    
    Shows which countries had the most events in the past month.
    Used for the "Top Countries" bar chart.
    
    Args:
        _c: Database connection
        t: Table name
    
    Returns:
        DataFrame with: country (country code), events (count)
    """
    dates = get_dates()  # Get current dates
    return safe_query(_c, f"""
        SELECT ACTOR_COUNTRY_CODE as country, COUNT(*) as events 
        FROM {t} 
        WHERE DATE >= '{dates['month_ago']}' 
            AND ACTOR_COUNTRY_CODE IS NOT NULL 
        GROUP BY 1 
        ORDER BY 2 DESC
    """)

@st.cache_data(ttl=300)
def get_timeseries(_c, t):
    """
    Get daily event counts for the past month.
    
    Used for the 30-day trend line chart. Shows:
    - Total events per day
    - Negative events per day (impact < -2)
    - Positive events per day (impact > 2)
    
    Args:
        _c: Database connection
        t: Table name
    
    Returns:
        DataFrame with: DATE, events, negative, positive
    """
    dates = get_dates()  # Get current dates
    return safe_query(_c, f"""
        SELECT 
            DATE, 
            COUNT(*) as events, 
            SUM(CASE WHEN IMPACT_SCORE < -2 THEN 1 ELSE 0 END) as negative, 
            SUM(CASE WHEN IMPACT_SCORE > 2 THEN 1 ELSE 0 END) as positive 
        FROM {t} 
        WHERE DATE >= '{dates['month_ago']}' 
        GROUP BY 1 
        ORDER BY 1
    """)

@st.cache_data(ttl=300)
def get_sentiment(_c, t):
    """
    Get overall sentiment statistics for the week.
    
    Calculates:
    - Average impact score
    - Count of negative events (impact < -3)
    - Count of positive events (impact > 3)
    - Total events
    
    Args:
        _c: Database connection
        t: Table name
    
    Returns:
        DataFrame with: avg, neg, pos, total
    """
    dates = get_dates()  # Get current dates
    return safe_query(_c, f"""
        SELECT 
            AVG(IMPACT_SCORE) as avg, 
            SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as neg, 
            SUM(CASE WHEN IMPACT_SCORE > 3 THEN 1 ELSE 0 END) as pos, 
            COUNT(*) as total 
        FROM {t} 
        WHERE DATE >= '{dates['week_ago']}' 
            AND IMPACT_SCORE IS NOT NULL
    """)

@st.cache_data(ttl=300)
def get_actors(_c, t):
    """
    Get most mentioned actors (people/organizations) this week.
    
    Shows who's making headlines - presidents, governments, militaries, etc.
    Includes average impact score to show if their coverage is positive/negative.
    
    Args:
        _c: Database connection
        t: Table name
    
    Returns:
        DataFrame with: MAIN_ACTOR, ACTOR_COUNTRY_CODE, events, avg_impact
    """
    dates = get_dates()  # Get current dates
    return safe_query(_c, f"""
        SELECT 
            MAIN_ACTOR, 
            ACTOR_COUNTRY_CODE, 
            COUNT(*) as events, 
            AVG(IMPACT_SCORE) as avg_impact 
        FROM {t} 
        WHERE DATE >= '{dates['week_ago']}' 
            AND MAIN_ACTOR IS NOT NULL 
            AND LENGTH(MAIN_ACTOR) > 3 
        GROUP BY 1, 2 
        ORDER BY 3 DESC 
        LIMIT 10
    """)

@st.cache_data(ttl=300)
def get_distribution(_c, t):
    """
    Get distribution of event tones (positive/negative/neutral).
    
    Categorizes events into 5 buckets based on impact score:
    - Crisis (< -5)
    - Negative (-5 to -2)
    - Neutral (-2 to 2)
    - Positive (2 to 5)
    - Very Positive (> 5)
    
    Used for the pie chart showing overall sentiment distribution.
    
    Args:
        _c: Database connection
        t: Table name
    
    Returns:
        DataFrame with: cat (category), cnt (count)
    """
    dates = get_dates()  # Get current dates
    return safe_query(_c, f"""
        SELECT 
            CASE 
                WHEN IMPACT_SCORE < -5 THEN 'Crisis' 
                WHEN IMPACT_SCORE < -2 THEN 'Negative' 
                WHEN IMPACT_SCORE < 2 THEN 'Neutral' 
                WHEN IMPACT_SCORE < 5 THEN 'Positive' 
                ELSE 'Very Positive' 
            END as cat, 
            COUNT(*) as cnt 
        FROM {t} 
        WHERE DATE >= '{dates['week_ago']}' 
            AND IMPACT_SCORE IS NOT NULL 
        GROUP BY 1
    """)

# ============================================================================
# SECTION 11: UI RENDERING FUNCTIONS
# ============================================================================
# Each function displays a specific component of the dashboard

def render_header():
    """
    Display the main header with logo and "LIVE DATA" badge.
    
    Shows:
    - 🌐 Globe icon
    - "GLOBAL NEWS INTELLIGENCE" title
    - "Powered by GDELT • Real-Time Analytics" subtitle
    - Green pulsing "LIVE DATA" badge on the right
    """
    st.markdown('''
        <div class="header">
            <div class="logo">
                <span class="logo-icon">🌐</span>
                <div>
                    <div class="logo-title">Global News Intelligence</div>
                    <div class="logo-sub">Powered by GDELT • Real-Time Analytics</div>
                </div>
            </div>
            <div class="live-badge">
                <span class="live-dot"></span> 
                LIVE DATA
            </div>
        </div>
    ''', unsafe_allow_html=True)

def render_metrics(c, t):
    """
    Display key metrics in 5 cards across the top.
    
    Shows:
    1. Total Events (all-time database count)
    2. 7 Days (events from past week)
    3. Critical (high-impact events this week)
    4. Hotspot (country with most events)
    5. Updated (current time and date)
    
    Each card includes a tooltip explaining what the number means.
    
    Args:
        c: Database connection
        t: Table name
    """
    # Get the metrics data
    m = get_metrics(c, t)
    
    # Create 5 equal-width columns
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Format large numbers with commas
    def fmt(n): 
        return f"{int(n or 0):,}"
    
    # Column 1: Total Events
    with c1:
        st.metric("📡 TOTAL", fmt(m['total']), "All time")
        st.markdown('''
            <div style="text-align:center;margin-top:-0.5rem;">
                <span style="font-size:0.7rem;color:#64748b;">
                    💡 Total global events tracked in database
                </span>
            </div>
        ''', unsafe_allow_html=True)
    
    # Column 2: Recent Events
    with c2:
        st.metric("⚡ 7 DAYS", fmt(m['recent']), "Recent")
        st.markdown('''
            <div style="text-align:center;margin-top:-0.5rem;">
                <span style="font-size:0.7rem;color:#64748b;">
                    💡 Events from the past week
                </span>
            </div>
        ''', unsafe_allow_html=True)
    
    # Column 3: Critical Events
    with c3:
        st.metric("🚨 CRITICAL", fmt(m['critical']), "High impact")
        st.markdown('''
            <div style="text-align:center;margin-top:-0.5rem;">
                <span style="font-size:0.7rem;color:#64748b;">
                    💡 Severe events (impact score > 6) this week
                </span>
            </div>
        ''', unsafe_allow_html=True)
    
    # Column 4: Hotspot Country
    hs = m['hotspot']
    with c4:
        country_name = get_country(hs) or hs or "N/A"
        
        # Smart truncation for very long country names
        if country_name == "United States":
            display_name = "United States"
        elif len(country_name) > 15:
            display_name = country_name[:15] + "..."
        else:
            display_name = country_name
        
        st.metric("🔥 HOTSPOT", display_name, hs or "")
        st.markdown('''
            <div style="text-align:center;margin-top:-0.5rem;">
                <span style="font-size:0.7rem;color:#64748b;">
                    💡 Country with most events this week
                </span>
            </div>
        ''', unsafe_allow_html=True)
    
    # Column 5: Last Updated Time
    with c5:
        now = datetime.datetime.now()  # Get current time
        st.metric("📅 UPDATED", now.strftime("%H:%M"), now.strftime("%d %b"))
        st.markdown('''
            <div style="text-align:center;margin-top:-0.5rem;">
                <span style="font-size:0.7rem;color:#64748b;">
                    💡 UTC timezone • Refreshes every 5 min
                </span>
            </div>
        ''', unsafe_allow_html=True)

def render_ticker(c, t):
    """
    Display scrolling ticker with high-impact alerts.
    
    Shows a red scrolling bar with recent crisis-level events.
    If no alerts, shows a generic monitoring message.
    
    Includes explanation tooltip above the ticker.
    
    Args:
        c: Database connection
        t: Table name
    """
    # Get alert data
    df = get_alerts(c, t)
    
    if df.empty:
        # No alerts - show generic message
        txt = "⚡ Monitoring global news │ "
    else:
        # Build scrolling text from alerts
        items = []
        for _, r in df.iterrows():
            actor = r.get('MAIN_ACTOR', '')[:30] or "Event"
            country = get_country(r.get('ACTOR_COUNTRY_CODE', '')) or 'Global'
            score = r.get('IMPACT_SCORE', 0)
            items.append(f"⚠️ {actor} ({country}) • {score:.1f}")
        
        # Join with separator and duplicate for seamless scroll
        txt = " │ ".join(items) + " │ "
    
    # Display explanation tooltip
    st.markdown('''
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;
                    padding:0.5rem;margin-bottom:0.5rem;text-align:center;">
            <span style="font-size:0.7rem;color:#64748b;">
                💡 <b>LIVE TICKER:</b> Shows high-impact events (score < -4) from 
                the past 3 days. Numbers indicate severity level (-10 to +10 scale, 
                where negative = conflict/crisis)
            </span>
        </div>
    ''', unsafe_allow_html=True)
    
    # Display ticker with scrolling animation
    st.markdown(f'''
        <div class="ticker">
            <div class="ticker-label">
                <span class="ticker-dot"></span> 
                LIVE
            </div>
            <div class="ticker-text">{txt + txt}</div>
        </div>
    ''', unsafe_allow_html=True)

def render_headlines(c, t):
    """
    Display latest headlines table.
    
    Shows recent news with links, organized by date.
    Includes tone emoji and region for quick scanning.
    
    Args:
        c: Database connection
        t: Table name
    """
    df = get_headlines(c, t)
    
    if df.empty:
        st.info("📰 Loading...")
        return
    
    # Process and limit to 12 rows
    df = process_df(df).head(12)
    
    if df.empty:
        st.info("📰 No headlines")
        return
    
    # Display table with specific columns
    st.dataframe(
        df[['TONE', 'DATE_FMT', 'HEADLINE', 'REGION', 'NEWS_LINK']], 
        hide_index=True, 
        height=350,
        column_config={
            "TONE": st.column_config.TextColumn("", width="small"),
            "DATE_FMT": st.column_config.TextColumn("Date", width="small"),
            "HEADLINE": st.column_config.TextColumn("Headline", width="large"),
            "REGION": st.column_config.TextColumn("Region", width="small"),
            "NEWS_LINK": st.column_config.LinkColumn("🔗", width="small")
        }, 
        width='stretch'
    )

def render_sentiment(c, t):
    """
    Display weekly sentiment summary card.
    
    Shows:
    - Overall status (Elevated/Moderate/Stable/Positive)
    - Average impact score
    - Counts of negative, total, and positive events
    
    Color-coded based on sentiment (red = negative, green = positive).
    
    Args:
        c: Database connection
        t: Table name
    """
    df = get_sentiment(c, t)
    
    if df.empty:
        st.info("Loading...")
        return
    
    # Extract values
    avg = df.iloc[0]['avg'] or 0
    neg = int(df.iloc[0]['neg'] or 0)
    pos = int(df.iloc[0]['pos'] or 0)
    total = int(df.iloc[0]['total'] or 1)
    
    # Determine status and color based on average
    if avg < -2:
        status, color = ("⚠️ ELEVATED", "#ef4444")
    elif avg < 0:
        status, color = ("🟡 MODERATE", "#f59e0b")
    elif avg < 2:
        status, color = ("🟢 STABLE", "#10b981")
    else:
        status, color = ("✨ POSITIVE", "#06b6d4")
    
    # Display main status card
    st.markdown(f'''
        <div style="text-align:center;padding:1.25rem;
                    background:linear-gradient(135deg,rgba(14,165,233,0.1),rgba(6,182,212,0.05));
                    border-radius:12px;border:1px solid #1e3a5f;margin-bottom:1rem;">
            <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;">
                Weekly Sentiment
            </div>
            <div style="font-size:1.75rem;font-weight:700;color:{color};">
                {status}
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;">
                Avg: <span style="color:{color}">{avg:.2f}</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Display breakdown in 3 columns
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f'''
            <div style="text-align:center;padding:0.75rem;
                        background:rgba(239,68,68,0.1);border-radius:8px;">
                <div style="font-size:1.25rem;font-weight:700;color:#ef4444;">
                    {neg:,}
                </div>
                <div style="font-size:0.6rem;color:#94a3b8;">NEGATIVE</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with c2:
        st.markdown(f'''
            <div style="text-align:center;padding:0.75rem;
                        background:rgba(107,114,128,0.1);border-radius:8px;">
                <div style="font-size:1.25rem;font-weight:700;color:#9ca3af;">
                    {total:,}
                </div>
                <div style="font-size:0.6rem;color:#94a3b8;">TOTAL</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with c3:
        st.markdown(f'''
            <div style="text-align:center;padding:0.75rem;
                        background:rgba(16,185,129,0.1);border-radius:8px;">
                <div style="font-size:1.25rem;font-weight:700;color:#10b981;">
                    {pos:,}
                </div>
                <div style="font-size:0.6rem;color:#94a3b8;">POSITIVE</div>
            </div>
        ''', unsafe_allow_html=True)

def render_actors(c, t):
    """
    Display horizontal bar chart of most mentioned actors.
    
    Shows top 10 people/organizations in the news this week.
    Bars are color-coded by average sentiment:
    - Red: Negative coverage
    - Orange: Mixed coverage
    - Cyan: Neutral coverage
    - Green: Positive coverage
    
    Args:
        c: Database connection
        t: Table name
    """
    df = get_actors(c, t)
    
    if df.empty:
        st.info("🎯 Loading...")
        return
    
    # Create labels with actor name and country
    labels = []
    for _, r in df.iterrows():
        actor = r['MAIN_ACTOR'][:25]  # Truncate long names
        country = get_country(r.get('ACTOR_COUNTRY_CODE', ''))
        
        if country:
            labels.append(f"{actor} ({country[:10]})")
        else:
            labels.append(actor)
    
    # Color-code bars based on average impact score
    colors = [
        '#ef4444' if x and x < -3 else (  # Red for very negative
            '#f59e0b' if x and x < 0 else (  # Orange for negative
                '#10b981' if x and x > 3 else '#06b6d4'  # Green for positive, cyan for neutral
            )
        ) 
        for x in df['avg_impact']
    ]
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=df['events'],  # Number of events
        y=labels,        # Actor names
        orientation='h', # Horizontal bars
        marker_color=colors,
        text=df['events'].apply(lambda x: f'{x:,}'),  # Show count as text
        textposition='outside',
        textfont=dict(color='#94a3b8', size=10)
    ))
    
    # Style the chart
    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=50,t=10,b=0),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(30,58,95,0.3)',
            tickfont=dict(color='#64748b')
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color='#e2e8f0', size=11),
            autorange='reversed'  # Top actor at top of chart
        ),
        bargap=0.3
    )
    
    # Display the chart
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch', key='actors_chart')

def render_distribution(c, t, chart_key='distribution'):
    """
    Display pie chart of event tone distribution.
    
    Shows breakdown of events by sentiment category:
    - Crisis (red)
    - Negative (orange)
    - Neutral (gray)
    - Positive (green)
    - Very Positive (cyan)
    
    Args:
        c: Database connection
        t: Table name
        chart_key: Unique key for the chart (required when multiple charts on page)
    """
    df = get_distribution(c, t)
    
    if df.empty:
        st.info("📊 Loading...")
        return
    
    # Define colors for each category
    colors = {
        'Crisis': '#ef4444',
        'Negative': '#f59e0b',
        'Neutral': '#64748b',
        'Positive': '#10b981',
        'Very Positive': '#06b6d4'
    }
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=df['cat'],    # Category names
        values=df['cnt'],    # Counts
        hole=0.6,            # Size of center hole
        marker_colors=[colors.get(c, '#64748b') for c in df['cat']],
        textinfo='percent',  # Show percentages
        textfont=dict(size=11, color='#e2e8f0')
    )])
    
    # Style the chart
    fig.update_layout(
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=10,b=10),
        showlegend=True,
        legend=dict(
            orientation='h',  # Horizontal legend
            y=-0.2,          # Below chart
            x=0.5,
            xanchor='center',
            font=dict(size=10, color='#94a3b8')
        )
    )
    
    # Display with unique key
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch', key=chart_key)

def render_countries(c, t):
    """
    Display bar chart of top countries by event count.
    
    Shows the 8 countries with the most events in the past month.
    All bars are cyan colored.
    
    Args:
        c: Database connection
        t: Table name
    """
    df = get_countries(c, t)
    
    if df.empty:
        st.info("🏆 Loading...")
        return
    
    # Limit to top 8
    df = df.head(8)
    
    # Convert country codes to full names
    df['name'] = df['country'].apply(lambda x: get_country(x) or x or 'Unknown')
    
    # Format numbers
    def fmt(n): 
        return f"{n/1000:.1f}K" if n >= 1000 else str(int(n))
    
    # Create vertical bar chart
    fig = go.Figure(go.Bar(
        x=df['name'],    # Country names
        y=df['events'],  # Event counts
        marker_color='#06b6d4',  # Cyan
        text=df['events'].apply(fmt),
        textposition='outside',
        textfont=dict(color='#94a3b8', size=10)
    ))
    
    # Style the chart
    fig.update_layout(
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color='#94a3b8', size=9),
            tickangle=-45  # Angle labels for readability
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(30,58,95,0.3)',
            showticklabels=False  # Hide y-axis numbers
        ),
        bargap=0.4
    )
    
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch', key='countries_chart')

def render_trending(c, t):
    """
    Display trending news table with intensity labels.
    
    Shows stories sorted by article count (most coverage = most trending).
    Includes:
    - Date
    - Intensity level (Armed Conflict, Crisis, etc.)
    - Headline
    - Region
    - Article count (📰)
    - Link (🔗)
    
    Args:
        c: Database connection
        t: Table name
    """
    df = get_trending(c, t)
    
    if df.empty:
        st.info("🔥 Loading...")
        return
    
    # Process and limit to 15 rows
    df = process_df(df).head(15)
    
    if df.empty:
        st.info("🔥 No stories")
        return
    
    # Display table with precise column widths
    st.dataframe(
        df[['DATE_FMT', 'INTENSITY', 'HEADLINE', 'REGION', 'ARTICLE_COUNT', 'NEWS_LINK']], 
        hide_index=True, 
        height=400,
        column_config={
            "DATE_FMT": st.column_config.TextColumn("Date", width=60),
            "INTENSITY": st.column_config.TextColumn("Intensity", width=140),
            "HEADLINE": st.column_config.TextColumn("Story", width=None),  # Auto-width
            "REGION": st.column_config.TextColumn("Region", width=100),
            "ARTICLE_COUNT": st.column_config.NumberColumn("📰", width=50),
            "NEWS_LINK": st.column_config.LinkColumn("🔗", width=40)
        }, 
        width='stretch'
    )

def render_feed(c, t):
    """
    Display chronological feed of recent events.
    
    Shows events in reverse chronological order (newest first).
    Similar to trending but sorted by date instead of article count.
    
    Includes:
    - Date
    - Intensity Level
    - Event headline
    - Region
    - Link (🔗)
    
    Args:
        c: Database connection
        t: Table name
    """
    df = get_feed(c, t)
    
    if df.empty:
        st.info("📋 Loading...")
        return
    
    # Process and limit to 30 rows
    df = process_df(df).head(30)
    
    if df.empty:
        st.info("📋 No events")
        return
    
    # Display table with precise column widths
    st.dataframe(
        df[['DATE_FMT', 'INTENSITY', 'HEADLINE', 'REGION', 'NEWS_LINK']], 
        hide_index=True, 
        height=600,  # Taller table for more events
        column_config={
            "DATE_FMT": st.column_config.TextColumn("Date", width=60),
            "INTENSITY": st.column_config.TextColumn("Intensity Level", width=140),
            "HEADLINE": st.column_config.TextColumn("Event", width=None),  # Auto-width
            "REGION": st.column_config.TextColumn("Region", width=100),
            "NEWS_LINK": st.column_config.LinkColumn("🔗", width=40)
        }, 
        width='stretch'
    )

def render_timeseries(c, t):
    """
    Display 30-day trend line chart.
    
    Shows three lines over time:
    - Total events (cyan, filled area)
    - Negative events (red line)
    - Positive events (green line)
    
    Helps visualize whether news is getting more positive or negative.
    
    Args:
        c: Database connection
        t: Table name
    """
    df = get_timeseries(c, t)
    
    if df.empty:
        st.info("📈 Loading...")
        return
    
    # Convert date strings to actual dates for plotting
    df['date'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    
    # Create figure with dual y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add total events line with filled area
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['events'], 
            fill='tozeroy',  # Fill area to zero
            fillcolor='rgba(6,182,212,0.15)',  # Light cyan fill
            line=dict(color='#06b6d4', width=2),
            name='Total'
        ), 
        secondary_y=False
    )
    
    # Add negative events line
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['negative'], 
            line=dict(color='#ef4444', width=2),
            name='Negative'
        ), 
        secondary_y=True
    )
    
    # Add positive events line
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['positive'], 
            line=dict(color='#10b981', width=2),
            name='Positive'
        ), 
        secondary_y=True
    )
    
    # Style the chart
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=30,b=0),
        showlegend=True,
        legend=dict(
            orientation='h', 
            y=1.02, 
            font=dict(size=11, color='#94a3b8')
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(30,58,95,0.3)',
            tickfont=dict(color='#64748b')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(30,58,95,0.3)',
            tickfont=dict(color='#64748b')
        ),
        hovermode='x unified'  # Show all values when hovering
    )
    
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch', key='timeseries_chart')

def render_ai_chat(c, sql_db):
    """
    Display AI-powered chat interface for querying the database.
    
    Users can ask questions in plain English like:
    - "What major events happened this week?"
    - "Top 5 countries by event count"
    - "Show crisis-level events"
    
    The AI converts these to SQL queries automatically using Cerebras.
    
    Features:
    - Chat history (stores last 8 messages)
    - Example questions
    - SQL query display (expandable)
    - Smart data cleaning and formatting
    - Fallback queries if AI fails
    
    Args:
        c: Database connection
        sql_db: SQL database wrapper for AI
    """
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Helper to render previous conversations selector
    if st.session_state.qa_history:
        # Take all but the latest as "past" and keep only last 5
        past_convos = st.session_state.qa_history[:-1] if len(st.session_state.qa_history) > 1 else st.session_state.qa_history
        past_convos = past_convos[-5:]  # up to 5 past convos

        with st.expander("🕒 Previous Conversations", expanded=False):
            def label_for(idx):
                q = past_convos[idx]["question"]
                q = q.strip().replace("\n", " ")
                return (q[:70] + "…") if len(q) > 70 else q

            selected_idx = st.selectbox(
                "Select a past question",
                options=list(range(len(past_convos))),
                format_func=label_for,
                key="prev_convo_select"
            )

            selected = past_convos[selected_idx]

            # Show the selected Q&A in a compact, themed card
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem;margin-top:0.5rem;">
                <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.5rem;">💬 Previous Conversation</div>
                <div style="margin-bottom:0.75rem;">
                    <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:0.25rem;"><b>Q:</b></div>
                    <div style="font-size:0.8rem;color:#e2e8f0;">{selected['question']}</div>
                </div>
                <div>
                    <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:0.25rem;"><b>A:</b></div>
                    <div style="font-size:0.8rem;color:#e2e8f0;">{selected['answer']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # If we stored SQL for that convo, show it like a mini SQL trace
            if selected.get("sql"):
                with st.expander("🔍 SQL Query (for this conversation)"):
                    st.code(selected["sql"], language="sql")

    # Info card with example questions
    st.markdown('''
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;
                    padding:0.75rem;margin-bottom:1rem;">
            <div style="color:#64748b;font-size:0.7rem;margin-bottom:0.5rem;">💡 EXAMPLE QUESTIONS:</div>
            <div style="color:#94a3b8;font-size:0.75rem;line-height:1.8;">
                • "What major events happened this week?"<br>
                • "Top 5 countries by event count"<br>
                • "Show crisis-level events"<br>
                • "What are the most severe events?"
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # We no longer replay the full chat history here – just handle the current interaction.
    prompt = st.chat_input("Ask about global events...", key="chat")

    if prompt:
        # Show current user message
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Get query engine
            qe = get_query_engine(sql_db)
            if not qe:
                st.error("❌ AI not available")
                return
            
            # Get Cerebras LLM
            cerebras_llm = get_cerebras_llm()
            if not cerebras_llm:
                st.error("❌ Cerebras AI not available")
                return

            try:
                dates = get_dates()

                short_prompt = f"""Query: "{prompt}"
Table: events_dagster
Columns: DATE (VARCHAR YYYYMMDD), MAIN_ACTOR, ACTOR_COUNTRY_CODE (3-letter ISO code), IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK

Rules:
- DATE is VARCHAR like '20251210'
- Always include: WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL
- For recent data use: DATE >= '{dates['week_ago']}'
- LIMIT 10 max
- ALWAYS include NEWS_LINK in SELECT

Return only the SQL query."""

                sql = None  # we'll fill this if/when we get a SQL query
                data = None  # will hold query results

                with st.spinner("🔍 Querying..."):
                    sql = None
                    
                    # PRIORITY #1: Crisis/severe queries (BEFORE AI - guaranteed correct filter)
                    if 'crisis' in prompt.lower() or 'severe' in prompt.lower():
                        sql = (
                            "SELECT DATE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, "
                            "ARTICLE_COUNT, NEWS_LINK FROM events_dagster "
                            f"WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL "
                            f"AND IMPACT_SCORE < -3 AND DATE >= '{dates['week_ago']}' "
                            "ORDER BY IMPACT_SCORE ASC LIMIT 10"
                        )
                        logger.info(f"Using crisis SQL: {sql}")
                    
                    # Otherwise, ask AI to generate SQL
                    else:
                        response = qe.query(short_prompt)
                        sql = response.metadata.get('sql_query')
                        logger.info(f"AI Generated SQL: {sql}")

                    # FALLBACK: Top countries query  
                    if not sql and ('top' in prompt.lower() and 'countr' in prompt.lower()):
                        limit = 5
                        import re as _re
                        match = _re.search(r'top\s+(\d+)', prompt.lower())
                        if match:
                            limit = int(match.group(1))
                        sql = (
                            "SELECT ACTOR_COUNTRY_CODE, COUNT(*) as count FROM events_dagster "
                            "WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL "
                            f"AND DATE >= '{dates['week_ago']}' "
                            f"GROUP BY ACTOR_COUNTRY_CODE ORDER BY count DESC LIMIT {limit}"
                        )
                        logger.info(f"Using top countries SQL: {sql}")

                    # FALLBACK: "What happened" queries
                    if not sql and ('what' in prompt.lower() and 'happen' in prompt.lower()):
                        sql = (
                            "SELECT DATE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, "
                            "ARTICLE_COUNT, NEWS_LINK FROM events_dagster "
                            "WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL "
                            f"AND DATE >= '{dates['week_ago']}' "
                            "ORDER BY ARTICLE_COUNT DESC, DATE DESC LIMIT 10"
                        )
                        logger.info(f"Using recent events SQL: {sql}")

                    # FALLBACK: If AI generated no SQL, get most recent events
                    if not sql:
                        sql = (
                            "SELECT DATE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, "
                            "ARTICLE_COUNT, NEWS_LINK FROM events_dagster "
                            "WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL "
                            f"AND DATE >= '{dates['week_ago']}' "
                            "ORDER BY DATE DESC, ARTICLE_COUNT DESC LIMIT 10"
                        )
                        logger.info(f"Using default fallback SQL: {sql}")

                    # ========== SAFETY SAFEGUARDS TO PREVENT TOKEN DRAIN ==========
                    if sql:
                        sql_upper = sql.upper()
                        import re as _re
                        
                        # SAFEGUARD 1: Add LIMIT 10 if missing
                        if 'LIMIT' not in sql_upper:
                            sql = sql.rstrip(';') + ' LIMIT 10'
                        
                        # SAFEGUARD 2: Reduce high LIMITs to max 10
                        else:
                            limit_match = _re.search(r'LIMIT\s+(\d+)', sql_upper)
                            if limit_match and int(limit_match.group(1)) > 10:
                                sql = _re.sub(r'LIMIT\s+\d+', 'LIMIT 10', sql, flags=_re.IGNORECASE)
                        
                        logger.info(f"Safe SQL: {sql}")

                    # STEP 2: Execute SQL and get results
                    if sql:
                        data = safe_query(c, sql)
                        
                        if not data.empty:
                            # STEP 3: Clean data FIRST (before AI summarization)
                            data_display = data.copy()
                            data_display.columns = [col.upper() for col in data_display.columns]

                            if 'EVENT_ID' in data_display.columns:
                                data_display = data_display.drop(columns=['EVENT_ID'])

                            # Convert country codes to full names and filter invalid ones
                            if 'ACTOR_COUNTRY_CODE' in data_display.columns:
                                data_display['COUNTRY'] = data_display['ACTOR_COUNTRY_CODE'].apply(
                                    lambda x: get_country(x) if x and isinstance(x, str) and len(x.strip()) > 0 else None
                                )
                                data_display = data_display[data_display['COUNTRY'].notna()]
                                data_display = data_display.drop(columns=['ACTOR_COUNTRY_CODE'])

                            # Rename count columns
                            if 'COUNT(*)' in data_display.columns:
                                data_display = data_display.rename(columns={'COUNT(*)': 'EVENTS'})
                            elif 'COUNT' in data_display.columns:
                                data_display = data_display.rename(columns={'COUNT': 'EVENTS'})

                            # Add severity labels
                            if 'IMPACT_SCORE' in data_display.columns:
                                data_display['SEVERITY'] = data_display['IMPACT_SCORE'].apply(get_impact_label)

                            # Format dates
                            if 'DATE' in data_display.columns:
                                try:
                                    data_display['DATE'] = pd.to_datetime(
                                        data_display['DATE'].astype(str), format='%Y%m%d'
                                    ).dt.strftime('%b %d')
                                except:
                                    pass

                            # Remove rows only if COUNTRY is missing (essential field)
                            if 'COUNTRY' in data_display.columns:
                                data_display = data_display[data_display['COUNTRY'].notna()]

                            # Rename link column
                            if 'NEWS_LINK' in data_display.columns:
                                data_display = data_display.rename(columns={'NEWS_LINK': '🔗'})

                            # Reorder columns
                            if 'EVENTS' in data_display.columns:
                                preferred_order = ['COUNTRY', 'EVENTS', 'MAIN_ACTOR']
                            else:
                                preferred_order = ['DATE', 'COUNTRY', 'MAIN_ACTOR', 'SEVERITY', 'IMPACT_SCORE', 'ARTICLE_COUNT']

                            link_cols = [col for col in data_display.columns if '🔗' in col]
                            other_cols = [col for col in data_display.columns if col not in preferred_order and col not in link_cols]
                            final_order = [col for col in preferred_order if col in data_display.columns] + other_cols + link_cols
                            data_display = data_display[final_order]

                            # STEP 4: Generate AI summary with safeguards
                            if not data_display.empty:
                                # SAFEGUARD 4: Limit to 5 rows max for AI
                                ai_data = data_display.head(5)
                                
                                # SAFEGUARD 6: Remove links from AI (not useful in text)
                                ai_cols = [c for c in ai_data.columns if '🔗' not in c]
                                summary_data = ai_data[ai_cols].to_string(index=False)
                                
                                # SAFEGUARD 7: Cap data at 3000 chars (reasonable limit)
                                if len(summary_data) > 3000:
                                    summary_data = summary_data[:3000]
                                
                                new_prompt = f"""Query: {prompt}

Events data:
{summary_data}

Based on this data, write a news-style summary. For each event:
- Explain what likely happened in the real world (the actual news story)
- Don't just repeat the table values - interpret and explain the significance
- Write 2-3 sentences per event with real-world context

Focus on the news story, not the data columns."""
                                
                                response_og = cerebras_llm.complete(new_prompt)
                                answer = str(response_og)
                                st.markdown(answer)

                                # STEP 5: Display table (same data as summary)
                                col_config = {
                                    "DATE": st.column_config.TextColumn("DATE", width="small"),
                                    "COUNTRY": st.column_config.TextColumn("COUNTRY", width="medium"),
                                    "EVENTS": st.column_config.NumberColumn("EVENTS", format="%d", width="small"),
                                    "MAIN_ACTOR": st.column_config.TextColumn("ACTOR", width="medium"),
                                    "SEVERITY": st.column_config.TextColumn("SEVERITY", width="medium"),
                                    "IMPACT_SCORE": st.column_config.NumberColumn("SCORE", format="%.1f", width="small"),
                                    "ARTICLE_COUNT": st.column_config.NumberColumn("ARTICLES", width="small"),
                                }
                                for col in link_cols:
                                    col_config[col] = st.column_config.LinkColumn(col, width="small")

                                st.dataframe(
                                    data_display.head(10),
                                    hide_index=True,
                                    width='stretch',
                                    column_config=col_config
                                )
                            else:
                                st.info("📭 No valid results after filtering")
                                answer = "No valid results found after filtering."

                            with st.expander("🔍 SQL Query"):
                                st.code(sql, language='sql')
                        else:
                            st.warning("📭 No results found")
                            answer = "No results found for your query."
                    else:
                        # No SQL generated, show original AI response
                        answer = str(response)
                        st.markdown(answer)
                        st.warning("⚠️ Could not generate SQL query. Try rephrasing your question or use one of the examples above.")

                    # Save this Q&A into compact history
                    st.session_state.qa_history.append({
                        "question": prompt,
                        "answer": answer,
                        "sql": sql
                    })

            except Exception as e:
                error_msg = str(e)
                if "MAX_TOKENS" in error_msg:
                    st.error("⚠️ Response too long. Try: 'Top 5 countries by event count'")
                else:
                    st.error(f"❌ Error: {error_msg[:100]}")
                logger.error(f"AI error: {e}")

def render_arch():
    """
    Display the Architecture page.
    
    Shows:
    - Pipeline diagram (GDELT → Dagster → dbt → MotherDuck → Cerebras → Streamlit)
    - 4 component cards (Data Ingestion, Transformation, Data Warehouse, AI Layer)
    - Tech stack badges (12 technologies)
    
    This helps recruiters understand the technical implementation.
    """
    st.markdown("""
    <div style="text-align:center;margin-bottom:2rem;">
        <h2 style="font-family:JetBrains Mono;color:#e2e8f0;">🏗️ System Architecture</h2>
        <p style="color:#64748b;">End-to-end data pipeline processing 100K+ daily events</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline flow diagram
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:2rem;text-align:center;margin-bottom:2rem;">
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">📰 GDELT API</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">⚡ Dagster Orchestration</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">🔧 dbt Transformations</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">🦆 MotherDuck DWH</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">🤖 Cerebras AI</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">🎨 Streamlit</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Component cards in 2 columns
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;margin-bottom:1rem;min-height:260px;">
            <h4 style="color:#06b6d4;font-size:0.9rem;">📥 DATA INGESTION</h4>
            <p style="color:#94a3b8;font-size:0.85rem;">GDELT Project monitors 100+ languages, 100K+ daily events</p>
            <ul style="color:#94a3b8;font-size:0.85rem;">
                <li>15-minute update intervals</li>
                <li>GitHub Actions scheduler</li>
                <li>Dagster orchestration</li>
                <li>Incremental loads</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;min-height:200px;">
            <h4 style="color:#10b981;font-size:0.9rem;">🔧 TRANSFORMATION</h4>
            <ul style="color:#94a3b8;font-size:0.85rem;">
                <li>dbt models for data quality</li>
                <li>Aggregations & metrics</li>
                <li>Country code mapping</li>
                <li>Impact scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;margin-bottom:1rem;min-height:260px;">
            <h4 style="color:#f59e0b;font-size:0.9rem;">🗄️ DATA WAREHOUSE</h4>
            <p style="color:#94a3b8;font-size:0.85rem;">Migrated from Snowflake → MotherDuck</p>
            <ul style="color:#94a3b8;font-size:0.85rem;">
                <li>DuckDB columnar format</li>
                <li>Sub-second queries</li>
                <li>Serverless architecture</li>
                <li><b>Saved:</b> ~$100/month vs Snowflake</li>
            </ul>
            <div style="margin-top:0.5rem;padding:0.5rem;background:rgba(16,185,129,0.1);border-radius:6px;border-left:3px solid #10b981;">
                <span style="color:#10b981;font-size:0.75rem;">💰 CURRENT COST: $0/month</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;min-height:200px;">
            <h4 style="color:#8b5cf6;font-size:0.9rem;">🤖 AI LAYER</h4>
            <ul style="color:#94a3b8;font-size:0.85rem;">
                <li>Cerebras Llama 3.1 8B</li>
                <li>Previously tested: Gemini, Groq</li>
                <li>LlamaIndex text-to-SQL</li>
                <li>Natural language queries</li>
                <li>Free tier usage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tech stack badges
    st.markdown("""
    <h3 style="text-align:center;color:#e2e8f0;margin-bottom:1rem;">🛠️ Tech Stack Evolution</h3>
    <div style="text-align:center;padding:1rem;">
        <span class="tech-badge">🐍 Python</span>
        <span class="tech-badge">❄️ Snowflake</span>
        <span class="tech-badge">🦆 DuckDB</span>
        <span class="tech-badge">☁️ MotherDuck</span>
        <span class="tech-badge">⚙️ Dagster</span>
        <span class="tech-badge">🔧 dbt</span>
        <span class="tech-badge">🤖 Gen AI</span>
        <span class="tech-badge">🦙 LlamaIndex</span>
        <span class="tech-badge">⚡ Cerebras</span>
        <span class="tech-badge">📊 Plotly</span>
        <span class="tech-badge">🎨 Streamlit</span>
        <span class="tech-badge">🔄 GitHub Actions</span>
    </div>
        <div style="margin-top:1rem;">
            <div style="color:#64748b;font-size:0.75rem;margin-bottom:0.5rem;text-align:center;">PREVIOUSLY TESTED</div>
            <div style="text-align:center;">
                <span style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:0.4rem 0.8rem;display:inline-block;margin:0.25rem;font-size:0.75rem;color:#64748b;">❄️ Snowflake</span>
                <span style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:0.4rem 0.8rem;display:inline-block;margin:0.25rem;font-size:0.75rem;color:#64748b;">✨ Gemini</span>
                <span style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:0.4rem 0.8rem;display:inline-block;margin:0.25rem;font-size:0.75rem;color:#64748b;">⚡ Groq</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_about():
    """
    Display the About page.
    
    Shows:
    - Project description and goals
    - Technical skills demonstrated
    - Project highlights (100K+ events, $0 cost, <1s queries, 100+ languages)
    - Contact links (GitHub, LinkedIn)
    
    This is the "portfolio piece" that explains the project to recruiters.
    """
    st.markdown("""
    <div style="text-align:center;padding:2rem 0;">
        <h2 style="font-family:JetBrains Mono;color:#e2e8f0;">👋 About This Project</h2>
        <p style="color:#94a3b8;max-width:750px;margin:0 auto 1.5rem;font-size:1.1rem;">
            Real-time analytics for <b>GDELT</b> — the world's largest open database monitoring global news in 100+ languages
        </p>
        <p style="color:#64748b;max-width:700px;margin:0 auto 2rem;">
            This project showcases end-to-end data engineering, from ingestion to visualization, with AI-powered natural language querying
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;min-height:280px;">
            <h4 style="color:#06b6d4;font-size:0.9rem;">🎯 PROJECT GOALS</h4>
            <ul style="color:#94a3b8;font-size:0.85rem;line-height:1.8;">
                <li>Demonstrate production-ready data pipelines</li>
                <li>Showcase modern data stack (Dagster, dbt, DuckDB)</li>
                <li>Integrate AI/LLM capabilities (Cerebras, LlamaIndex)</li>
                <li>Build scalable, cost-effective architecture</li>
                <li>Create intuitive data visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;min-height:280px;">
            <h4 style="color:#10b981;font-size:0.9rem;">🛠️ TECHNICAL SKILLS</h4>
            <ul style="color:#94a3b8;font-size:0.85rem;line-height:1.8;">
                <li><b>Languages:</b> Python, SQL</li>
                <li><b>Data Engineering:</b> ETL/ELT, Data Modeling</li>
                <li><b>Orchestration:</b> Dagster, dbt, GitHub Actions</li>
                <li><b>Cloud:</b> Snowflake, MotherDuck (DuckDB)</li>
                <li><b>AI/ML:</b> LLMs, RAG, Text-to-SQL</li>
                <li><b>Visualization:</b> Streamlit, Plotly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project highlights grid
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:2rem;margin:2rem 0;">
        <h4 style="color:#e2e8f0;text-align:center;margin-bottom:1rem;">📈 PROJECT HIGHLIGHTS</h4>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;">
            <div style="text-align:center;padding:1rem;background:rgba(6,182,212,0.1);border-radius:8px;">
                <div style="font-size:2rem;font-weight:700;color:#06b6d4;">100K+</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Daily Events Processed</div>
            </div>
            <div style="text-align:center;padding:1rem;background:rgba(16,185,129,0.1);border-radius:8px;">
                <div style="font-size:2rem;font-weight:700;color:#10b981;">$0</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Monthly Operating Cost</div>
            </div>
            <div style="text-align:center;padding:1rem;background:rgba(245,158,11,0.1);border-radius:8px;">
                <div style="font-size:2rem;font-weight:700;color:#f59e0b;">&lt;1s</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Average Query Time</div>
            </div>
            <div style="text-align:center;padding:1rem;background:rgba(139,92,246,0.1);border-radius:8px;">
                <div style="font-size:2rem;font-weight:700;color:#8b5cf6;">100+</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Languages Monitored</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cost efficiency section - using Streamlit columns for reliable layout
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:2rem;margin:2rem 0;">
        <h4 style="color:#e2e8f0;text-align:center;margin-bottom:1.5rem;">💰 COST-EFFICIENT ARCHITECTURE</h4>
        <p style="color:#94a3b8;font-size:0.9rem;text-align:center;margin-bottom:1rem;">Built with cost optimization in mind, avoiding expensive enterprise tools while maintaining production-grade quality</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Streamlit columns for side-by-side layout (more reliable than CSS grid)
    cost_col1, cost_col2 = st.columns(2)
    
    with cost_col1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;height:100%;">
            <h5 style="color:#f59e0b;font-size:0.9rem;margin-bottom:1rem;">❌ AVOIDED (Expensive)</h5>
            <ul style="font-size:0.85rem;line-height:1.8;color:#94a3b8;padding-left:1.2rem;">
                <li>Apache Spark / PySpark (~$5-10k/month)</li>
                <li>Hadoop clusters (~$3-8k/month)</li>
                <li>Azure Synapse (~$2-5k/month)</li>
                <li>AWS Redshift (~$2-4k/month)</li>
                <li>Databricks (~$3-7k/month)</li>
                <li>Snowflake compute (~$1-3k/month)</li>
                <li>OpenAI GPT-4 API (~$500-1k/month)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with cost_col2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;height:100%;">
            <h5 style="color:#10b981;font-size:0.9rem;margin-bottom:1rem;">✅ USED INSTEAD (Free/Cheap)</h5>
            <ul style="font-size:0.85rem;line-height:1.8;color:#94a3b8;padding-left:1.2rem;">
                <li><b style="color:#e2e8f0;">DuckDB:</b> In-process analytics (free)</li>
                <li><b style="color:#e2e8f0;">MotherDuck:</b> Serverless DuckDB ($0 free tier)</li>
                <li><b style="color:#e2e8f0;">Dagster:</b> Orchestration (free self-hosted)</li>
                <li><b style="color:#e2e8f0;">dbt:</b> Transformations (free core)</li>
                <li><b style="color:#e2e8f0;">GitHub Actions:</b> CI/CD (free tier)</li>
                <li><b style="color:#e2e8f0;">Cerebras:</b> LLM inference (pay-as-you-go)</li>
                <li><b style="color:#e2e8f0;">Streamlit:</b> Free hosting + dashboards</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Savings summary
    st.markdown("""
    <div style="margin-top:1rem;padding:1.25rem;background:rgba(16,185,129,0.1);border-radius:12px;border-left:4px solid #10b981;text-align:center;">
        <div style="font-size:1.3rem;font-weight:700;color:#10b981;margin-bottom:0.5rem;">Total Monthly Savings: $15,000 - $40,000</div>
        <div style="font-size:0.85rem;color:#94a3b8;">Achieved enterprise-scale data processing at near-zero cost</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Contact section
    st.markdown("""
    <div style="text-align:center;">
        <h4 style="color:#e2e8f0;">📬 CONTACT</h4>
        <p style="color:#94a3b8;margin-bottom:1rem;">Interested in data engineering roles or collaborations</p>
        <div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;">
            <a href="https://github.com/Mohith-akash" target="_blank" style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1.25rem;color:#e2e8f0;text-decoration:none;display:inline-block;">
                ⭐ GitHub
            </a>
            <a href="https://www.linkedin.com/in/mohith-akash/" target="_blank" style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1.25rem;color:#e2e8f0;text-decoration:none;display:inline-block;">
                💼 LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SECTION 12: MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application entry point.
    
    This function:
    1. Injects CSS styling
    2. Connects to database
    3. Initializes AI
    4. Renders header
    5. Creates 5 tabs (HOME, TRENDS, AI, TECH, ABOUT)
    6. Populates each tab with appropriate content
    7. Shows footer
    
    This is what runs when you launch the app!
    """
    # Apply custom CSS
    inject_css()
    
    # Get database connection
    conn = get_db()
    tbl = detect_table(conn)
    
    # Initialize AI
    sql_db = get_ai_engine(get_engine())
    
    # Render header
    render_header()
    
    # Create 5 tabs
    tabs = st.tabs(["📊 HOME", "📈 TRENDS", "🤖 AI", "🏗️ TECH", "👤 ABOUT"])
    
    # ========== HOME TAB ==========
    with tabs[0]:
        # Metrics at top
        render_metrics(conn, tbl)
        
        # Live ticker
        render_ticker(conn, tbl)
        
        st.markdown("---")
        
        # Main content: Trending + Sentiment
        c1, c2 = st.columns([6, 4])
        
        with c1:
            st.markdown('<div class="card-hdr"><span>🔥</span><span class="card-title">Trending News</span></div>', unsafe_allow_html=True)
            render_trending(conn, tbl)
        
        with c2:
            st.markdown('<div class="card-hdr"><span>⚡</span><span class="card-title">Weekly Sentiment</span></div>', unsafe_allow_html=True)
            render_sentiment(conn, tbl)
        
        st.markdown("---")
        
        # Bottom section: Actors + Distribution + Countries
        c1, c2 = st.columns([6, 4])
        
        with c1:
            st.markdown('<div class="card-hdr"><span>🎯</span><span class="card-title">Most Mentioned</span></div>', unsafe_allow_html=True)
            render_actors(conn, tbl)
        
        with c2:
            st.markdown('<div class="card-hdr"><span>📊</span><span class="card-title">Tone Breakdown</span></div>', unsafe_allow_html=True)
            render_distribution(conn, tbl, 'home_distribution')
            
            st.markdown('<div class="card-hdr" style="margin-top:1rem;"><span>🏆</span><span class="card-title">Top Countries</span></div>', unsafe_allow_html=True)
            render_countries(conn, tbl)
    
    # ========== TRENDS TAB ==========
    with tabs[1]:
        # Layout: Recent events + Intensity guide + Distribution
        c1, c2 = st.columns([7, 3])
        
        with c1:
            st.markdown('<div class="card-hdr"><span>📋</span><span class="card-title">Recent Events Feed</span></div>', unsafe_allow_html=True)
            render_feed(conn, tbl)
        
        with c2:
            # Intensity level guide
            st.markdown('<div class="card-hdr"><span>📊</span><span class="card-title">Intensity Guide</span></div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.25rem;margin-bottom:1rem;">
                <h4 style="color:#06b6d4;font-size:0.85rem;margin-bottom:1rem;">🎯 EVENT INTENSITY LEVELS</h4>
                <div style="padding:0.5rem;margin:0.5rem 0;background:rgba(239,68,68,0.1);border-left:3px solid #ef4444;border-radius:4px;">
                    <div style="font-weight:600;color:#ef4444;">⚔️ Armed Conflict</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">Score: -10 to -8 • Military action</div>
                </div>
                <div style="padding:0.5rem;margin:0.5rem 0;background:rgba(239,68,68,0.1);border-left:3px solid #ef4444;border-radius:4px;">
                    <div style="font-weight:600;color:#ef4444;">🔴 Major Crisis</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">Score: -7 to -6 • Severe situation</div>
                </div>
                <div style="padding:0.5rem;margin:0.5rem 0;background:rgba(245,158,11,0.1);border-left:3px solid #f59e0b;border-radius:4px;">
                    <div style="font-weight:600;color:#f59e0b;">🟠 Serious Tension</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">Score: -5 to -4 • High stakes</div>
                </div>
                <div style="padding:0.5rem;margin:0.5rem 0;background:rgba(234,179,8,0.1);border-left:3px solid #eab308;border-radius:4px;">
                    <div style="font-weight:600;color:#eab308;">🟡 Verbal Dispute</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">Score: -3 to -2 • Disagreements</div>
                </div>
                <div style="padding:0.5rem;margin:0.5rem 0;background:rgba(148,163,184,0.1);border-left:3px solid #94a3b8;border-radius:4px;">
                    <div style="font-weight:600;color:#94a3b8;">⚪ Neutral Event</div>
                    <div style="font-size:0.7rem;color:#64748b;">Score: -2 to 2 • Standard news</div>
                </div>
                <div style="padding:0.5rem;margin:0.5rem 0;background:rgba(16,185,129,0.1);border-left:3px solid #10b981;border-radius:4px;">
                    <div style="font-weight:600;color:#10b981;">🟢 Diplomatic Talk</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">Score: 2 to 4 • Cooperation</div>
                </div>
                <div style="padding:0.5rem;margin:0.5rem 0;background:rgba(16,185,129,0.1);border-left:3px solid #10b981;border-radius:4px;">
                    <div style="font-weight:600;color:#10b981;">🤝 Active Partnership</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">Score: 4 to 6 • Joint efforts</div>
                </div>
                <div style="padding:0.5rem;margin:0.5rem 0;background:rgba(6,182,212,0.1);border-left:3px solid #06b6d4;border-radius:4px;">
                    <div style="font-weight:600;color:#06b6d4;">✨ Peace Agreement</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">Score: 6+ • Major accord</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # This week distribution
            st.markdown('<div class="card-hdr" style="margin-top:1rem;"><span>📈</span><span class="card-title">This Week</span></div>', unsafe_allow_html=True)
            render_distribution(conn, tbl, 'trends_distribution')
        
        st.markdown("---")
        
        # 30-day trend chart
        st.markdown('<div class="card-hdr"><span>📈</span><span class="card-title">30-Day Trend Analysis</span></div>', unsafe_allow_html=True)
        render_timeseries(conn, tbl)
    
    # ========== AI TAB ==========
    with tabs[2]:
        # Layout: Chat + Info sidebar
        c1, c2 = st.columns([7, 3])
        
        with c1:
            st.markdown('<div class="card-hdr"><span>🤖</span><span class="card-title">Ask in Plain English</span></div>', unsafe_allow_html=True)
            render_ai_chat(conn, sql_db)
        
        with c2:
            # How it works card
            st.markdown("""
            <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.25rem;">
                <h4 style="color:#06b6d4;font-size:0.85rem;">ℹ️ HOW IT WORKS</h4>
                <p style="color:#94a3b8;font-size:0.8rem;">Your question → Cerebras AI → SQL query → Results with links</p>
                <hr style="border-color:#1e3a5f;margin:1rem 0;">
                <p style="color:#94a3b8;font-size:0.75rem;">📅 Dates: YYYYMMDD<br>👤 Actors: People/Orgs<br>📊 Impact: -10 to +10<br>🔗 Links: News sources</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== TECH TAB ==========
    with tabs[3]:
        render_arch()
    
    # ========== ABOUT TAB ==========
    with tabs[4]:
        render_about()
    
    # Footer
    st.markdown('''
        <div style="text-align:center;padding:2rem 0 1rem;border-top:1px solid #1e3a5f;margin-top:2rem;">
            <p style="color:#64748b;font-size:0.8rem;">
                <b>GDELT</b> monitors worldwide news in real-time • 100K+ daily events
            </p>
            <p style="color:#475569;font-size:0.75rem;">
                Built by <a href="https://www.linkedin.com/in/mohith-akash/" style="color:#06b6d4;">Mohith Akash</a> • Portfolio Project
            </p>
        </div>
    ''', unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

# This is where the program starts!
# When you run "streamlit run app.py", Python looks for this block
# and executes the main() function, which starts the whole application.
if __name__ == "__main__":
    main()