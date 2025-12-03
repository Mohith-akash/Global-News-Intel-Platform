import streamlit as st
import os
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, text, inspect
import datetime
import pycountry
import logging
import streamlit.components.v1 as components
import re
from urllib.parse import urlparse
import duckdb

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Global Intelligence Platform", 
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed" # Clean look, sidebar hidden by default
)
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gip")

# Validation
REQUIRED_ENVS = ["MOTHERDUCK_TOKEN", "GOOGLE_API_KEY"]
missing = [k for k in REQUIRED_ENVS if not os.getenv(k)]
if missing:
    st.error(f"❌ CRITICAL ERROR: Missing env vars: {', '.join(missing)}")
    st.stop()

# Constants
GEMINI_MODEL = "models/gemini-2.5-flash-preview-09-2025"
GEMINI_EMBED_MODEL = "models/embedding-001"

# --- 2. STYLING ---
def style_app():
    st.markdown("""
    <style>
        .stApp { background-color: #0b0f19; }
        /* Navbar Buttons */
        div[data-testid="stColumn"] button {
            width: 100%;
            background-color: #1e293b;
            color: white;
            border: 1px solid #334155;
            border-radius: 5px;
            height: 50px;
            font-weight: bold;
        }
        div[data-testid="stColumn"] button:hover {
            border-color: #3b82f6;
            color: #3b82f6;
        }
        /* Hide default elements */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        /* Metrics & Tables */
        div[data-testid="stMetric"] { background-color: #111827; border: 1px solid #374151; border-radius: 8px; padding: 15px; }
        div[data-testid="stChatMessage"] { background-color: #1f2937; border: 1px solid #374151; border-radius: 12px; margin-bottom: 10px; }
        .report-box { background-color: #1e293b; padding: 25px; border-radius: 10px; border: 1px solid #475569; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BACKEND ---

@st.cache_resource
def get_db_connection():
    token = os.getenv("MOTHERDUCK_TOKEN")
    return duckdb.connect(f'md:gdelt_db?motherduck_token={token}')

@st.cache_resource
def get_sql_engine():
    token = os.getenv("MOTHERDUCK_TOKEN")
    return create_engine(f'duckdb:///md:gdelt_db?motherduck_token={token}')

def safe_read_sql(conn, query):
    try:
        return conn.execute(query).df()
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return pd.DataFrame()

def is_safe_sql(sql: str) -> bool:
    if not sql: return False
    low = sql.lower()
    banned = ["delete ", "update ", "drop ", "alter ", "insert ", "grant ", "revoke ", "--"]
    return not any(b in low for b in banned)

def format_headline(url, actor):
    if not url: return f"Update on {actor}"
    try:
        path = urlparse(url).path
        slug = path.rstrip('/').split('/')[-1]
        if len(slug) < 5 or slug.isdigit() or 'index' in slug.lower():
            slug = path.rstrip('/').split('/')[-2]
        text = slug.replace('-', ' ').replace('_', ' ').replace('+', ' ')
        text = re.sub(r'\.html?$', '', text)
        text = re.sub(r'\b20\d{2}[\s/-]?\d{1,2}[\s/-]?\d{1,2}\b', '', text) 
        text = re.sub(r'\b\d{8}\b', '', text) 
        if re.search(r'[A-Za-z0-9]{15,}', text): return f"Latest Intelligence: {actor}"
        text = re.sub(r'^(article|story|news|report|default)\s*', '', text, flags=re.IGNORECASE)
        headline = " ".join(text.split()).title()
        if len(headline) < 5: return f"Update on {actor}"
        return headline
    except:
        return f"Briefing: {actor}"

@st.cache_resource
def get_query_engine(_engine):
    api_key = os.getenv("GOOGLE_API_KEY")
    try:
        llm = Gemini(model=GEMINI_MODEL, api_key=api_key)
        embed_model = GeminiEmbedding(model_name=GEMINI_EMBED_MODEL, api_key=api_key)
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        inspector = inspect(_engine)
        combined_names = inspector.get_table_names() + inspector.get_view_names()
        target_table = next((t for t in combined_names if t.upper() == "EVENTS_DAGSTER"), None)
        
        if not target_table: return None
        
        sql_database = SQLDatabase(_engine, include_tables=[target_table])
        query_engine = NLSQLTableQueryEngine(sql_database=sql_database, llm=llm)
        
        update_str = (
            "You are a Geopolitical Intelligence AI. Querying 'EVENTS_DAGSTER'.\n"
            "**RULES:**\n"
            "1. **INCLUDE LINKS:** ALWAYS select the `NEWS_LINK` column.\n"
            "2. **NO NULLS:** Add `WHERE IMPACT_SCORE IS NOT NULL`.\n"
            "3. **NULLS LAST:** Use `ORDER BY [col] DESC NULLS LAST`.\n"
            "4. **DIALECT:** Use DuckDB/Postgres syntax.\n"
            "5. **Response:** Return SQL in metadata."
        )
        query_engine.update_prompts({"text_to_sql_prompt": update_str})
        return query_engine
    except: return None

# --- 4. VIEW LOGIC ---

def set_view(view_name):
    st.session_state.current_view = view_name

# --- 5. UI SECTIONS ---

def render_chat_section(conn_ui, engine_ai):
    # Input Area at the Top
    with st.container():
        c1, c2 = st.columns([4, 1])
        with c1:
            prompt = st.chat_input("Ask the Global Analyst...", key="chat_input_main")
        with c2:
            if st.button("Clear History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        # Example Pills
        examples = ["🚨 Recent Conflicts", "🇺🇳 UN Activities", "📈 China vs USA", "📉 Economic Instability"]
        cols = st.columns(len(examples))
        selected_example = None
        for i, ex in enumerate(examples):
            if cols[i].button(ex, use_container_width=True):
                selected_example = ex

    # Process Input
    user_input = prompt or selected_example
    
    if user_input:
        if 'llm_locked' not in st.session_state: st.session_state.llm_locked = False
        
        if not st.session_state.llm_locked:
            st.session_state.llm_locked = True
            
            # Add User Message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate Response
            with st.spinner("Analyzing Global Streams..."):
                try:
                    qe = get_query_engine(engine_ai)
                    if qe:
                        response_obj = qe.query(user_input)
                        response_text = response_obj.response
                        
                        # Check for SQL Data
                        sql_data = None
                        if hasattr(response_obj, 'metadata') and 'sql_query' in response_obj.metadata:
                            sql = response_obj.metadata['sql_query']
                            if is_safe_sql(sql):
                                df = safe_read_sql(conn_ui, sql)
                                if not df.empty:
                                    df.columns = [c.upper() for c in df.columns]
                                    if 'NEWS_LINK' in df.columns:
                                        # Format for chat display
                                        sql_data = df
                        
                        # Add Assistant Message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_text,
                            "data": sql_data
                        })
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")
                finally:
                    st.session_state.llm_locked = False

    # Display History (Newest on Top, Limit 5)
    # We reverse the list and slice the last 5
    history_to_show = list(reversed(st.session_state.messages))[:5]
    
    st.markdown("### 💬 Recent Analysis")
    for msg in history_to_show:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If there is attached data, show it
            if "data" in msg and msg["data"] is not None:
                st.dataframe(
                    msg["data"],
                    column_config={"NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read")},
                    hide_index=True
                )

def render_trending_section(conn):
    st.subheader("🔥 Global Trending News")
    sql = """
        SELECT NEWS_LINK, ACTOR_COUNTRY_CODE, ARTICLE_COUNT, MAIN_ACTOR
        FROM EVENTS_DAGSTER 
        WHERE NEWS_LINK IS NOT NULL 
        ORDER BY ARTICLE_COUNT DESC 
        LIMIT 50
    """
    df = safe_read_sql(conn, sql)
    if not df.empty:
        df.columns = [c.upper() for c in df.columns]
        df['Headline'] = df.apply(lambda x: format_headline(x['NEWS_LINK'], x['MAIN_ACTOR']), axis=1)
        df = df.drop_duplicates(subset=['Headline']).head(20)
        
        st.dataframe(
            df[['Headline', 'ACTOR_COUNTRY_CODE', 'ARTICLE_COUNT', 'NEWS_LINK']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Headline": st.column_config.TextColumn("Topic", width="large"),
                "ACTOR_COUNTRY_CODE": st.column_config.TextColumn("Country", width="small"),
                "ARTICLE_COUNT": st.column_config.NumberColumn("Reports", format="%d 📉"),
                "NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read Article")
            }
        )
    else:
        st.info("No trending data available.")

def render_feed_section(conn):
    st.subheader("📋 Real-Time Intelligence Feed")
    sql = """
        SELECT 
            DATE, 
            NEWS_LINK, 
            MAX(MAIN_ACTOR) as MAIN_ACTOR,
            AVG(IMPACT_SCORE) as IMPACT_SCORE 
        FROM EVENTS_DAGSTER 
        WHERE NEWS_LINK IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1 DESC 
        LIMIT 50
    """
    df = safe_read_sql(conn, sql)
    if not df.empty:
        df.columns = [c.upper() for c in df.columns]
        df['Headline'] = df.apply(lambda x: format_headline(x['NEWS_LINK'], x['MAIN_ACTOR']), axis=1)
        try:
            df['Date'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
        except: df['Date'] = df['DATE']

        def get_type(score):
            if score < -3: return "🔥 Conflict"
            if score > 3: return "🤝 Diplomacy"
            return "📢 General"
        df['Type'] = df['IMPACT_SCORE'].apply(get_type)

        st.dataframe(
            df[['Date', 'Headline', 'Type', 'NEWS_LINK']], 
            use_container_width=True, 
            hide_index=True, 
            column_config={
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Headline": st.column_config.TextColumn("Headline", width="large"),
                "Type": st.column_config.TextColumn("Category", width="small"),
                "NEWS_LINK": st.column_config.LinkColumn("Link", display_text="🔗 Read")
            }
        )
    else:
        st.info("No feed data.")

def render_visuals_section(conn):
    st.subheader("🌐 Global Conflict Map (3D)")
    df = safe_read_sql(conn, """
        SELECT ACTOR_COUNTRY_CODE as "Country", COUNT(*) as "Events", AVG(IMPACT_SCORE) as "Impact" 
        FROM EVENTS_DAGSTER 
        WHERE ACTOR_COUNTRY_CODE IS NOT NULL 
        GROUP BY 1
    """)
    if not df.empty:
        fig = px.choropleth(
            df, 
            locations="Country", 
            locationmode='ISO-3', 
            color="Events", 
            hover_name="Country", 
            hover_data=["Impact"], 
            color_continuous_scale="Viridis", 
            template="plotly_dark"
        )
        fig.update_geos(
            projection_type="orthographic", 
            showcoastlines=True, 
            showland=True, 
            landcolor="#0f172a", 
            showocean=True, 
            oceancolor="#1e293b"
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Map Data available.")

# --- 6. MAIN ---
def main():
    style_app()
    
    # Init Session State
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "Chat"
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Database Connections
    conn_ui = get_db_connection()
    engine_ai = get_sql_engine()

    # --- HEADER & NAVBAR ---
    c_title, c_nav = st.columns([1, 2])
    with c_title:
        st.title("🦅 Global Intel")
    
    with c_nav:
        st.markdown("<br>", unsafe_allow_html=True) # Spacing
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("💬 Chat", use_container_width=True): set_view("Chat")
        if b2.button("🔥 Trending", use_container_width=True): set_view("Trending")
        if b3.button("📋 Feed", use_container_width=True): set_view("Feed")
        if b4.button("🌐 Visuals", use_container_width=True): set_view("Visuals")

    st.markdown("---") # Horizontal Line

    # --- MAIN CONTENT AREA ---
    view = st.session_state.current_view
    
    if view == "Chat":
        render_chat_section(conn_ui, engine_ai)
    elif view == "Trending":
        render_trending_section(conn_ui)
    elif view == "Feed":
        render_feed_section(conn_ui)
    elif view == "Visuals":
        render_visuals_section(conn_ui)

if __name__ == "__main__":
    main()