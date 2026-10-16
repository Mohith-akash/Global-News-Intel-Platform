"""
CSS styles for GDELT dashboard.
"""
import streamlit as st


def inject_css():
    """Inject custom CSS."""
    st.markdown("""
    <style>
        /* Import modern fonts from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
        
        /* Define color palette as variables for easy reuse */
        :root { 
            --bg:#0a0e17;
            --card:#111827;
            --card-inner:#1a2332;
            --border:#1e3a5f;
            --text:#e2e8f0;
            --muted:#94a3b8;
            --dim:#64748b;
            --cyan:#06b6d4;
            --green:#10b981;
            --red:#ef4444;
            --amber:#f59e0b;
            --purple:#8b5cf6;
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
        .logo-icon { font-size: 2.5rem; }
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
        
        /* Metric cards */
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
        
        /* Card headers */
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
            background: var(--card-inner);
            color: var(--cyan);
            border-radius: 6px;
            border-bottom: none !important;
            box-shadow: none !important;
        }
        /* Kill tab highlight/underline - every known method for Streamlit */
        [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-highlight"],
        div[data-baseweb="tab-highlight"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0 !important;
            max-height: 0 !important;
            width: 0 !important;
            background: transparent !important;
            background-color: #0d1320 !important;
            border: none !important;
            position: absolute !important;
            pointer-events: none !important;
            overflow: hidden !important;
        }
        [data-baseweb="tab-border"],
        .stTabs [data-baseweb="tab-border"],
        div[data-baseweb="tab-border"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            background: transparent !important;
            background-color: #0d1320 !important;
        }
        /* Fallback: hide non-tab children of tab-list (catches renamed highlight divs) */
        [data-baseweb="tab-list"] > div:not([data-baseweb="tab"]):not([role="tab"]):not(button) {
            background: transparent !important;
            background-color: #0d1320 !important;
            height: 0 !important;
        }
        
        /* Data table styling */
        div[data-testid="stDataFrame"] { 
            background: var(--card); 
            border: 1px solid var(--border); 
            border-radius: 12px; 
        }
        div[data-testid="stDataFrame"] th { 
            background: var(--card-inner) !important; 
            color: var(--muted) !important; 
            font-size: 0.75rem; 
            text-transform: uppercase; 
        }
        
        /* Live ticker */
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
            left: 0; top: 0; bottom: 0; 
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
        .ticker-dot { 
            width: 7px; height: 7px; 
            background: var(--red); 
            border-radius: 50%; 
            animation: blink 1s infinite; 
        }
        @keyframes blink { 
            0%,100% { opacity:1; } 
            50% { opacity:0.3; } 
        }
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

        /* Horizontal divider line */
        hr { 
            border: none; 
            border-top: 1px solid var(--border); 
            margin: 1.5rem 0; 
        }
        
        /* ============================================
           CARD STYLES
           ============================================ */
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
        }

        /* ============================================
           TITLE COLORS
           ============================================ */
        .title-cyan { color: var(--cyan); font-size: 0.9rem; }
        
        /* ============================================
           TEXT STYLES
           ============================================ */
        .text-muted { color: var(--muted); font-size: 0.85rem; }
        .text-xs { font-size: 0.7rem; }

        /* ============================================
           HIGHLIGHT BOXES (stats grid)
           ============================================ */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        /* ============================================
           INTENSITY GUIDE
           ============================================ */
        .intensity-item {
            padding: 0.4rem 0.75rem;
            margin: 0.4rem 0;
            border-radius: 4px;
            border-left: 3px solid;
        }
        .intensity-red {
            background: rgba(239,68,68,0.1);
            border-left-color: var(--red);
        }
        .intensity-red .intensity-title { color: var(--red); }
        
        .intensity-amber {
            background: rgba(245,158,11,0.1);
            border-left-color: var(--amber);
        }
        .intensity-amber .intensity-title { color: var(--amber); }
        
        .intensity-yellow {
            background: rgba(234,179,8,0.1);
            border-left-color: #eab308;
        }
        .intensity-yellow .intensity-title { color: #eab308; }
        
        .intensity-gray {
            background: rgba(148,163,184,0.1);
            border-left-color: var(--muted);
        }
        .intensity-gray .intensity-title { color: var(--muted); }
        
        .intensity-green {
            background: rgba(16,185,129,0.1);
            border-left-color: var(--green);
        }
        .intensity-green .intensity-title { color: var(--green); }
        
        .intensity-cyan {
            background: rgba(6,182,212,0.1);
            border-left-color: var(--cyan);
        }
        .intensity-cyan .intensity-title { color: var(--cyan); }
        
        .intensity-title { font-weight: 600; }
        .intensity-score { font-size: 0.7rem; color: var(--muted); }

        /* ============================================
           FOOTER
           ============================================ */
        .footer {
            text-align: center;
            padding: 2rem 0 1rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }
        .footer-main {
            color: var(--dim);
            font-size: 0.8rem;
        }
        .footer-sub {
            color: #475569;
            font-size: 0.75rem;
        }
        .footer-link {
            color: var(--cyan);
        }

        /* ============================================
           AI CHAT
           ============================================ */
        .ai-info-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 1rem;
        }
        .ai-example-label {
            color: var(--dim);
            font-size: 0.7rem;
            margin-bottom: 0.5rem;
        }
        .ai-examples {
            color: var(--muted);
            font-size: 0.75rem;
            line-height: 1.8;
        }
        
        .prev-convo-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 0.5rem;
        }
        .prev-convo-label {
            font-size: 0.75rem;
            color: var(--dim);
            margin-bottom: 0.5rem;
        }
        .prev-convo-q {
            font-size: 0.75rem;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }
        .prev-convo-text {
            font-size: 0.8rem;
            color: var(--text);
        }
    </style>
    """, unsafe_allow_html=True)
