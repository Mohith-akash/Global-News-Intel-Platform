"""
CSS styling for the GDELT News Intelligence Platform.
"""
import streamlit as st


def inject_css():
    """Add custom CSS styling to make the dashboard look professional."""
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
