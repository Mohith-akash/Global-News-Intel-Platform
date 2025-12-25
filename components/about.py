"""
About page component for GDELT dashboard.
"""

import streamlit as st


def render_about():
    """About page with architecture, tool comparison, and evolution."""
    
    # TITLE
    st.markdown("""
    <div style="text-align:center;padding:0.75rem 0;">
        <h2 style="font-family:JetBrains Mono;color:#e2e8f0;font-size:1.5rem;margin:0;">🏗️ About This Project</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # ARCHITECTURE - Compact single row with subtitles
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:0.75rem 0;margin-bottom:0.5rem;">
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:0.6rem 0.5rem;text-align:center;flex:1;margin-right:0.2rem;">
            <div style="font-size:1.5rem;">📰</div>
            <div style="color:#e2e8f0;font-size:0.75rem;font-weight:600;">GDELT</div>
            <div style="color:#64748b;font-size:0.6rem;">100K+ events</div>
        </div>
        <span style="color:#06b6d4;font-size:1.1rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:0.6rem 0.5rem;text-align:center;flex:1;margin:0 0.2rem;">
            <div style="font-size:1.5rem;">⚡</div>
            <div style="color:#e2e8f0;font-size:0.75rem;font-weight:600;">Dagster</div>
            <div style="color:#64748b;font-size:0.6rem;">Orchestration</div>
        </div>
        <span style="color:#06b6d4;font-size:1.1rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:0.6rem 0.5rem;text-align:center;flex:1;margin:0 0.2rem;">
            <div style="font-size:1.5rem;">🦆</div>
            <div style="color:#e2e8f0;font-size:0.75rem;font-weight:600;">MotherDuck</div>
            <div style="color:#64748b;font-size:0.6rem;">DWH + Vectors</div>
        </div>
        <span style="color:#06b6d4;font-size:1.1rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:0.6rem 0.5rem;text-align:center;flex:1.3;margin:0 0.2rem;">
            <div style="font-size:1.5rem;">🚀 / 🦙</div>
            <div style="color:#e2e8f0;font-size:0.75rem;font-weight:600;">Voyage / LlamaIndex</div>
            <div style="color:#64748b;font-size:0.6rem;">RAG | SQL Mode</div>
        </div>
        <span style="color:#06b6d4;font-size:1.1rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:0.6rem 0.5rem;text-align:center;flex:1;margin:0 0.2rem;">
            <div style="font-size:1.5rem;">🧠</div>
            <div style="color:#e2e8f0;font-size:0.75rem;font-weight:600;">Cerebras</div>
            <div style="color:#64748b;font-size:0.6rem;">Llama 3.1 8B</div>
        </div>
        <span style="color:#06b6d4;font-size:1.1rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:0.6rem 0.5rem;text-align:center;flex:1;margin-left:0.2rem;">
            <div style="font-size:1.5rem;">🎨</div>
            <div style="color:#e2e8f0;font-size:0.75rem;font-weight:600;">Streamlit</div>
            <div style="color:#64748b;font-size:0.6rem;">Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ENTERPRISE vs MY STACK - Full width, bigger font
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:1rem;">
        <h4 style="color:#e2e8f0;text-align:center;margin-bottom:0.75rem;font-size:1.1rem;">💰 ENTERPRISE TOOLS vs MY STACK</h4>
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
            <tr style="border-bottom:1px solid #1e3a5f;">
                <th style="text-align:left;padding:0.4rem;color:#f59e0b;width:30%;">Enterprise Tool</th>
                <th style="text-align:left;padding:0.4rem;color:#10b981;width:18%;">My Stack</th>
                <th style="text-align:left;padding:0.4rem;color:#64748b;">How I Replaced It</th>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.4rem;color:#94a3b8;"><b>Databricks/Spark</b> <span style="color:#ef4444;font-size:0.7rem;">~$500/mo</span></td>
                <td style="padding:0.4rem;color:#e2e8f0;"><b>DuckDB</b></td>
                <td style="padding:0.4rem;color:#64748b;">Columnar OLAP for 100K+ events — runs in-process</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.4rem;color:#94a3b8;"><b>Snowflake/BigQuery</b> <span style="color:#ef4444;font-size:0.7rem;">~$300/mo</span></td>
                <td style="padding:0.4rem;color:#e2e8f0;"><b>MotherDuck</b></td>
                <td style="padding:0.4rem;color:#64748b;">Serverless cloud DWH with vector search built-in</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.4rem;color:#94a3b8;"><b>Managed Airflow</b> <span style="color:#ef4444;font-size:0.7rem;">~$300/mo</span></td>
                <td style="padding:0.4rem;color:#e2e8f0;"><b>Dagster</b></td>
                <td style="padding:0.4rem;color:#64748b;">Asset-based DAGs with GitHub Actions scheduling</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.4rem;color:#94a3b8;"><b>Pinecone/Weaviate</b> <span style="color:#ef4444;font-size:0.7rem;">~$70/mo</span></td>
                <td style="padding:0.4rem;color:#e2e8f0;"><b>MotherDuck</b></td>
                <td style="padding:0.4rem;color:#64748b;">DuckDB native vector search (array_cosine_similarity)</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.4rem;color:#94a3b8;"><b>OpenAI Embeddings</b> <span style="color:#ef4444;font-size:0.7rem;">~$50/mo</span></td>
                <td style="padding:0.4rem;color:#e2e8f0;"><b>Voyage AI</b></td>
                <td style="padding:0.4rem;color:#64748b;">200M free tokens — creates RAG embeddings</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.4rem;color:#94a3b8;"><b>OpenAI GPT-4</b> <span style="color:#ef4444;font-size:0.7rem;">~$100/mo</span></td>
                <td style="padding:0.4rem;color:#e2e8f0;"><b>Cerebras</b></td>
                <td style="padding:0.4rem;color:#64748b;">Llama 3.1 8B free tier — Text-to-SQL + RAG</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.4rem;color:#94a3b8;"><b>dbt Cloud</b> <span style="color:#ef4444;font-size:0.7rem;">~$100/mo</span></td>
                <td style="padding:0.4rem;color:#e2e8f0;"><b>SQL in Python</b></td>
                <td style="padding:0.4rem;color:#64748b;">Transformations in pipeline.py — same result</td>
            </tr>
            <tr>
                <td style="padding:0.4rem;color:#94a3b8;"><b>Tableau/Power BI</b> <span style="color:#ef4444;font-size:0.7rem;">~$70/mo</span></td>
                <td style="padding:0.4rem;color:#e2e8f0;"><b>Streamlit</b></td>
                <td style="padding:0.4rem;color:#64748b;">Python dashboards with Plotly — free hosting</td>
            </tr>
        </table>
        <div style="display:flex;justify-content:space-around;margin-top:1rem;padding-top:1rem;border-top:1px solid #1e3a5f;">
            <div style="text-align:center;">
                <div style="color:#ef4444;font-size:1.5rem;font-weight:700;">$1,490+</div>
                <div style="color:#64748b;font-size:0.8rem;">Enterprise monthly</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#10b981;font-size:1.75rem;font-weight:700;">$0</div>
                <div style="color:#64748b;font-size:0.8rem;">My monthly cost</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # TWO COLUMNS - Evolution (left) + Tech Stack with Metrics (right)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # EVOLUTION - Half width
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:1rem;margin-top:0.75rem;">
            <h4 style="color:#e2e8f0;text-align:center;margin-bottom:0.75rem;font-size:0.95rem;">🔄 TECHNOLOGY EVOLUTION</h4>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;margin-bottom:0.5rem;">
                <div><span style="color:#06b6d4;font-size:0.7rem;">DATA WAREHOUSE</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">❄️ Snowflake → 🦆 <b>MotherDuck</b></span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">$0 cost, same SQL + vector search</div>
            </div>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;margin-bottom:0.5rem;">
                <div><span style="color:#06b6d4;font-size:0.7rem;">RAG EMBEDDINGS</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">🚀 <b>Voyage AI</b></span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">Vector embeddings + semantic search</div>
            </div>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;margin-bottom:0.5rem;">
                <div><span style="color:#8b5cf6;font-size:0.7rem;">AI CHAT</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">🦙 LlamaIndex + 🧠 Cerebras</span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">Text-to-SQL + RAG dual mode</div>
            </div>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;">
                <div><span style="color:#10b981;font-size:0.7rem;">ETL PIPELINE</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">⚙️ Dagster + GitHub Actions</span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">30min schedule, auto-embeds headlines</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # TECH STACK + KEY METRICS combined
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:1rem;margin-top:0.75rem;">
            <h4 style="color:#e2e8f0;text-align:center;margin-bottom:0.75rem;font-size:0.95rem;">🛠️ TECH STACK</h4>
            <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.4rem;">
                <span style="background:#1e3a5f;border-radius:6px;padding:0.4rem 0.6rem;color:#e2e8f0;font-size:0.8rem;">🐍 Python</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.4rem 0.6rem;color:#e2e8f0;font-size:0.8rem;">📝 SQL</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.4rem 0.6rem;color:#e2e8f0;font-size:0.8rem;">🦆 DuckDB</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.4rem 0.6rem;color:#e2e8f0;font-size:0.8rem;">☁️ MotherDuck</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.4rem 0.6rem;color:#e2e8f0;font-size:0.8rem;">⚙️ Dagster</span>
            </div>
            <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.4rem;">
                <span style="background:#1e3a5f;border-radius:6px;padding:0.4rem 0.6rem;color:#e2e8f0;font-size:0.8rem;">🚀 Voyage AI</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.4rem 0.6rem;color:#e2e8f0;font-size:0.8rem;">🦙 LlamaIndex</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.4rem 0.6rem;color:#e2e8f0;font-size:0.8rem;">🧠 Cerebras</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.4rem 0.6rem;color:#e2e8f0;font-size:0.8rem;">🎨 Streamlit</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.4rem 0.6rem;color:#e2e8f0;font-size:0.8rem;">🔄 GitHub Actions</span>
            </div>
            <div style="display:flex;justify-content:space-around;padding-top:0.75rem;border-top:1px solid #1e3a5f;">
                <div style="text-align:center;">
                    <div style="font-size:1.25rem;font-weight:700;color:#06b6d4;">100K+</div>
                    <div style="font-size:0.65rem;color:#64748b;">Events/day</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.25rem;font-weight:700;color:#10b981;">$0</div>
                    <div style="font-size:0.65rem;color:#64748b;">Cost</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.25rem;font-weight:700;color:#f59e0b;">&lt;1s</div>
                    <div style="font-size:0.65rem;color:#64748b;">Query</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.25rem;font-weight:700;color:#8b5cf6;">100+</div>
                    <div style="font-size:0.65rem;color:#64748b;">Languages</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # CONTACT
    st.markdown("""
    <div style="text-align:center;margin-top:1.25rem;padding-top:1rem;border-top:1px solid #1e3a5f;">
        <span style="color:#94a3b8;font-size:0.9rem;">📬 Open to opportunities</span>
        <a href="https://github.com/Mohith-akash" target="_blank" style="margin-left:1rem;background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.5rem 1rem;color:#e2e8f0;text-decoration:none;">⭐ GitHub</a>
        <a href="https://www.linkedin.com/in/mohith-akash/" target="_blank" style="margin-left:0.5rem;background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.5rem 1rem;color:#e2e8f0;text-decoration:none;">💼 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
