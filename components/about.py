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
    
    # ARCHITECTURE - Full width edge to edge
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:1rem 0;margin-bottom:0.5rem;">
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin-right:0.5rem;">
            <div style="font-size:1.75rem;">📰</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">GDELT API</div>
            <div style="color:#64748b;font-size:0.7rem;">100K+ events/day</div>
        </div>
        <span style="color:#06b6d4;font-size:1.75rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin:0 0.5rem;">
            <div style="font-size:1.75rem;">⚡</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">Dagster</div>
            <div style="color:#64748b;font-size:0.7rem;">Orchestration</div>
        </div>
        <span style="color:#06b6d4;font-size:1.75rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin:0 0.5rem;">
            <div style="font-size:1.75rem;">🦆</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">MotherDuck</div>
            <div style="color:#64748b;font-size:0.7rem;">Cloud DuckDB</div>
        </div>
        <span style="color:#06b6d4;font-size:1.75rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin:0 0.5rem;">
            <div style="font-size:1.75rem;">🦙</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">LlamaIndex</div>
            <div style="color:#64748b;font-size:0.7rem;">Text-to-SQL</div>
        </div>
        <span style="color:#06b6d4;font-size:1.75rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin:0 0.5rem;">
            <div style="font-size:1.75rem;">🧠</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">Cerebras</div>
            <div style="color:#64748b;font-size:0.7rem;">Llama 3.1 8B</div>
        </div>
        <span style="color:#06b6d4;font-size:1.75rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin-left:0.5rem;">
            <div style="font-size:1.75rem;">🎨</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">Streamlit</div>
            <div style="color:#64748b;font-size:0.7rem;">Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ENTERPRISE vs MY STACK - Full width, bigger font
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:1rem;">
        <h4 style="color:#e2e8f0;text-align:center;margin-bottom:0.75rem;font-size:1.1rem;">💰 ENTERPRISE TOOLS vs MY STACK</h4>
        <table style="width:100%;border-collapse:collapse;font-size:0.95rem;">
            <tr style="border-bottom:1px solid #1e3a5f;">
                <th style="text-align:left;padding:0.5rem;color:#f59e0b;width:28%;">Enterprise Tool</th>
                <th style="text-align:left;padding:0.5rem;color:#10b981;width:18%;">My Stack</th>
                <th style="text-align:left;padding:0.5rem;color:#64748b;">How I Replaced It</th>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>Spark/PySpark</b> <span style="color:#ef4444;font-size:0.75rem;">~$500/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>DuckDB</b></td>
                <td style="padding:0.5rem;color:#64748b;">Columnar OLAP for 100K+ events — no cluster needed, runs in-process</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>Snowflake/Hadoop</b> <span style="color:#ef4444;font-size:0.75rem;">~$300/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>MotherDuck</b></td>
                <td style="padding:0.5rem;color:#64748b;">Serverless cloud DWH, same SQL syntax, free tier handles my scale</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>Managed Airflow</b> <span style="color:#ef4444;font-size:0.75rem;">~$300/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>Dagster</b></td>
                <td style="padding:0.5rem;color:#64748b;">Asset-based DAGs with lineage tracking — modern orchestration UI</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>dbt Cloud</b> <span style="color:#ef4444;font-size:0.75rem;">~$100/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>SQL in Python</b></td>
                <td style="padding:0.5rem;color:#64748b;">Data transformations via raw SQL in pipeline.py — same result, no cost</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>AWS Lambda/CI</b> <span style="color:#ef4444;font-size:0.75rem;">~$100/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>GitHub Actions</b></td>
                <td style="padding:0.5rem;color:#64748b;">Scheduled ETL runs every 30 min — free CI/CD with cron triggers</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>OpenAI GPT-4</b> <span style="color:#ef4444;font-size:0.75rem;">~$50/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>Cerebras</b></td>
                <td style="padding:0.5rem;color:#64748b;">Llama 3.1 8B via free tier — fastest LLM inference for Text-to-SQL</td>
            </tr>
            <tr>
                <td style="padding:0.5rem;color:#94a3b8;"><b>Tableau/Power BI</b> <span style="color:#ef4444;font-size:0.75rem;">~$70/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>Streamlit</b></td>
                <td style="padding:0.5rem;color:#64748b;">Python-native dashboards with Plotly — free Streamlit Cloud hosting</td>
            </tr>
        </table>
        <div style="display:flex;justify-content:space-around;margin-top:1rem;padding-top:1rem;border-top:1px solid #1e3a5f;">
            <div style="text-align:center;">
                <div style="color:#ef4444;font-size:1.5rem;font-weight:700;">$1,420+</div>
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
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">Migrated for $0 cost, same SQL syntax</div>
            </div>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;margin-bottom:0.5rem;">
                <div><span style="color:#8b5cf6;font-size:0.7rem;">AI / LLM (RAG)</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">✨ Gemini → ⚡ Groq → 🧠 <b>Cerebras</b></span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">LlamaIndex RAG + Text-to-SQL pipeline</div>
            </div>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;margin-bottom:0.5rem;">
                <div><span style="color:#f59e0b;font-size:0.7rem;">MODELS</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">Llama 70B → <b>Llama 3.1 8B</b></span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">Smaller model, faster, good enough for task</div>
            </div>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;">
                <div><span style="color:#10b981;font-size:0.7rem;">ETL PIPELINE (CI/CD)</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">Manual → <b>GitHub Actions 30min</b></span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">Dagster orchestration, fully automated</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # TECH STACK + KEY METRICS combined
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:1rem;margin-top:0.75rem;">
            <h4 style="color:#e2e8f0;text-align:center;margin-bottom:0.75rem;font-size:0.95rem;">🛠️ TECH STACK</h4>
            <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.5rem;">
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🐍 Python</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">📝 SQL</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🐼 Pandas</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🦆 DuckDB</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">☁️ MotherDuck</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">⚙️ Dagster</span>
            </div>
            <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.75rem;">
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🦙 LlamaIndex</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">⚡ Cerebras</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">📊 Plotly</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🎨 Streamlit</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🔄 GitHub Actions</span>
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
