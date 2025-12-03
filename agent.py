import os
import sys
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine
import duckdb

# --- CONFIGURATION ---
load_dotenv() 

GEMINI_MODEL_NAME = "models/gemini-2.5-flash-preview-09-2025"
GEMINI_EMBED_MODEL = "models/embedding-001"

def get_motherduck_engine():
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise ValueError("❌ Missing MOTHERDUCK_TOKEN in .env file")
    
    # SQLAlchemy connection string for MotherDuck
    return create_engine(f'duckdb:///md:gdelt_db?motherduck_token={token}')

def build_query_engine():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing GOOGLE_API_KEY in .env file")
    
    llm = Gemini(model=GEMINI_MODEL_NAME, api_key=api_key)
    embed_model = GeminiEmbedding(model_name=GEMINI_EMBED_MODEL, api_key=api_key)
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    engine = get_motherduck_engine()
    
    # Connect to table (MotherDuck is cleaner with case)
    sql_database = SQLDatabase(engine, include_tables=["events_dagster"])

    return NLSQLTableQueryEngine(sql_database=sql_database, llm=llm)

if __name__ == "__main__":
    try:
        print("🦆 Connecting to MotherDuck Agent...")
        agent = build_query_engine()
        print("--- Agent Online ---")
        
        # Test Query
        response = agent.query("How many events are in the database?")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")