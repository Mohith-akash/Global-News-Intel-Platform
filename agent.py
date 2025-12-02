import os
import sys
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine

# --- CONFIGURATION ---
load_dotenv() 

GEMINI_MODEL_NAME = "models/gemini-2.5-flash-preview-09-2025"
GEMINI_EMBED_MODEL = "models/embedding-001"

# Snowflake Config
SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"), # FIXED: Uses .env value
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "role": "ACCOUNTADMIN"
}

def get_snowflake_engine():
    if not SNOWFLAKE_CONFIG["password"]:
        raise ValueError("Check .env for password")

    url = (
        f"snowflake://{SNOWFLAKE_CONFIG['user']}:{SNOWFLAKE_CONFIG['password']}"
        f"@{SNOWFLAKE_CONFIG['account']}/{SNOWFLAKE_CONFIG['database']}/"
        f"{SNOWFLAKE_CONFIG['schema']}?warehouse={SNOWFLAKE_CONFIG['warehouse']}"
        f"&role={SNOWFLAKE_CONFIG['role']}"
    )
    return create_engine(url)

def build_query_engine():
    api_key = os.getenv("GOOGLE_API_KEY")
    
    llm = Gemini(model=GEMINI_MODEL_NAME, api_key=api_key)
    embed_model = GeminiEmbedding(model_name=GEMINI_EMBED_MODEL, api_key=api_key)
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    engine = get_snowflake_engine()
    
    # Connect to table (Case Insensitive)
    # We use the uppercase name which is standard in Snowflake
    sql_database = SQLDatabase(engine, include_tables=["STG_GDELT_EVENTS"])

    return NLSQLTableQueryEngine(sql_database=sql_database, llm=llm)

if __name__ == "__main__":
    try:
        agent = build_query_engine()
        print("--- Agent Online ---")
        response = agent.query("What is the total number of events?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Failed: {e}")