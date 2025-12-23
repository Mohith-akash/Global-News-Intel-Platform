import duckdb
import os
from dotenv import load_dotenv

load_dotenv()
con = duckdb.connect(f'md:gdelt_db?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}')

# Check most recent ingestion
recent = con.execute("""
    SELECT DATE, HEADLINE 
    FROM events_dagster 
    ORDER BY DATE DESC 
    LIMIT 20
""").fetchdf()

print(recent)
con.close()