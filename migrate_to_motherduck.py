import snowflake.connector
import duckdb
import os
from dotenv import load_dotenv
import pandas as pd

# Load secrets
load_dotenv()

# 1. Connect to Snowflake
conn_snow = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)

# 2. Export Data to Parquet (Local File)
print("🚀 Starting Export from Snowflake...")
query = "SELECT * FROM EVENTS_DAGSTER"

# We use DuckDB to read from Snowflake cursor efficiently if possible, 
# but for simplicity/stability with 11m rows, we stream to Parquet.
# NOTE: If you have <16GB RAM, we might need to chunk this. 
# Let's try a direct Pandas Arrow write which is fast.

cur = conn_snow.cursor()
cur.execute(query)

print("⏳ Fetching data (this may take a few minutes)...")
# Fetch as Arrow Batches (Fastest way)
df = cur.fetch_pandas_all()

print(f"✅ Downloaded {len(df)} rows. Saving to Parquet...")
df.to_parquet("gdelt_backup.parquet", index=False)
print("✅ Local Backup Created: gdelt_backup.parquet")

# Close Snowflake
conn_snow.close()

# 3. Upload to MotherDuck
print("🚀 Connecting to MotherDuck...")
# Connect to MotherDuck using your token
# md: means "MotherDuck" (Cloud), plain string means local file
md_token = os.getenv("MOTHERDUCK_TOKEN")
conn_duck = duckdb.connect(f'md:gdelt_db?motherduck_token={md_token}')

print("⏳ Uploading to MotherDuck Cloud...")
# Create table directly from the local parquet file
conn_duck.execute("""
    CREATE OR REPLACE TABLE events_dagster AS 
    SELECT * FROM 'gdelt_backup.parquet'
""")

print("🎉 MIGRATION COMPLETE! Check your MotherDuck UI.")