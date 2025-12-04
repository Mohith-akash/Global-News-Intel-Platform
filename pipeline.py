import requests
import datetime
import zipfile
import io
import pandas as pd
import duckdb
import os
from dagster import asset, Output, Definitions, ScheduleDefinition, define_asset_job
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
TARGET_TABLE = "events_dagster"

# --- HELPER: Smart GDELT URL ---
def get_gdelt_url():
    """
    Calculates the URL for the GDELT CSV file from 20 minutes ago.
    This ensures we only ask for files that have definitely finished uploading.
    """
    now = datetime.datetime.utcnow() - datetime.timedelta(minutes=20)
    
    # Round down to the nearest 15-minute interval (00, 15, 30, 45)
    rounded_minute = (now.minute // 15) * 15
    rounded_time = now.replace(minute=rounded_minute, second=0, microsecond=0)
    
    timestamp = rounded_time.strftime("%Y%m%d%H%M00")
    return f"http://data.gdeltproject.org/gdeltv2/{timestamp}.export.CSV.zip"

# --- ASSETS ---

@asset
def gdelt_raw_data() -> pd.DataFrame:
    print(f"--- Starting Extraction ---")
    url = get_gdelt_url()
    print(f"🔗 Target URL: {url}")
    
    try:
        # Timeout set to 30s to prevent hanging
        r = requests.get(url, timeout=30)
        
        if r.status_code != 200:
            print(f"⚠️ File not found (Status {r.status_code}). Skipping this run.")
            return pd.DataFrame() 
            
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = z.namelist()[0]
        print(f"📂 Processing: {csv_name}")
        
        # Read specifically the columns we need to save memory
        with z.open(csv_name) as f:
            df = pd.read_csv(f, sep='\t', header=None, 
                             usecols=[0, 1, 6, 7, 29, 30, 31, 34, 60])
        
        # Rename to match our database schema
        df.columns = [
            "EVENT_ID", "DATE", "MAIN_ACTOR", "ACTOR_COUNTRY_CODE", 
            "EVENT_CATEGORY_CODE", "IMPACT_SCORE", "ARTICLE_COUNT", 
            "SENTIMENT_SCORE", "NEWS_LINK"
        ]
        
        # Ensure Types
        df['EVENT_ID'] = df['EVENT_ID'].astype(str)
        df['DATE'] = df['DATE'].astype(str)
        
        print(f"✅ Extracted {len(df)} rows.")
        return df

    except Exception as e:
        print(f"❌ Extraction Error: {e}")
        return pd.DataFrame()

@asset
def gdelt_motherduck_table(gdelt_raw_data: pd.DataFrame) -> Output:
    df = gdelt_raw_data
    
    if df.empty:
        print("⚠️ No data to load. Exiting.")
        return Output(None, metadata={"status": "Skipped"})

    print(f"🦆 Connecting to MotherDuck...")
    token = os.getenv("MOTHERDUCK_TOKEN")
    
    # Connect directly to MotherDuck using the token
    con = duckdb.connect(f'md:gdelt_db?motherduck_token={token}')
    
    try:
        # Append new data to the existing table
        con.execute(f"INSERT INTO {TARGET_TABLE} SELECT * FROM df")
        print(f"🎉 Inserted {len(df)} rows into MotherDuck.")
    except Exception as e:
        # Fallback: If table doesn't exist yet, create it
        print(f"ℹ️ Table check: {e}. Attempting to create table...")
        con.execute(f"CREATE TABLE IF NOT EXISTS {TARGET_TABLE} AS SELECT * FROM df")
        print(f"🎉 Created table with {len(df)} rows.")
        
    return Output(f"Uploaded {len(df)} rows", metadata={"rows": len(df)})

# --- JOB DEFINITIONS ---

# Define the job that runs the assets
gdelt_job = define_asset_job(name="gdelt_ingestion_job", selection="*")

# Define the schedule (Safe Mode: Every 30 Minutes)
gdelt_schedule = ScheduleDefinition(
    job=gdelt_job,
    cron_schedule="*/30 * * * *", 
    execution_timezone="UTC"
)

# Bundle everything into Definitions for Dagster to read
defs = Definitions(
    assets=[gdelt_raw_data, gdelt_motherduck_table],
    jobs=[gdelt_job],
    schedules=[gdelt_schedule]
)