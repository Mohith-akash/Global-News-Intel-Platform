import requests
import datetime
import zipfile
import io
import pandas as pd
import snowflake.connector
import os
from snowflake.connector.pandas_tools import write_pandas
from dagster import asset, Output, Definitions, ScheduleDefinition, define_asset_job
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

# Snowflake Config
SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "GDELT"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "RAW"),
    "role": "ACCOUNTADMIN"
}

TARGET_TABLE = "EVENTS_DAGSTER"

# --- HELPER: Get Latest GDELT URL ---
def get_gdelt_url():
    """
    Constructs the URL for the previous 15-minute interval to ensure file exists.
    """
    # Go back 15 minutes to be safe (GDELT takes time to upload)
    now = datetime.datetime.utcnow() - datetime.timedelta(minutes=15)
    
    # Round down to nearest 15 minutes
    rounded_minute = (now.minute // 15) * 15
    rounded_time = now.replace(minute=rounded_minute, second=0, microsecond=0)
    
    timestamp = rounded_time.strftime("%Y%m%d%H%M00")
    url = f"http://data.gdeltproject.org/gdeltv2/{timestamp}.export.CSV.zip"
    
    print(f"🕒 Generated Timestamp: {timestamp}")
    print(f"🔗 Target URL: {url}")
    return url

# --- ASSETS ---

@asset
def gdelt_raw_data() -> pd.DataFrame:
    print(f"--- Starting Hourly Extraction ---")
    
    url = get_gdelt_url()
    
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            print(f"⚠️ File not found (Status {r.status_code}). Skipping this run.")
            return pd.DataFrame() # Return empty DF, don't crash
            
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_filename = z.namelist()[0]
        
        print(f"📂 Processing: {csv_filename}")
        with z.open(csv_filename) as f:
            df = pd.read_csv(f, sep='\t', header=None, 
                             usecols=[0, 1, 6, 7, 29, 30, 31, 34, 60])
        
        df.columns = [
            "EVENT_ID", "DATE", "MAIN_ACTOR", "ACTOR_COUNTRY_CODE", 
            "EVENT_CATEGORY_CODE", "IMPACT_SCORE", "ARTICLE_COUNT", 
            "SENTIMENT_SCORE", "NEWS_LINK"
        ]
        
        # Enforce Types
        df['EVENT_ID'] = df['EVENT_ID'].astype(str)
        df['DATE'] = df['DATE'].astype(str)
        
        print(f"✅ Extracted {len(df)} rows.")
        return df

    except Exception as e:
        print(f"❌ Extraction Error: {e}")
        return pd.DataFrame()

@asset
def gdelt_snowflake_table(gdelt_raw_data: pd.DataFrame) -> Output:
    df = gdelt_raw_data
    
    if df.empty:
        print("⚠️ No data to load. Exiting.")
        return Output(None, metadata={"status": "Skipped"})

    print(f"🔌 Connecting to Snowflake...")
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    
    try:
        # 1. Write Data
        success, n_chunks, n_rows, _ = write_pandas(
            conn, 
            df, 
            TARGET_TABLE, 
            auto_create_table=True,
            overwrite=False
        )
        
        # 2. CRITICAL: Commit the transaction!
        conn.commit()
        
        print(f"🎉 SUCCESS: Inserted {n_rows} rows into {TARGET_TABLE}.")
        
    except Exception as e:
        print(f"❌ Load Error: {e}")
        raise e # Fail the run so we know
        
    finally:
        conn.close()
    
    return Output(f"Uploaded {n_rows} rows", metadata={"rows": n_rows})

# --- JOB DEFINITIONS ---
gdelt_job = define_asset_job(name="gdelt_ingestion_job", selection="*")

gdelt_schedule = ScheduleDefinition(
    job=gdelt_job,
    cron_schedule="0 * * * *", 
    execution_timezone="UTC"
)

defs = Definitions(
    assets=[gdelt_raw_data, gdelt_snowflake_table],
    jobs=[gdelt_job],
    schedules=[gdelt_schedule]
)