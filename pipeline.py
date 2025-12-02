import requests
import datetime
import zipfile
import io
import pandas as pd
import snowflake.connector
import os
from snowflake.connector.pandas_tools import write_pandas
from dagster import asset, Output, MetadataValue, Definitions, ScheduleDefinition, define_asset_job
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

GDELT_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
TARGET_TABLE = "EVENTS_DAGSTER"

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

# --- ASSETS ---

@asset
def gdelt_raw_data() -> pd.DataFrame:
    """
    Ingests the latest GDELT Event batch.
    """
    print(f"--- Starting Extraction Pipeline ---")
    
    # GDELT updates every 15 mins. We look for the MOST RECENT file.
    # We grab the master list and find the last entry.
    try:
        response = requests.get(GDELT_MASTER_URL)
        response.raise_for_status()
        
        # Get the last line that is an export CSV
        lines = response.text.splitlines()
        target_url = None
        
        # Iterate backwards to find the latest English export
        for line in reversed(lines):
            if "export.CSV.zip" in line:
                target_url = line.split(" ")[-1]
                break
                
        if not target_url:
            print("Warning: No URL found.")
            return pd.DataFrame()
            
        print(f"Found Target URL: {target_url}")

        r = requests.get(target_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_filename = z.namelist()[0]
        
        print("Parsing CSV...")
        with z.open(csv_filename) as f:
            df = pd.read_csv(f, sep='\t', header=None, 
                             usecols=[0, 1, 6, 7, 29, 30, 31, 34, 60])
        
        df.columns = [
            "EVENT_ID", "DATE", "MAIN_ACTOR", "ACTOR_COUNTRY_CODE", 
            "EVENT_CATEGORY_CODE", "IMPACT_SCORE", "ARTICLE_COUNT", 
            "SENTIMENT_SCORE", "NEWS_LINK"
        ]
        
        # Data Type Enforcement
        df['EVENT_ID'] = df['EVENT_ID'].astype(str)
        df['DATE'] = df['DATE'].astype(str)
        
        print(f"Extraction Complete: {len(df)} rows ready for loading.")
        return df

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

@asset
def gdelt_snowflake_table(gdelt_raw_data: pd.DataFrame) -> Output:
    """
    Loads data into Snowflake.
    """
    df = gdelt_raw_data
    
    if df.empty:
        return Output(None, metadata={"status": "Skipped - No Data"})

    print(f"Connecting to Snowflake Account: {SNOWFLAKE_CONFIG['account']}...")
    
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    
    try:
        cursor = conn.cursor()
        cursor.execute(f"USE WAREHOUSE {SNOWFLAKE_CONFIG['warehouse']}")
        cursor.execute(f"USE DATABASE {SNOWFLAKE_CONFIG['database']}")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_CONFIG['schema']}")
        cursor.close()

        success, n_chunks, n_rows, _ = write_pandas(
            conn, 
            df, 
            TARGET_TABLE, 
            auto_create_table=True,
            overwrite=False # Append new data, don't overwrite!
        )
        print(f"Upload Successful: {n_rows} rows inserted into {TARGET_TABLE}.")
        
    finally:
        conn.close()
    
    return Output(
        f"Uploaded {n_rows} rows", 
        metadata={
            "Row Count": n_rows,
            "Target Table": TARGET_TABLE
        }
    )

# --- JOB & SCHEDULE ---

# 1. Define a Job that materializes the assets
gdelt_job = define_asset_job(name="gdelt_ingestion_job", selection="*")

# 2. Define a Schedule (Run every hour)
# Cron expression: "0 * * * *" means "At minute 0 of every hour"
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