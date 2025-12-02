import requests
import datetime
import zipfile
import io
import pandas as pd
import snowflake.connector
import os
import time
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

GDELT_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
TARGET_TABLE = "EVENTS_DAGSTER"
DAYS_TO_BACKFILL = 14
BATCH_SIZE = 50  # Process 50 GDELT files before uploading to Snowflake (Optimizes network)

# Snowflake Config
SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "role": "ACCOUNTADMIN"
}

def get_snowflake_conn():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)

def fetch_gdelt_urls(days):
    """
    Parses the massive GDELT master list to find ALL files for the last N days.
    """
    print("📋 Fetching GDELT Master File List (This might take 10-20 seconds)...")
    try:
        response = requests.get(GDELT_MASTER_URL)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch master list: {e}")
        return []

    # Generate list of date strings we care about (e.g., '20250101', '20250102')
    target_dates = []
    today = datetime.datetime.utcnow()
    for i in range(days):
        date_str = (today - datetime.timedelta(days=i)).strftime("%Y%m%d")
        target_dates.append(date_str)
    
    print(f"📅 Looking for data from: {target_dates[-1]} to {target_dates[0]}")

    valid_urls = []
    # Stream process the lines to save memory
    for line in response.text.splitlines():
        # Check if line contains any of our target dates AND is an export CSV
        if "export.CSV.zip" in line:
            # GDELT format: ID HASH URL
            parts = line.split(" ")
            if len(parts) > 2:
                url = parts[-1]
                # Check if URL contains our date pattern (e.g. /20250101120000.export...)
                if any(d in url for d in target_dates):
                    valid_urls.append(url)
    
    print(f"✅ Found {len(valid_urls)} data files to ingest.")
    return valid_urls

def process_and_upload(urls):
    """
    Downloads batches of URLs, converts to DataFrame, and uploads to Snowflake.
    """
    conn = get_snowflake_conn()
    total_uploaded = 0
    
    # Process in chunks
    for i in range(0, len(urls), BATCH_SIZE):
        batch_urls = urls[i : i + BATCH_SIZE]
        print(f"\n🔄 Processing Batch {i // BATCH_SIZE + 1} ({len(batch_urls)} files)...")
        
        batch_dfs = []
        
        for url in batch_urls:
            try:
                r = requests.get(url, timeout=10)
                if r.status_code != 200:
                    continue
                    
                z = zipfile.ZipFile(io.BytesIO(r.content))
                csv_filename = z.namelist()[0]
                
                with z.open(csv_filename) as f:
                    # Parse only columns we need
                    df = pd.read_csv(f, sep='\t', header=None, 
                                     usecols=[0, 1, 6, 7, 29, 30, 31, 34, 60])
                    
                    df.columns = [
                        "EVENT_ID", "DATE", "MAIN_ACTOR", "ACTOR_COUNTRY_CODE", 
                        "EVENT_CATEGORY_CODE", "IMPACT_SCORE", "ARTICLE_COUNT", 
                        "SENTIMENT_SCORE", "NEWS_LINK"
                    ]
                    
                    # Data Cleaning
                    df['EVENT_ID'] = df['EVENT_ID'].astype(str)
                    df['DATE'] = df['DATE'].astype(str)
                    batch_dfs.append(df)
                    
            except Exception as e:
                print(f"⚠️ Skipped file {url}: {e}")
                continue
        
        if not batch_dfs:
            continue
            
        # Combine all DFs in this batch
        master_df = pd.concat(batch_dfs, ignore_index=True)
        
        if master_df.empty:
            continue

        print(f"   ⬆️ Uploading {len(master_df)} rows to Snowflake...")
        
        try:
            success, n_chunks, n_rows, _ = write_pandas(
                conn, 
                master_df, 
                TARGET_TABLE, 
                auto_create_table=True,
                overwrite=False  # IMPORTANT: False means APPEND to existing data
            )
            total_uploaded += n_rows
            print(f"   ✅ Batch Complete. Total uploaded so far: {total_uploaded:,}")
        except Exception as e:
            print(f"   ❌ Batch Upload Failed: {e}")

    conn.close()
    return total_uploaded

if __name__ == "__main__":
    print("🚀 Starting 14-Day Backfill Operation...")
    
    urls = fetch_gdelt_urls(DAYS_TO_BACKFILL)
    
    if urls:
        total = process_and_upload(urls)
        print(f"\n🎉 BACKFILL COMPLETE!")
        print(f"📊 Total Incidents Ingested: {total:,}")
        print("IMPORTANT: Now run the 'CREATE OR REPLACE TABLE' SQL command in Snowflake to update your clean table.")
    else:
        print("No URLs found. Check connection.")