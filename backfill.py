import requests
import datetime
import zipfile
import io
import pandas as pd
import duckdb
import os
import time
from dotenv import load_dotenv

load_dotenv()
TARGET_TABLE = "events_dagster"
TOKEN = os.getenv("MOTHERDUCK_TOKEN")

def get_urls_for_range(days_back=3):
    """Generate GDELT URLs for every 15 mins for the last N days"""
    urls = []
    now = datetime.datetime.utcnow()
    # Round down to nearest 15 min
    start_time = now - datetime.timedelta(days=days_back)
    current = start_time
    
    while current < now:
        minute = (current.minute // 15) * 15
        time_str = current.replace(minute=minute, second=0).strftime("%Y%m%d%H%M00")
        url = f"http://data.gdeltproject.org/gdeltv2/{time_str}.export.CSV.zip"
        urls.append(url)
        current += datetime.timedelta(minutes=15)
    return urls

def run_backfill():
    print("🚀 Starting Backfill Process...")
    urls = get_urls_for_range(days_back=2) # Load last 48 hours
    print(f"📦 Found {len(urls)} files to process")
    
    con = duckdb.connect(f'md:gdelt_db?motherduck_token={TOKEN}')
    
    count = 0
    for url in urls:
        try:
            r = requests.get(url)
            if r.status_code != 200: continue
            
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]
            
            # USE YOUR FIXED 9 COLUMNS
            with z.open(csv_name) as f:
                df = pd.read_csv(f, sep='\t', header=None, 
                               usecols=[0, 1, 6, 7, 29, 30, 31, 34, 60]) # NO 53
            
            df.columns = ["EVENT_ID", "DATE", "MAIN_ACTOR", "ACTOR_COUNTRY_CODE", 
                          "EVENT_CATEGORY_CODE", "IMPACT_SCORE", "ARTICLE_COUNT", 
                          "SENTIMENT_SCORE", "NEWS_LINK"] # NO LOCATION_NAME
            
            # Clean Types
            df['EVENT_ID'] = df['EVENT_ID'].astype(str)
            df['DATE'] = df['DATE'].astype(str)
            
            # Insert
            con.execute(f"INSERT INTO {TARGET_TABLE} SELECT * FROM df")
            count += 1
            print(f"✅ Loaded: {url.split('/')[-1]} ({len(df)} rows)")
            
        except Exception as e:
            print(f"❌ Failed: {url} - {e}")
            
    print(f"🎉 Backfill Complete! Processed {count} files.")

if __name__ == "__main__":
    run_backfill()
