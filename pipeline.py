import requests
import datetime
import zipfile
import io
import pandas as pd
import duckdb
import os
from dagster import asset, Output, Definitions, ScheduleDefinition, define_asset_job
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
TARGET_TABLE = "events_dagster"

def get_gdelt_url():
    now = datetime.datetime.utcnow() - datetime.timedelta(minutes=20)
    rounded_minute = (now.minute // 15) * 15
    rounded_time = now.replace(minute=rounded_minute, second=0, microsecond=0)
    timestamp = rounded_time.strftime("%Y%m%d%H%M00")
    return f"http://data.gdeltproject.org/gdeltv2/{timestamp}.export.CSV.zip"

@asset
def gdelt_raw_data() -> pd.DataFrame:
    url = get_gdelt_url()
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200: return pd.DataFrame()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, sep='\t', header=None, usecols=[0, 1, 6, 7, 29, 30, 31, 34, 60])
        df.columns = ["EVENT_ID", "DATE", "MAIN_ACTOR", "ACTOR_COUNTRY_CODE", 
                      "EVENT_CATEGORY_CODE", "IMPACT_SCORE", "ARTICLE_COUNT", 
                      "SENTIMENT_SCORE", "NEWS_LINK"]
        return df
    except Exception as e:
        print(f"Extraction Error: {e}")
        return pd.DataFrame()

@asset
def gdelt_motherduck_table(gdelt_raw_data: pd.DataFrame) -> Output:
    df = gdelt_raw_data
    if df.empty: return Output(None, metadata={"status": "Skipped"})

    token = os.getenv("MOTHERDUCK_TOKEN")
    con = duckdb.connect(f'md:gdelt_db?motherduck_token={token}')
    
    # Write to MotherDuck
    con.execute(f"INSERT INTO {TARGET_TABLE} SELECT * FROM df")
    
    return Output(f"Uploaded {len(df)} rows", metadata={"rows": len(df)})

gdelt_job = define_asset_job(name="gdelt_ingestion_job", selection="*")
gdelt_schedule = ScheduleDefinition(job=gdelt_job, cron_schedule="*/30 * * * *", execution_timezone="UTC")

defs = Definitions(assets=[gdelt_raw_data, gdelt_motherduck_table], jobs=[gdelt_job], schedules=[gdelt_schedule])