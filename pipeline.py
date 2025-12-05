import requests
import datetime
import zipfile
import io
import pandas as pd
import duckdb
import os
import time
from dagster import asset, Output, Definitions, ScheduleDefinition, define_asset_job
from dotenv import load_dotenv
import logging

# --- CONFIGURATION ---
load_dotenv()
TARGET_TABLE = "events_dagster"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def download_with_retry(url, max_retries=MAX_RETRIES, timeout=30):
    """Download with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}: {url}")
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                logger.info(f"✅ Download successful ({len(response.content)} bytes)")
                return response
            elif response.status_code == 404:
                logger.warning(f"⚠️ File not found (404). File may not be ready yet.")
                return None
            else:
                logger.warning(f"⚠️ Unexpected status code: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"⏱️ Timeout on attempt {attempt + 1}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"🌐 Network error on attempt {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
            logger.info(f"⏳ Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    logger.error(f"❌ Failed to download after {max_retries} attempts")
    return None

def validate_dataframe(df):
    """Validate data quality before insertion"""
    if df.empty:
        logger.warning("⚠️ DataFrame is empty")
        return False
    
    # Check for required columns
    required_cols = ['EVENT_ID', 'DATE']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"❌ Missing required columns: {missing_cols}")
        return False
    
    # Check for null EVENT_IDs
    null_count = df['EVENT_ID'].isna().sum()
    if null_count > 0:
        logger.warning(f"⚠️ Found {null_count} rows with null EVENT_ID (will be filtered)")
        df.dropna(subset=['EVENT_ID'], inplace=True)
    
    # Check for duplicates within this batch
    dup_count = df['EVENT_ID'].duplicated().sum()
    if dup_count > 0:
        logger.warning(f"⚠️ Found {dup_count} duplicate EVENT_IDs in batch (will be removed)")
        df.drop_duplicates(subset=['EVENT_ID'], keep='first', inplace=True)
    
    logger.info(f"✅ Data validation passed: {len(df)} valid rows")
    return True

# --- ASSETS ---

@asset
def gdelt_raw_data() -> pd.DataFrame:
    """
    Extract raw GDELT data - STREAMLINED VERSION
    Only essential columns, no lat/long, no Actor2
    """
    logger.info("🚀 Starting GDELT data extraction")
    url = get_gdelt_url()
    logger.info(f"🔗 Target URL: {url}")
    
    # Download with retry logic
    response = download_with_retry(url)
    if not response:
        logger.warning("⚠️ No data retrieved. Returning empty DataFrame.")
        return pd.DataFrame()
    
    try:
        # Extract ZIP
        z = zipfile.ZipFile(io.BytesIO(response.content))
        csv_name = z.namelist()[0]
        logger.info(f"📂 Processing: {csv_name}")
        
        # STREAMLINED: Only 10 essential columns
        # Removed: lat/long, Actor2, Actor2Country
        with z.open(csv_name) as f:
            df = pd.read_csv(
                f, 
                sep='\t', 
                header=None,
                usecols=[0, 1, 6, 7, 29, 30, 31, 34, 53, 60],
                low_memory=False
            )
        
        # Rename columns
        df.columns = [
            "EVENT_ID",              # 0: Unique ID
            "DATE",                  # 1: YYYYMMDD (keep for SQL queries)
            "MAIN_ACTOR",            # 6: Primary actor
            "ACTOR_COUNTRY_CODE",    # 7: Country code (keep for SQL, display converts to full name)
            "EVENT_CATEGORY_CODE",   # 29: CAMEO code
            "IMPACT_SCORE",          # 30: Goldstein scale
            "ARTICLE_COUNT",         # 31: Mentions
            "SENTIMENT_SCORE",       # 34: Tone
            "LOCATION_NAME",         # 53: Full location (e.g., "Tokyo, Japan")
            "NEWS_LINK"              # 60: Source URL
        ]
        
        # Type conversions
        df['EVENT_ID'] = df['EVENT_ID'].astype(str)
        df['DATE'] = df['DATE'].astype(str)
        
        # Convert numeric columns
        numeric_cols = ['IMPACT_SCORE', 'ARTICLE_COUNT', 'SENTIMENT_SCORE']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate data quality
        if not validate_dataframe(df):
            logger.error("❌ Data validation failed")
            return pd.DataFrame()
        
        logger.info(f"✅ Successfully extracted {len(df):,} rows with {len(df.columns)} columns")
        return df

    except zipfile.BadZipFile:
        logger.error("❌ Corrupted ZIP file")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"❌ CSV parsing error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ Unexpected error during extraction: {e}")
        return pd.DataFrame()

@asset
def gdelt_motherduck_table(gdelt_raw_data: pd.DataFrame) -> Output:
    """
    Load data into MotherDuck with deduplication.
    Returns metadata about the operation.
    """
    if gdelt_raw_data.empty:
        logger.warning("⚠️ No data to load. Skipping.")
        return Output(None, metadata={"status": "Skipped", "rows": 0})

    start_time = time.time()
    logger.info(f"🦆 Connecting to MotherDuck...")
    token = os.getenv("MOTHERDUCK_TOKEN")
    
    if not token:
        logger.error("❌ MOTHERDUCK_TOKEN not found in environment")
        return Output(None, metadata={"status": "Error", "message": "Missing token"})
    
    # Use context manager for proper connection handling
    try:
        with duckdb.connect(f'md:gdelt_db?motherduck_token={token}') as con:
            
            # Check if table exists and get existing EVENT_IDs for deduplication
            try:
                existing_ids = con.execute(
                    f"SELECT EVENT_ID FROM {TARGET_TABLE} WHERE DATE >= '{gdelt_raw_data['DATE'].min()}'"
                ).df()
                
                if not existing_ids.empty:
                    # Remove duplicates
                    initial_count = len(gdelt_raw_data)
                    gdelt_raw_data = gdelt_raw_data[
                        ~gdelt_raw_data['EVENT_ID'].isin(existing_ids['EVENT_ID'])
                    ]
                    removed_count = initial_count - len(gdelt_raw_data)
                    if removed_count > 0:
                        logger.info(f"🗑️ Removed {removed_count} duplicate events")
                
                if gdelt_raw_data.empty:
                    logger.info("✅ All events already exist. No new data to insert.")
                    return Output(
                        "No new data", 
                        metadata={"status": "Skipped", "reason": "All duplicates", "rows": 0}
                    )
                
                # Insert new data
                con.execute(f"INSERT INTO {TARGET_TABLE} SELECT * FROM gdelt_raw_data")
                logger.info(f"✅ Inserted {len(gdelt_raw_data):,} new rows")
                
            except Exception as e:
                # Table doesn't exist - create it
                if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                    logger.info(f"📋 Creating new table: {TARGET_TABLE}")
                    con.execute(
                        f"CREATE TABLE {TARGET_TABLE} AS SELECT * FROM gdelt_raw_data"
                    )
                    logger.info(f"✅ Created table with {len(gdelt_raw_data):,} rows")
                else:
                    raise e
            
            # Get table statistics
            stats = con.execute(f"SELECT COUNT(*) as total FROM {TARGET_TABLE}").df()
            total_rows = stats.iloc[0]['total']
            
            elapsed = time.time() - start_time
            logger.info(f"⏱️ Processing completed in {elapsed:.2f} seconds")
            logger.info(f"📊 Total rows in table: {total_rows:,}")
            
            return Output(
                f"Uploaded {len(gdelt_raw_data):,} rows",
                metadata={
                    "status": "Success",
                    "rows_inserted": len(gdelt_raw_data),
                    "total_rows": int(total_rows),
                    "processing_time": round(elapsed, 2),
                    "columns": len(gdelt_raw_data.columns)
                }
            )
            
    except Exception as e:
        logger.error(f"❌ MotherDuck operation failed: {e}")
        return Output(
            None, 
            metadata={"status": "Error", "message": str(e)}
        )

# --- JOB DEFINITIONS ---

# Define the job that runs the assets
gdelt_job = define_asset_job(
    name="gdelt_ingestion_job", 
    selection="*",
    description="Ingest GDELT data into MotherDuck (streamlined version)"
)

# Define the schedule (Every 30 Minutes)
gdelt_schedule = ScheduleDefinition(
    job=gdelt_job,
    cron_schedule="*/30 * * * *",
    execution_timezone="UTC",
    description="Run GDELT ingestion every 30 minutes"
)

# Bundle everything into Definitions for Dagster to read
defs = Definitions(
    assets=[gdelt_raw_data, gdelt_motherduck_table],
    jobs=[gdelt_job],
    schedules=[gdelt_schedule]
)
