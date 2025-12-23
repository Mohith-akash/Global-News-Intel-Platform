import requests
import datetime
import zipfile
import io
import re
import pandas as pd
import duckdb
import os
import time
from urllib.parse import urlparse, unquote
from dagster import asset, Output, Definitions, ScheduleDefinition, define_asset_job
from dotenv import load_dotenv
import logging

load_dotenv()
TARGET_TABLE = "events_dagster"
MAX_RETRIES = 3
RETRY_DELAY = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_headline_from_url(url):
    """Extract clean headline from URL at ingestion time."""
    if not url or not isinstance(url, str):
        return None
    
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        segments = [s for s in path.split('/') if s and len(s) > 8]
        if not segments:
            return None
        
        for seg in reversed(segments):
            headline = clean_url_segment(seg)
            if headline and len(headline) > 20:
                return headline
        return None
    except Exception:
        return None


def clean_url_segment(text):
    """Clean a URL path segment into a readable headline."""
    if not text:
        return None
    
    text = str(text).strip()
    
    # Reject common garbage patterns early
    reject_patterns = [
        r'^[a-f0-9]{8}[-][a-f0-9]{4}',
        r'^[a-f0-9\-]{20,}$',
        r'^(article|post|item|id)[-_]?[a-f0-9]{6,}',
        r'^\d{10,}$',
        r'^\d+$',
        r'^[A-Z]{2,5}\s*\d{5,}',
    ]
    
    for pattern in reject_patterns:
        if re.match(pattern, text.lower()):
            return None
    
    # Remove file extensions
    text = re.sub(r'\.(html?|php|aspx?|jsp|shtml)$', '', text, flags=re.I)
    
    # Remove date patterns at start
    text = re.sub(r'^\d{8}[-_]?', '', text)
    text = re.sub(r'^\d{4}[-/]\d{2}[-/]\d{2}[-_]?', '', text)
    text = re.sub(r'^\d{4}[-_]', '', text)
    
    # Remove hex codes
    text = re.sub(r'^[a-f0-9]{6,8}[-_]', '', text)
    
    # Convert hyphens and underscores to spaces
    text = re.sub(r'[-_]+', ' ', text)
    
    # Remove garbage patterns anywhere
    text = re.sub(r'\s+\d{5,}$', '', text)
    text = re.sub(r'\s+[a-f0-9]{8,}$', '', text, flags=re.I)
    text = re.sub(r'\s+[a-z]{1,3}\d[a-z\d]{3,}', ' ', text, flags=re.I)
    
    # Remove trailing junk
    text = re.sub(r'\s+\d{1,8}$', '', text)
    text = re.sub(r'\s+[A-Za-z]\d[A-Za-z0-9]{1,5}$', '', text)
    text = re.sub(r'\s+[A-Z]{1,3}\d+$', '', text)
    text = re.sub(r'[\s,]+\d{1,6}$', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # NOW remove leading/trailing punctuation (AFTER all replacements)
    text = re.sub(r'^[.,;:\'\"!?\-_\s]+', '', text)
    text = re.sub(r'[.,;:\'\"!?\-_\s]+$', '', text)
    
    # Validate length and word count
    if len(text) < 15 or ' ' not in text:
        return None
    
    words = text.split()
    if len(words) < 3:
        return None
    
    # Reject if it's just a country/city name (all caps, short)
    if text.isupper() and len(words) <= 3:
        return None
    
    # Check for excessive numbers or hex characters
    text_no_spaces = text.replace(' ', '')
    if text_no_spaces:
        if sum(c.isdigit() for c in text_no_spaces) / len(text_no_spaces) > 0.15:
            return None
        if sum(c in '0123456789abcdefABCDEF' for c in text_no_spaces) / len(text_no_spaces) > 0.3:
            return None
    
    # Reject if last word looks like a code
    last_word = words[-1]
    if re.match(r'^[A-Za-z]{0,2}\d+[A-Za-z]*$', last_word) and len(last_word) < 8:
        words = words[:-1]
        text = ' '.join(words)
        if len(words) < 3:
            return None
    
    # Remove trailing junk words
    trailing_junk = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                     'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be', 'if',
                     'with', 'from', 'into', 'that', 'this', 'than', 'when', 'where',
                     'n', 'b', 'na', 'th', 'wh', 's', 't'}
    
    words = text.split()
    while words and (words[-1].lower() in trailing_junk or len(words[-1]) <= 1):
        words.pop()
        if len(words) < 3:
            return None
    
    if len(words) < 3:
        return None
    
    text = ' '.join(words)
    
    # Truncate to 100 chars but don't cut mid-word
    if len(text) > 100:
        text = text[:100].rsplit(' ', 1)[0]
    
    # Final validation
    if len(text) < 15:
        return None
    
    # Apply proper title case
    text = text.lower()
    words = text.split()
    small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                   'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be'}
    result = []
    for i, word in enumerate(words):
        if i == 0:
            result.append(word.capitalize())
        elif word in small_words:
            result.append(word)
        else:
            result.append(word.capitalize())
    
    return ' '.join(result)


def get_gdelt_url():
    now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=20)
    rounded_minute = (now.minute // 15) * 15
    rounded_time = now.replace(minute=rounded_minute, second=0, microsecond=0)
    timestamp = rounded_time.strftime("%Y%m%d%H%M00")
    return f"http://data.gdeltproject.org/gdeltv2/{timestamp}.export.CSV.zip"


def download_with_retry(url, max_retries=MAX_RETRIES, timeout=30):
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}: {url}")
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                logger.info(f"Download successful ({len(response.content)} bytes)")
                return response
            elif response.status_code == 404:
                logger.warning("File not found (404)")
                return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY * (2 ** attempt))
    
    logger.error(f"Failed after {max_retries} attempts")
    return None


def validate_dataframe(df):
    if df.empty:
        return False
    if 'EVENT_ID' not in df.columns or 'DATE' not in df.columns:
        return False
    df.dropna(subset=['EVENT_ID'], inplace=True)
    # DON'T deduplicate here - we'll do smart deduplication with best headlines
    logger.info(f"Validated: {len(df)} rows (before headline selection)")
    return True


def select_best_headline_per_event(df):
    """Group by EVENT_ID, try all URLs, keep row with best headline."""
    if df.empty:
        return df
    
    logger.info(f"Selecting best headlines from {len(df):,} rows...")
    
    # Group by EVENT_ID
    grouped = df.groupby('EVENT_ID')
    
    best_rows = []
    for event_id, group in grouped:
        best_headline = None
        best_row = None
        best_score = 0
        url_count = len(group)  # Our own article count
        
        for idx, row in group.iterrows():
            url = row.get('NEWS_LINK', '')
            if not url:
                continue
            
            # Try to extract headline
            headline = extract_headline_from_url(url)
            
            if headline:
                # Score by length and word count
                score = len(headline) + len(headline.split()) * 5
                if score > best_score:
                    best_score = score
                    best_headline = headline
                    best_row = row.copy()
        
        # If we found a good headline, use that row
        if best_row is not None and best_headline:
            best_row['HEADLINE'] = best_headline
            best_row['ARTICLE_COUNT'] = url_count  # Our own count
            best_rows.append(best_row)
        elif len(group) > 0:
            # No good headline found, keep first row but with our count
            first_row = group.iloc[0].copy()
            first_row['ARTICLE_COUNT'] = url_count
            first_row['HEADLINE'] = None
            best_rows.append(first_row)
    
    if best_rows:
        result = pd.DataFrame(best_rows)
        headlines_found = result['HEADLINE'].notna().sum()
        logger.info(f"Selected {len(result):,} events with {headlines_found:,} headlines")
        return result
    
    return pd.DataFrame()


@asset
def gdelt_raw_data() -> pd.DataFrame:
    """Extract raw GDELT data with best headline selection per event."""
    logger.info("Starting GDELT extraction")
    url = get_gdelt_url()
    
    response = download_with_retry(url)
    if not response:
        return pd.DataFrame()
    
    try:
        z = zipfile.ZipFile(io.BytesIO(response.content))
        csv_name = z.namelist()[0]
        
        with z.open(csv_name) as f:
            df = pd.read_csv(f, sep='\t', header=None, usecols=[0, 1, 6, 7, 29, 30, 31, 34, 60], low_memory=False)
        
        df.columns = ["EVENT_ID", "DATE", "MAIN_ACTOR", "ACTOR_COUNTRY_CODE", "EVENT_CATEGORY_CODE", 
                      "IMPACT_SCORE", "ARTICLE_COUNT", "SENTIMENT_SCORE", "NEWS_LINK"]
        
        df['EVENT_ID'] = df['EVENT_ID'].astype(str)
        df['DATE'] = df['DATE'].astype(str)
        
        for col in ['IMPACT_SCORE', 'ARTICLE_COUNT', 'SENTIMENT_SCORE']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if not validate_dataframe(df):
            return pd.DataFrame()
        
        # Select best headline from all URLs per event
        df = select_best_headline_per_event(df)
        
        return df
    except Exception as e:
        logger.error(f"Error: {e}")
        return pd.DataFrame()


@asset
def gdelt_motherduck_table(gdelt_raw_data: pd.DataFrame) -> Output:
    """Load data into MotherDuck with deduplication."""
    if gdelt_raw_data.empty:
        return Output(None, metadata={"status": "Skipped", "rows": 0})

    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        return Output(None, metadata={"status": "Error", "message": "Missing token"})
    
    try:
        with duckdb.connect(f'md:gdelt_db?motherduck_token={token}') as con:
            try:
                existing = con.execute(f"SELECT EVENT_ID FROM {TARGET_TABLE} WHERE DATE >= '{gdelt_raw_data['DATE'].min()}'").df()
                if not existing.empty:
                    gdelt_raw_data = gdelt_raw_data[~gdelt_raw_data['EVENT_ID'].isin(existing['EVENT_ID'])]
                
                if gdelt_raw_data.empty:
                    return Output("No new data", metadata={"status": "Skipped", "rows": 0})
                
                con.execute(f"""
                    INSERT INTO {TARGET_TABLE} 
                    (EVENT_ID, DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, EVENT_CATEGORY_CODE, 
                     IMPACT_SCORE, ARTICLE_COUNT, SENTIMENT_SCORE, NEWS_LINK, HEADLINE)
                    SELECT EVENT_ID, DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, EVENT_CATEGORY_CODE, 
                           IMPACT_SCORE, ARTICLE_COUNT, SENTIMENT_SCORE, NEWS_LINK, HEADLINE
                    FROM gdelt_raw_data
                """)
            except Exception as e:
                if "does not exist" in str(e).lower():
                    con.execute(f"CREATE TABLE {TARGET_TABLE} AS SELECT * FROM gdelt_raw_data")
                else:
                    raise e
            
            total = con.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE}").fetchone()[0]
            return Output(f"Inserted {len(gdelt_raw_data):,}", metadata={"rows": len(gdelt_raw_data), "total": total})
    except Exception as e:
        return Output(None, metadata={"status": "Error", "message": str(e)})


gdelt_job = define_asset_job(name="gdelt_ingestion_job", selection="*")
gdelt_schedule = ScheduleDefinition(job=gdelt_job, cron_schedule="*/30 * * * *", execution_timezone="UTC")
defs = Definitions(assets=[gdelt_raw_data, gdelt_motherduck_table], jobs=[gdelt_job], schedules=[gdelt_schedule])