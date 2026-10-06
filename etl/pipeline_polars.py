"""
GDELT ingestion pipeline: Polars-based extraction, Great Expectations
validation, loading into MotherDuck via Dagster assets.

Runs every 15 minutes via GitHub Actions; embeddings run separately
(see embedding_job.py, every 12 hours).
"""

import requests
import datetime
import zipfile
import io
import polars as pl
import duckdb
import os
import time
from dagster import asset, Output, Definitions, ScheduleDefinition, define_asset_job, AssetExecutionContext
from dotenv import load_dotenv
import logging

from src.headline_utils import extract_headline_from_url, clean_headline

load_dotenv()

# Configuration
TARGET_TABLE = "events_dagster"
MAX_RETRIES = 3
RETRY_DELAY = 5

# Voyage AI Configuration
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
try:
    from src.config import VOYAGE_MODEL, EMBEDDING_DIMENSIONS
except ImportError:
    VOYAGE_MODEL = "voyage-3.5-lite"
    EMBEDDING_DIMENSIONS = 1024
MIN_ARTICLE_COUNT_FOR_EMBEDDING = 3

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GDELT Column Schema (for validation)
GDELT_SCHEMA = {
    "EVENT_ID": pl.Utf8,
    "DATE": pl.Utf8,
    "MAIN_ACTOR": pl.Utf8,
    "ACTOR_COUNTRY_CODE": pl.Utf8,
    "EVENT_CATEGORY_CODE": pl.Float64,
    "IMPACT_SCORE": pl.Float64,
    "ARTICLE_COUNT": pl.Float64,
    "SENTIMENT_SCORE": pl.Float64,
    "NEWS_LINK": pl.Utf8,
}


# =============================================================================
# DATA QUALITY VALIDATION (Great Expectations / GX Core)
# =============================================================================

def build_gdelt_suite(batches: int = 1):
    """
    Expectation suite for one ingestion run of GDELT events.
    Row count bounds are per 15-min batch, scaled by how many batches the
    run pulled (96 by default) - a fixed 100-50,000 range failed every run
    once the 24h lookback went in.
    """
    import great_expectations as gx
    import great_expectations.expectations as gxe

    suite = gx.ExpectationSuite(name="gdelt_events")

    # Required columns must exist
    for column in ("EVENT_ID", "DATE", "MAIN_ACTOR"):
        suite.add_expectation(gxe.ExpectColumnToExist(column=column))

    # Primary key and date should never be null
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="EVENT_ID"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="DATE"))

    # Reasonable row count: 100 to 50,000 per batch
    suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(
        min_value=100 * batches, max_value=50000 * batches))

    # Goldstein scale is -10 to 10, AvgTone typically -100 to 100.
    # Nulls are skipped, 1% outliers allowed.
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="IMPACT_SCORE", min_value=-10.0, max_value=10.0, mostly=0.99))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="SENTIMENT_SCORE", min_value=-100.0, max_value=100.0, mostly=0.99))

    return suite


def validate_gdelt_data(df: pl.DataFrame, batches: int = 1) -> dict:
    """
    Run the GX suite against an ingestion run's DataFrame.
    Returns success, GX statistics and the failed expectations only.
    """
    # imported here, not at module top: dagster runs each step in its own
    # process and only this one needs GX (slow import)
    os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
    # basicConfig above puts root at INFO, GX floods it with registry noise on import
    logging.getLogger("great_expectations").setLevel(logging.WARNING)
    import great_expectations as gx
    from great_expectations.data_context.types.base import ProgressBarsConfig

    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    batch_definition = (
        context.data_sources.add_pandas("gdelt")
        .add_dataframe_asset(name="events")
        .add_batch_definition_whole_dataframe("ingest_run")
    )
    suite = context.suites.add(build_gdelt_suite(batches))
    validation = context.validation_definitions.add(
        gx.ValidationDefinition(name="gdelt_events_check", data=batch_definition, suite=suite)
    )

    result = validation.run(batch_parameters={"dataframe": df.to_pandas()})

    failed = []
    for r in result.results:
        if r.success:
            continue
        failed.append({
            "expectation": r.expectation_config.type,
            "column": r.expectation_config.kwargs.get("column"),
            "observed": r.result.get("observed_value", r.result.get("unexpected_percent")),
        })

    return {
        "success": result.success,
        "statistics": result.statistics,
        "results": failed,
    }


# =============================================================================
# HEADLINE EXTRACTION
# =============================================================================
# Uses src.headline_utils as single source of truth (imported above).


# =============================================================================
# POLARS-BASED DATA PROCESSING
# =============================================================================

def _latest_batch_times(n: int = 4) -> list:
    """Timestamps of the last n GDELT 15-minute batches, newest first.
    GDELT publishes roughly 15 minutes behind the wall clock, hence the
    20-minute offset before rounding down."""
    now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=20)
    rounded_minute = (now.minute // 15) * 15
    rounded_time = now.replace(minute=rounded_minute, second=0, microsecond=0)
    return [rounded_time - datetime.timedelta(minutes=15 * i) for i in range(n)]


def _batch_count() -> int:
    """How many 15-minute batches to fetch per run.

    Default 96 (24 hours). This used to be 20 (five hours), sized for a cron
    that skipped the odd hour. Github's scheduler has since degraded much
    further - observed gaps of 8, 10 and 11 hours between runs - and every
    hour past the lookback is a permanent hole: 2026-08-27 landed 59,510
    events against ~108,000 on the days either side, roughly 45% lost.
    A 24-hour window self-heals any gap short of a full day.

    Overlap is nearly free: the writers dedup on EVENT_ID / GKG_ID, so
    re-fetched batches insert nothing. The extra cost is downloading ~96
    small files per run inside Actions (free), not MotherDuck compute,
    which matters on the Lite plan's 10 CU-hours/month.
    Override with GDELT_BATCHES (workflow_dispatch passes it) for backfills.
    """
    try:
        return max(1, min(288, int(os.getenv("GDELT_BATCHES", "96"))))
    except ValueError:
        return 96


def get_gdelt_urls(n: int = None) -> list:
    """URLs for the last n GDELT export batches (default: _batch_count())."""
    if n is None:
        n = _batch_count()
    return [
        f"http://data.gdeltproject.org/gdeltv2/{t.strftime('%Y%m%d%H%M00')}.export.CSV.zip"
        for t in _latest_batch_times(n)
    ]


def download_with_retry(url: str, max_retries: int = MAX_RETRIES, timeout: int = 30):
    """Download with exponential backoff retry."""
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


def process_gdelt_batch_polars(content: bytes) -> pl.DataFrame:
    """
    Process GDELT CSV content using Polars.
    10x faster than Pandas for this operation.
    """
    z = zipfile.ZipFile(io.BytesIO(content))
    csv_name = z.namelist()[0]
    
    with z.open(csv_name) as f:
        # Read with Polars - significantly faster than Pandas
        df = pl.read_csv(
            f,
            separator='\t',
            has_header=False,
            columns=[0, 1, 6, 7, 29, 30, 31, 34, 60],  # Select only needed columns
            new_columns=["EVENT_ID", "DATE", "MAIN_ACTOR", "ACTOR_COUNTRY_CODE", 
                        "EVENT_CATEGORY_CODE", "IMPACT_SCORE", "ARTICLE_COUNT", 
                        "SENTIMENT_SCORE", "NEWS_LINK"],
            infer_schema_length=10000,
            ignore_errors=True,
        )
    
    # Cast columns to correct types
    df = df.with_columns([
        pl.col("EVENT_ID").cast(pl.Utf8),
        pl.col("DATE").cast(pl.Utf8),
        pl.col("IMPACT_SCORE").cast(pl.Float64),
        pl.col("ARTICLE_COUNT").cast(pl.Float64),
        pl.col("SENTIMENT_SCORE").cast(pl.Float64),
    ])
    
    return df


def select_best_headline_per_event_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Group by EVENT_ID, extract best headline from URLs.
    Polars implementation with lazy evaluation.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(None).alias("HEADLINE"))
    
    logger.info(f"Selecting best headlines from {len(df):,} rows...")
    
    # Add headline column by extracting from URLs
    # We need to do this row by row since headline extraction is complex
    headlines = []
    for url in df["NEWS_LINK"].to_list():
        headlines.append(extract_headline_from_url(url))
    
    df = df.with_columns(pl.Series("HEADLINE", headlines))

    # Count articles per event BEFORE deduplication (one raw row = one source URL)
    article_counts = df.group_by("EVENT_ID").len().rename({"len": "OUR_ARTICLE_COUNT"})

    # Group by EVENT_ID and keep row with best headline
    # Use Polars' powerful groupby + agg
    df = df.with_columns(
        pl.col("HEADLINE").is_not_null().alias("has_headline"),
        pl.col("HEADLINE").str.len_chars().fill_null(0).alias("headline_len")
    )

    # Sort to prioritize rows with headlines, then by headline length
    df = df.sort(["EVENT_ID", "has_headline", "headline_len"], descending=[False, True, True])

    # Keep first (best) row per EVENT_ID
    df = df.group_by("EVENT_ID", maintain_order=True).first()

    df = df.join(article_counts, on="EVENT_ID", how="left")
    
    # Use our article count if higher than GDELT's
    df = df.with_columns(
        pl.when(pl.col("OUR_ARTICLE_COUNT") > pl.col("ARTICLE_COUNT"))
        .then(pl.col("OUR_ARTICLE_COUNT"))
        .otherwise(pl.col("ARTICLE_COUNT"))
        .alias("ARTICLE_COUNT")
    )
    
    # Drop temp columns
    df = df.drop(["has_headline", "headline_len", "OUR_ARTICLE_COUNT"])
    
    headlines_found = df.filter(pl.col("HEADLINE").is_not_null()).height
    logger.info(f"Selected {len(df):,} events with {headlines_found:,} headlines")
    
    return df


# =============================================================================
# DAGSTER ASSETS - INGESTION JOB (Every 15 minutes)
# =============================================================================

@asset(description="Extract and transform GDELT data using Polars (no embeddings)")
def gdelt_raw_data_polars(context: AssetExecutionContext) -> pl.DataFrame:
    """
    Extract raw GDELT data with Polars.
    Runs hourly and re-pulls the last _batch_count() 15-minute batches
    (20 = five hours) so a skipped cron run does not leave a hole.
    Embeddings are computed by a separate job every 12 hours.
    """
    logger.info("🚀 Starting GDELT extraction (Polars-powered)")

    parts = []
    for url in get_gdelt_urls():
        response = download_with_retry(url)
        if not response:
            context.log.warning(f"Batch download failed, skipping: {url}")
            continue
        batch = process_gdelt_batch_polars(response.content)
        if not batch.is_empty():
            parts.append(batch)

    if not parts:
        context.log.warning("All batch downloads failed, returning empty DataFrame")
        return pl.DataFrame()

    try:
        # Process with Polars (10x faster than Pandas)
        df = pl.concat(parts, how="vertical_relaxed")
        logger.info(f"📊 Loaded {len(df):,} rows from {len(parts)} batches with Polars")
        
        # Data Quality Validation
        logger.info("🔍 Running data quality validation...")
        validation_result = validate_gdelt_data(df, batches=len(parts))
        
        if not validation_result["success"]:
            context.log.warning(f"⚠️ Data quality issues: {validation_result['results']}")
            # Log but don't fail - we'll filter bad data
        else:
            context.log.info(f"✅ Data quality passed: {validation_result['statistics']}")
        
        # Filter out rows with null EVENT_ID
        df = df.filter(pl.col("EVENT_ID").is_not_null())
        
        # Select best headline per event
        df = select_best_headline_per_event_polars(df)
        
        # Add EMBEDDING column as null (will be populated by separate job)
        df = df.with_columns(pl.lit(None).cast(pl.List(pl.Float64)).alias("EMBEDDING"))
        
        logger.info(f"✅ Processed {len(df):,} unique events")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error processing GDELT data: {e}")
        context.log.error(f"Processing error: {e}")
        return pl.DataFrame()


@asset(description="Load data into MotherDuck with deduplication")
def gdelt_motherduck_table_polars(context: AssetExecutionContext, gdelt_raw_data_polars: pl.DataFrame) -> Output:
    """Load Polars DataFrame into MotherDuck with deduplication."""
    if gdelt_raw_data_polars.is_empty():
        return Output(None, metadata={"status": "Skipped", "rows": 0})

    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        return Output(None, metadata={"status": "Error", "message": "Missing token"})
    
    try:
        # Convert Polars to Pandas for DuckDB insertion
        # (DuckDB has better Pandas support currently)
        pdf = gdelt_raw_data_polars.to_pandas()
        
        with duckdb.connect(f'md:gdelt_db?motherduck_token={token}') as con:
            try:
                # Check if EMBEDDING column exists
                try:
                    con.execute(f"SELECT EMBEDDING FROM {TARGET_TABLE} LIMIT 1")
                except Exception:
                    logger.info("Adding EMBEDDING column to table...")
                    con.execute(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN EMBEDDING DOUBLE[]")
                
                # Deduplicate against existing data
                existing = con.execute(f"""
                    SELECT EVENT_ID FROM {TARGET_TABLE} 
                    WHERE DATE >= '{pdf['DATE'].min()}'
                """).df()
                
                if not existing.empty:
                    pdf = pdf[~pdf['EVENT_ID'].isin(existing['EVENT_ID'])]
                
                if pdf.empty:
                    return Output("No new data", metadata={"status": "Skipped", "rows": 0})
                
                # Register and insert
                con.register('new_data', pdf)
                con.execute(f"""
                    INSERT INTO {TARGET_TABLE} 
                    (EVENT_ID, DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, EVENT_CATEGORY_CODE, 
                     IMPACT_SCORE, ARTICLE_COUNT, SENTIMENT_SCORE, NEWS_LINK, HEADLINE, EMBEDDING)
                    SELECT EVENT_ID, DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, EVENT_CATEGORY_CODE, 
                           IMPACT_SCORE, ARTICLE_COUNT, SENTIMENT_SCORE, NEWS_LINK, HEADLINE, EMBEDDING
                    FROM new_data
                """)
                
                total = con.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE}").fetchone()[0]
                embedded = con.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE} WHERE EMBEDDING IS NOT NULL").fetchone()[0]
                
                context.log.info(f"✅ Inserted {len(pdf):,} rows. Total: {total:,}, Embedded: {embedded:,}")
                return Output(
                    f"Inserted {len(pdf):,}", 
                    metadata={"rows": len(pdf), "total": total, "embedded": embedded}
                )
                
            except Exception as e:
                if "does not exist" in str(e).lower():
                    con.register('new_data', pdf)
                    con.execute(f"CREATE TABLE {TARGET_TABLE} AS SELECT * FROM new_data")
                    return Output(f"Created table with {len(pdf):,} rows", metadata={"rows": len(pdf)})
                else:
                    raise e
                
    except Exception as e:
        context.log.error(f"❌ MotherDuck error: {e}")
        return Output(None, metadata={"status": "Error", "message": str(e)})


# =============================================================================
# GKG (GLOBAL KNOWLEDGE GRAPH) - EMOTIONS & THEMES
# =============================================================================

GKG_TABLE = "gkg_emotions"
GKG_RETENTION_DAYS = 30  # dashboard uses 24h, dbt daily models up to 30d

# Key GCAM emotion codes we want to extract
# Format: c{dictionary_id}.{dimension_id}
GCAM_EMOTIONS = {
    "c9.1": "fear",
    "c9.2": "anger", 
    "c9.3": "sadness",
    "c9.4": "joy",
    "c9.5": "disgust",
    "c9.6": "surprise",
    "c9.7": "trust",
    "c9.8": "anticipation",
    "c18.1": "anxiety",
    "c18.2": "hostility",
    "c18.3": "depression",
}


def get_gdelt_gkg_urls(n: int = None) -> list:
    """URLs for the last n GDELT GKG batches - same coverage window as the
    event export files."""
    if n is None:
        n = _batch_count()
    return [
        f"http://data.gdeltproject.org/gdeltv2/{t.strftime('%Y%m%d%H%M00')}.gkg.csv.zip"
        for t in _latest_batch_times(n)
    ]


def parse_gcam_field(gcam_str: str) -> dict:
    """
    Parse GCAM field into emotion scores.
    GCAM format: dimension:value,dimension:value,...
    """
    emotions = {name: 0.0 for name in GCAM_EMOTIONS.values()}
    
    if not gcam_str or not isinstance(gcam_str, str):
        return emotions
    
    try:
        for pair in gcam_str.split(","):
            if ":" in pair:
                parts = pair.split(":")
                if len(parts) == 2:
                    code, value = parts
                    if code in GCAM_EMOTIONS:
                        try:
                            emotions[GCAM_EMOTIONS[code]] = float(value)
                        except ValueError:
                            pass
    except Exception:
        pass
    
    return emotions


def parse_themes_field(themes_str: str) -> list:
    """Extract top themes from GKG themes field."""
    if not themes_str or not isinstance(themes_str, str):
        return []
    
    try:
        # Themes are semicolon-separated, may have character offsets
        themes = []
        for item in themes_str.split(";"):
            # Remove character offset if present (format: THEME,offset)
            theme = item.split(",")[0].strip()
            if theme and len(theme) > 2:
                themes.append(theme)
        return themes[:10]  # Keep top 10 themes
    except Exception:
        return []


def process_gkg_batch_polars(content: bytes) -> pl.DataFrame:
    """
    Process GDELT GKG CSV content using Polars.
    Extracts emotions and themes from news articles.
    """
    z = zipfile.ZipFile(io.BytesIO(content))
    csv_name = z.namelist()[0]
    
    with z.open(csv_name) as f:
        # GKG columns we need:
        # 0: GKGRECORDID, 1: DATE, 3: SourceCommonName, 7: Themes
        # 11: Persons, 12: Organizations, 15: Tone, 17: GCAM
        try:
            df = pl.read_csv(
                f,
                separator='\t',
                has_header=False,
                columns=[0, 1, 3, 7, 11, 12, 15, 17],
                new_columns=["GKG_ID", "DATE", "SOURCE", "THEMES", 
                            "PERSONS", "ORGS", "TONE", "GCAM"],
                infer_schema_length=10000,
                ignore_errors=True,
                truncate_ragged_lines=True,
                # ~6% of gkg files carry stray non-utf8 bytes (in columns we
                # don't keep) and strict decoding dropped the whole file
                encoding="utf8-lossy",
            )
        except Exception as e:
            logger.error(f"Error reading GKG CSV: {e}")
            return pl.DataFrame()
    
    if df.is_empty():
        return df
    
    # Parse TONE field (format: tone,positive,negative,polarity,activity,self/group)
    def extract_tone(tone_str):
        if not tone_str:
            return (0.0, 0.0, 0.0)
        try:
            parts = str(tone_str).split(",")
            avg_tone = float(parts[0]) if len(parts) > 0 else 0.0
            positive = float(parts[1]) if len(parts) > 1 else 0.0
            negative = float(parts[2]) if len(parts) > 2 else 0.0
            return (avg_tone, positive, negative)
        except Exception:
            return (0.0, 0.0, 0.0)
    
    # Extract tone components
    tones = [extract_tone(t) for t in df["TONE"].to_list()]
    df = df.with_columns([
        pl.Series("AVG_TONE", [t[0] for t in tones]),
        pl.Series("POSITIVE_SCORE", [t[1] for t in tones]),
        pl.Series("NEGATIVE_SCORE", [t[2] for t in tones]),
    ])
    
    # Parse GCAM emotions
    gcam_data = [parse_gcam_field(g) for g in df["GCAM"].to_list()]
    for emotion_name in GCAM_EMOTIONS.values():
        df = df.with_columns(
            pl.Series(f"EMOTION_{emotion_name.upper()}", 
                     [d.get(emotion_name, 0.0) for d in gcam_data])
        )
    
    # Parse themes into list (store as comma-separated string for simplicity)
    themes_list = [",".join(parse_themes_field(t)) for t in df["THEMES"].to_list()]
    df = df.with_columns(pl.Series("TOP_THEMES", themes_list))
    
    # Clean up - drop raw fields, keep processed ones
    df = df.drop(["TONE", "GCAM", "THEMES"])
    
    # Filter out rows with null GKG_ID
    df = df.filter(pl.col("GKG_ID").is_not_null())
    
    logger.info(f"📊 Processed {len(df):,} GKG records")
    return df


@asset(description="Extract emotions and themes from GDELT GKG")
def gdelt_gkg_data(context: AssetExecutionContext) -> pl.DataFrame:
    """
    Extract GKG data with emotions and themes.
    Runs alongside event ingestion, hourly, covering the past hour of batches.
    """
    logger.info("🧠 Starting GDELT GKG extraction (emotions & themes)")

    parts = []
    for url in get_gdelt_gkg_urls():
        response = download_with_retry(url)
        if not response:
            context.log.warning(f"GKG batch download failed, skipping: {url}")
            continue
        try:
            batch = process_gkg_batch_polars(response.content)
        except Exception as e:
            context.log.warning(f"GKG batch processing failed, skipping: {e}")
            continue
        if not batch.is_empty():
            parts.append(batch)

    if not parts:
        context.log.warning("No GKG data processed")
        return pl.DataFrame()

    df = pl.concat(parts, how="vertical_relaxed")
    logger.info(f"✅ Extracted {len(df):,} GKG records with emotions from {len(parts)} batches")
    return df


@asset(description="Load GKG emotions data into MotherDuck")
def gdelt_gkg_motherduck(context: AssetExecutionContext, gdelt_gkg_data: pl.DataFrame) -> Output:
    """Load GKG emotions data into MotherDuck."""
    if gdelt_gkg_data.is_empty():
        return Output(None, metadata={"status": "Skipped", "rows": 0})
    
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        return Output(None, metadata={"status": "Error", "message": "Missing token"})
    
    try:
        pdf = gdelt_gkg_data.to_pandas()
        
        with duckdb.connect(f'md:gdelt_db?motherduck_token={token}') as con:
            try:
                # Check if table exists and deduplicate
                existing = con.execute(f"""
                    SELECT GKG_ID FROM {GKG_TABLE} 
                    WHERE DATE >= '{pdf['DATE'].min()}'
                """).df()
                
                if not existing.empty:
                    pdf = pdf[~pdf['GKG_ID'].isin(existing['GKG_ID'])]
                
                if pdf.empty:
                    return Output("No new GKG data", metadata={"status": "Skipped", "rows": 0})
                
                con.register('gkg_data', pdf)
                con.execute(f"INSERT INTO {GKG_TABLE} SELECT * FROM gkg_data")

                # Retention: the dashboard only reads the last 24h of GKG data
                # (dbt daily models use up to 30 days), so anything older is
                # dead weight — without this the table grows ~3M rows/month.
                # GKG DATE is a 14-digit numeric timestamp (YYYYMMDDHHMMSS).
                cutoff = int((datetime.datetime.now(datetime.timezone.utc)
                              - datetime.timedelta(days=GKG_RETENTION_DAYS)).strftime('%Y%m%d000000'))
                con.execute(f"DELETE FROM {GKG_TABLE} WHERE DATE < {cutoff}")

                total = con.execute(f"SELECT COUNT(*) FROM {GKG_TABLE}").fetchone()[0]
                context.log.info(f"✅ Inserted {len(pdf):,} GKG rows. Total after retention: {total:,}")
                return Output(f"Inserted {len(pdf):,}", metadata={"rows": len(pdf), "total": total})
                
            except Exception as e:
                if "does not exist" in str(e).lower():
                    con.register('gkg_data', pdf)
                    con.execute(f"CREATE TABLE {GKG_TABLE} AS SELECT * FROM gkg_data")
                    context.log.info(f"✅ Created {GKG_TABLE} with {len(pdf):,} rows")
                    return Output(f"Created table with {len(pdf):,} rows", metadata={"rows": len(pdf)})
                else:
                    raise e
                
    except Exception as e:
        context.log.error(f"❌ GKG MotherDuck error: {e}")
        return Output(None, metadata={"status": "Error", "message": str(e)})


# =============================================================================
# DAGSTER JOBS & SCHEDULES
# =============================================================================

# Main ingestion job - runs every 15 minutes (now includes GKG)
gdelt_ingestion_job = define_asset_job(
    name="gdelt_ingestion_job",
    selection=["gdelt_raw_data_polars", "gdelt_motherduck_table_polars", 
               "gdelt_gkg_data", "gdelt_gkg_motherduck"],
    description="Ingest GDELT events + GKG emotions every 15 minutes"
)

# Schedule: Every 15 minutes
gdelt_ingestion_schedule = ScheduleDefinition(
    job=gdelt_ingestion_job,
    cron_schedule="*/15 * * * *",  # Every 15 minutes
    execution_timezone="UTC",
    description="Run GDELT ingestion every 15 minutes"
)

# Definitions
defs = Definitions(
    assets=[gdelt_raw_data_polars, gdelt_motherduck_table_polars,
            gdelt_gkg_data, gdelt_gkg_motherduck],
    jobs=[gdelt_ingestion_job],
    schedules=[gdelt_ingestion_schedule]
)

