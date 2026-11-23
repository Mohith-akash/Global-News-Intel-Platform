"""
GDELT ingestion pipeline: Polars extraction, Great Expectations checks and
loading into MotherDuck, as Dagster assets.

GitHub Actions runs the job hourly; embeddings run separately
(embedding_job.py, every 12 hours).
"""

import requests
import datetime
import zipfile
import io
import polars as pl
import duckdb
import os
import time
from dagster import asset, Output, Definitions, define_asset_job, AssetExecutionContext
from dotenv import load_dotenv
import logging

from src.headline_utils import extract_headline_from_url

load_dotenv()

# Configuration
TARGET_TABLE = "events_dagster"
MAX_RETRIES = 3
RETRY_DELAY = 5

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    """One zipped GDELT export file -> the columns the app uses."""
    z = zipfile.ZipFile(io.BytesIO(content))
    csv_name = z.namelist()[0]

    with z.open(csv_name) as f:
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
    """One row per EVENT_ID, keeping the source URL that gives the best headline."""
    if df.is_empty():
        return df.with_columns(pl.lit(None).alias("HEADLINE"))

    logger.info(f"Selecting best headlines from {len(df):,} rows...")

    # row by row: headline extraction is regex-heavy Python
    df = df.with_columns(pl.Series("HEADLINE", [extract_headline_from_url(u) for u in df["NEWS_LINK"].to_list()]))

    # Count articles per event BEFORE deduplication (one raw row = one source URL)
    article_counts = df.group_by("EVENT_ID").len().rename({"len": "OUR_ARTICLE_COUNT"})

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
# DAGSTER ASSETS
# =============================================================================

@asset(description="Extract and transform GDELT events with Polars (no embeddings)")
def gdelt_raw_data_polars(context: AssetExecutionContext) -> pl.DataFrame:
    """
    Download the last _batch_count() 15-minute export files (24 hours by
    default, so a skipped hourly run leaves no hole) and keep one row per event.
    Embeddings are computed by a separate job every 12 hours.
    """
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
        context.log.warning("All batch downloads failed")
        return pl.DataFrame()

    df = pl.concat(parts, how="vertical_relaxed")
    logger.info(f"Loaded {len(df):,} rows from {len(parts)} batches")

    validation = validate_gdelt_data(df, batches=len(parts))
    if validation["success"]:
        context.log.info(f"Data quality passed: {validation['statistics']}")
    else:
        # reported, not fatal: the filters below drop the bad rows
        context.log.warning(f"Data quality issues: {validation['results']}")

    df = select_best_headline_per_event_polars(df.filter(pl.col("EVENT_ID").is_not_null()))
    # EMBEDDING is filled later by the embedding job
    df = df.with_columns(pl.lit(None).cast(pl.List(pl.Float64)).alias("EMBEDDING"))
    logger.info(f"Processed {len(df):,} unique events")
    return df


def _connect():
    """MotherDuck, or a local duckdb file when GDELT_DB is set (tests, local runs)."""
    if os.getenv("GDELT_DB"):
        return duckdb.connect(os.environ["GDELT_DB"])
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN is not set")
    return duckdb.connect(f"md:gdelt_db?motherduck_token={token}")


def _insert_new(con, table: str, df: pl.DataFrame, key: str) -> int:
    """Insert the rows of `df` whose `key` isn't in `table` yet; returns rows inserted.

    The duplicate check runs inside the warehouse, against rows from the
    batch's first DATE on, so no ids travel back to the runner. Columns are
    matched by name (the table has columns the batch doesn't, like
    HEADLINE_AI). The first run creates the table.
    """
    con.register("incoming", df.to_arrow())
    try:
        exists = con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = ? "
                             "AND table_catalog = current_database()", [table]).fetchone()
        if not exists:
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM incoming")
            return df.height
        return con.execute(f"""
            INSERT INTO {table} BY NAME
            SELECT * FROM incoming i
            WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE t.{key} = i.{key} AND t.DATE >= ?)
        """, [df["DATE"].min()]).fetchone()[0]
    finally:
        con.unregister("incoming")


@asset(description="Load new events into MotherDuck")
def gdelt_motherduck_table_polars(context: AssetExecutionContext, gdelt_raw_data_polars: pl.DataFrame) -> Output:
    """Insert events not stored yet. Any error fails the run (and files the alert issue)."""
    if gdelt_raw_data_polars.is_empty():
        return Output(None, metadata={"status": "Skipped", "rows": 0})
    with _connect() as con:
        inserted = _insert_new(con, TARGET_TABLE, gdelt_raw_data_polars, "EVENT_ID")
        total = con.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE}").fetchone()[0]
    context.log.info(f"Inserted {inserted:,} new events, {total:,} in total")
    return Output(f"Inserted {inserted:,}", metadata={"rows": inserted, "total": total})


# =============================================================================
# GKG (GLOBAL KNOWLEDGE GRAPH) - EMOTIONS & THEMES
# =============================================================================

GKG_TABLE = "gkg_emotions"
GKG_RETENTION_DAYS = 30  # dashboard uses 24h, dbt daily models up to 30d

# GCAM dictionary 15 is WordNet Affect 1.1. Codes checked against GDELT's
# GCAM master codebook; the first version pointed at Roget's Thesaurus
# categories (c9.*) and GKG theme counts (c18.*), which aren't emotions.
GCAM_EMOTIONS = {
    "c15.198": "fear",          # negative-fear
    "c15.15": "anger",
    "c15.227": "sadness",
    "c15.167": "joy",
    "c15.81": "disgust",
    "c15.254": "surprise",
    "c15.57": "confidence",
    "c15.20": "anticipation",
    "c15.22": "anxiety",
    "c15.150": "hostility",
    "c15.71": "depression",
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
    """Emotion scores per 1,000 words from a GCAM field ("wc:639,c15.15:2,...").

    GCAM holds raw word counts; dividing by the article's word count (wc)
    makes long and short articles comparable.
    """
    emotions = dict.fromkeys(GCAM_EMOTIONS.values(), 0.0)
    if not isinstance(gcam_str, str):
        return emotions
    counts = dict(pair.split(":", 1) for pair in gcam_str.split(",") if ":" in pair)
    try:
        words = float(counts.get("wc", 0))
    except ValueError:
        return emotions
    if words <= 0:
        return emotions
    for code, name in GCAM_EMOTIONS.items():
        try:
            emotions[name] = float(counts.get(code, 0)) / words * 1000
        except ValueError:
            pass
    return emotions


def parse_themes_field(themes_str: str) -> list:
    """Extract top themes from GKG themes field."""
    if not themes_str or not isinstance(themes_str, str):
        return []
    # semicolon-separated "THEME,offset" items; keep the first 10 themes
    themes = (item.split(",")[0].strip() for item in themes_str.split(";"))
    return [t for t in themes if len(t) > 2][:10]


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
        except (ValueError, IndexError):
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

    # Keep real records only. Line breaks inside the extras field split some
    # records, and the fragments ("---", "</PAGE_LINKS>...") came through as
    # rows with no DATE; a real GKG_ID looks like 20261006073000-123
    df = df.filter(pl.col("GKG_ID").str.contains(r"^\d{14}-T?\d+$").fill_null(False))

    logger.info(f"Processed {len(df):,} GKG records")
    return df


@asset(description="Extract emotions and themes from GDELT GKG")
def gdelt_gkg_data(context: AssetExecutionContext) -> pl.DataFrame:
    """
    Extract GKG data with emotions and themes.
    Runs alongside event ingestion, hourly, covering the past hour of batches.
    """
    logger.info("Starting GDELT GKG extraction (emotions & themes)")

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
    logger.info(f"Extracted {len(df):,} GKG records with emotions from {len(parts)} batches")
    return df


@asset(description="Load GKG emotions data into MotherDuck")
def gdelt_gkg_motherduck(context: AssetExecutionContext, gdelt_gkg_data: pl.DataFrame) -> Output:
    """Insert new GKG records and drop ones past retention. Errors fail the run."""
    if gdelt_gkg_data.is_empty():
        return Output(None, metadata={"status": "Skipped", "rows": 0})
    # GKG DATE is a 14-digit number (YYYYMMDDHHMMSS). The dashboard reads 24h,
    # the dbt daily models up to 30 days; older rows are dead weight.
    cutoff = int((datetime.datetime.now(datetime.timezone.utc)
                  - datetime.timedelta(days=GKG_RETENTION_DAYS)).strftime('%Y%m%d000000'))
    with _connect() as con:
        inserted = _insert_new(con, GKG_TABLE, gdelt_gkg_data, "GKG_ID")
        con.execute(f"DELETE FROM {GKG_TABLE} WHERE DATE < ?", [cutoff])
        total = con.execute(f"SELECT COUNT(*) FROM {GKG_TABLE}").fetchone()[0]
    context.log.info(f"Inserted {inserted:,} GKG rows, {total:,} after retention")
    return Output(f"Inserted {inserted:,}", metadata={"rows": inserted, "total": total})


# =============================================================================
# DAGSTER JOB (scheduled by GitHub Actions, see .github/workflows/gdelt_ingest.yml)
# =============================================================================

gdelt_ingestion_job = define_asset_job(
    name="gdelt_ingestion_job",
    selection=["gdelt_raw_data_polars", "gdelt_motherduck_table_polars",
               "gdelt_gkg_data", "gdelt_gkg_motherduck"],
    description="Ingest GDELT events and GKG emotions (run hourly by GitHub Actions)",
)

defs = Definitions(
    assets=[gdelt_raw_data_polars, gdelt_motherduck_table_polars, gdelt_gkg_data, gdelt_gkg_motherduck],
    jobs=[gdelt_ingestion_job],
)
