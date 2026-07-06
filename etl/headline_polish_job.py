"""
Headline polish job: repairs URL-slug headlines with Cerebras before display.

GDELT has no headlines, so ingestion reconstructs them from URL slugs. The
regex cleanup in src/headline_utils handles most cases but cannot fix acronym
casing (US/RAF/NATO), keyword-stuffed slugs, or glued tokens ("america250").
This job sends the top displayed-tier headlines to Cerebras in batches and
stores repaired versions in HEADLINE_AI; the dashboard prefers HEADLINE_AI
and falls back to the regex path when it is NULL.

Guardrails: temperature 0, strict reformat-only prompt, and per-headline
validation that rejects outputs with invented words or digits. Runs every
12 hours alongside the embedding job.
"""

import json
import logging
import os
import re
import time

import duckdb
import requests
from dagster import (
    AssetExecutionContext,
    Definitions,
    Output,
    ScheduleDefinition,
    asset,
    define_asset_job,
)
from dotenv import load_dotenv

load_dotenv()

TARGET_TABLE = "events_dagster"
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
try:
    from src.config import CEREBRAS_MODEL
except ImportError:
    CEREBRAS_MODEL = "gpt-oss-120b"

MIN_ARTICLE_COUNT = 3      # same displayed-tier filter as the dashboard queries
HEADLINES_PER_RUN = 500    # distinct headlines per run
BATCH_SIZE = 25            # headlines per LLM request
LOOKBACK_DAYS = 2          # only polish headlines still likely to be displayed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "These are news headlines extracted from URL slugs. Repair each one:\n"
    "- fix capitalization (headline case; acronyms like US, RAF, NATO in caps)\n"
    "- remove duplicated or keyword-stuffed words\n"
    "- split glued tokens (e.g. 'america250' -> 'America 250')\n"
    "- if the ending is a dangling fragment, trim it to the last complete phrase\n"
    "- NEVER add facts or words that are not in the input\n"
    "- if a headline is unfixable garbage, use null\n"
    "Return ONLY a JSON array of strings/null, same length and order as the input.\n\n"
    "{headlines_json}"
)


def validate_polished(original: str, polished) -> str | None:
    """Accept an LLM-polished headline only if it stays faithful to the input.

    Rejects outputs that are empty, wildly resized, contain digits not present
    in the original, or contain words that cannot be found in the original
    (hallucination guard). Returns the cleaned headline or None.
    """
    if not polished or not isinstance(polished, str) or not original:
        return None

    polished = polished.strip()
    if not (15 <= len(polished) <= 200):
        return None
    if len(polished) > 1.5 * len(original) + 10:
        return None

    original_lower = original.lower()

    # No digits that don't exist in the original
    for number in re.findall(r"\d+", polished):
        if number not in original_lower:
            return None

    # Every word must exist somewhere in the original (case-insensitive);
    # substring match tolerates de-glued tokens and stripped punctuation.
    for word in re.findall(r"[a-z]+", polished.lower()):
        if len(word) > 2 and word not in original_lower:
            return None

    return polished


def polish_batch(headlines: list[str], api_key: str) -> list[str | None]:
    """Send one batch of headlines to Cerebras; return validated results.

    Any parse failure or API error returns all-None for the batch — rows stay
    NULL and the dashboard keeps using the regex-cleaned headline.
    """
    prompt = PROMPT_TEMPLATE.format(headlines_json=json.dumps(headlines))
    try:
        response = requests.post(
            CEREBRAS_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": CEREBRAS_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 4000,
            },
            timeout=60,
        )
        if response.status_code != 200:
            logger.warning(f"Cerebras API error: {response.status_code}")
            return [None] * len(headlines)

        content = response.json()["choices"][0]["message"]["content"].strip()
        content = content[content.index("[") : content.rindex("]") + 1]
        results = json.loads(content)
        if not isinstance(results, list) or len(results) != len(headlines):
            logger.warning("Cerebras returned wrong shape, skipping batch")
            return [None] * len(headlines)

        return [validate_polished(orig, res) for orig, res in zip(headlines, results)]
    except Exception as e:
        logger.warning(f"Polish batch failed: {e}")
        return [None] * len(headlines)


@asset(description="Repair displayed-tier headlines with Cerebras (runs every 12 hours)")
def gdelt_headline_polish(context: AssetExecutionContext) -> Output:
    token = os.getenv("MOTHERDUCK_TOKEN")
    api_key = os.getenv("CEREBRAS_API_KEY")

    if not token:
        return Output(None, metadata={"status": "Error", "message": "Missing MOTHERDUCK_TOKEN"})
    if not api_key:
        context.log.warning("CEREBRAS_API_KEY not set, skipping headline polish")
        return Output(None, metadata={"status": "Skipped", "message": "No API key"})

    import datetime

    since = (datetime.date.today() - datetime.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")

    with duckdb.connect(f"md:gdelt_db?motherduck_token={token}") as con:
        # Distinct headlines: top stories repeat across many events, so one
        # LLM call covers all duplicates via the UPDATE below.
        rows = con.execute(
            f"""
            SELECT HEADLINE, MAX(ARTICLE_COUNT) AS max_articles
            FROM {TARGET_TABLE}
            WHERE HEADLINE_AI IS NULL
              AND HEADLINE IS NOT NULL
              AND LENGTH(HEADLINE) > 20
              AND ARTICLE_COUNT > {MIN_ARTICLE_COUNT}
              AND DATE >= '{since}'
            GROUP BY HEADLINE
            ORDER BY max_articles DESC
            LIMIT {HEADLINES_PER_RUN}
            """
        ).fetchall()
        headlines = [r[0] for r in rows]

        if not headlines:
            context.log.info("No headlines need polishing")
            return Output("Nothing to polish", metadata={"status": "Complete", "polished": 0})

        logger.info(f"Polishing {len(headlines)} distinct headlines...")

        polished_count = 0
        events_updated = 0
        for i in range(0, len(headlines), BATCH_SIZE):
            batch = headlines[i : i + BATCH_SIZE]
            results = polish_batch(batch, api_key)
            for original, polished in zip(batch, results):
                if polished is None:
                    continue
                cursor = con.execute(
                    f"""
                    UPDATE {TARGET_TABLE}
                    SET HEADLINE_AI = ?
                    WHERE HEADLINE = ? AND HEADLINE_AI IS NULL
                    """,
                    [polished, original],
                )
                polished_count += 1
                try:
                    events_updated += cursor.fetchall()[0][0]
                except Exception:
                    pass
            time.sleep(1)  # rate limiting

        context.log.info(
            f"Polished {polished_count}/{len(headlines)} headlines ({events_updated} events updated)"
        )
        return Output(
            f"Polished {polished_count} headlines",
            metadata={
                "distinct_headlines": len(headlines),
                "polished": polished_count,
                "events_updated": events_updated,
            },
        )


gdelt_headline_polish_job = define_asset_job(
    name="gdelt_headline_polish_job",
    selection=["gdelt_headline_polish"],
    description="Repair displayed headlines with Cerebras every 12 hours",
)

gdelt_headline_polish_schedule = ScheduleDefinition(
    job=gdelt_headline_polish_job,
    cron_schedule="15 */12 * * *",  # offset from the embedding job
    execution_timezone="UTC",
    description="Run headline polish every 12 hours",
)

defs = Definitions(
    assets=[gdelt_headline_polish],
    jobs=[gdelt_headline_polish_job],
    schedules=[gdelt_headline_polish_schedule],
)
