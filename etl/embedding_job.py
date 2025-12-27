"""
GDELT Embedding Job - Runs every 12 hours
Computes Voyage AI embeddings for events that don't have them yet.

This is separated from the main ingestion job to:
1. Respect Voyage AI rate limits
2. Allow faster 15-min ingestion updates
3. Batch embedding generation efficiently
"""

import requests
import duckdb
import os
import time
import logging
from dagster import asset, Output, Definitions, ScheduleDefinition, define_asset_job, AssetExecutionContext
from dotenv import load_dotenv

load_dotenv()

# Configuration
TARGET_TABLE = "events_dagster"
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = "voyage-3.5-lite"
EMBEDDING_DIMENSIONS = 1024
MIN_ARTICLE_COUNT = 3  # Only embed events with 3+ articles
BATCH_SIZE = 50  # Voyage AI batch size
EVENTS_PER_RUN = 500  # Max events to embed per run (token efficiency)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_embedding_text(row: dict) -> str | None:
    """Create rich embedding text from multiple fields."""
    parts = []
    
    actor = row.get('MAIN_ACTOR', '')
    if actor and isinstance(actor, str) and len(actor.strip()) > 2:
        parts.append(actor.strip())
    
    country = row.get('ACTOR_COUNTRY_CODE', '')
    if country and isinstance(country, str) and len(country) == 3:
        parts.append(country)
    
    headline = row.get('HEADLINE', '')
    if headline and isinstance(headline, str) and len(headline.strip()) > 10:
        parts.append(headline.strip())
    else:
        parts.append("news event")
    
    text = ' '.join(parts)
    
    if len(text) < 15:
        return None
    
    return text[:512]


def get_embeddings_batch(texts: list, api_key: str, batch_size: int = BATCH_SIZE) -> list:
    """Get embeddings for multiple texts from Voyage AI."""
    if not api_key:
        return [None] * len(texts)
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        clean_batch = []
        valid_indices = []
        
        for j, text in enumerate(batch):
            if text and isinstance(text, str) and len(text.strip()) >= 15:
                clean_batch.append(text.strip()[:512])
                valid_indices.append(j)
        
        if not clean_batch:
            results.extend([None] * len(batch))
            continue
        
        try:
            response = requests.post(
                VOYAGE_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": clean_batch,
                    "model": VOYAGE_MODEL
                },
                timeout=30
            )
            
            if response.status_code == 200:
                embeddings = [d["embedding"] for d in response.json()["data"]]
                batch_results = [None] * len(batch)
                for j, emb in zip(valid_indices, embeddings):
                    batch_results[j] = emb
                results.extend(batch_results)
            else:
                logger.warning(f"Voyage API error: {response.status_code}")
                results.extend([None] * len(batch))
        except Exception as e:
            logger.warning(f"Embedding error: {e}")
            results.extend([None] * len(batch))
        
        # Rate limiting
        time.sleep(0.6)
    
    return results


@asset(description="Compute embeddings for events without them (runs every 12 hours)")
def gdelt_embeddings_job(context: AssetExecutionContext) -> Output:
    """
    Compute Voyage AI embeddings for events that don't have them yet.
    Only embeds events with ARTICLE_COUNT >= 3 (quality filter).
    Processes up to 500 events per run to respect rate limits.
    """
    token = os.getenv("MOTHERDUCK_TOKEN")
    voyage_key = os.getenv("VOYAGE_API_KEY")
    
    if not token:
        return Output(None, metadata={"status": "Error", "message": "Missing MOTHERDUCK_TOKEN"})
    
    if not voyage_key:
        context.log.warning("VOYAGE_API_KEY not set, skipping embedding generation")
        return Output(None, metadata={"status": "Skipped", "message": "No API key"})
    
    try:
        with duckdb.connect(f'md:gdelt_db?motherduck_token={token}') as con:
            # Get events needing embeddings (recent first, quality filter)
            events = con.execute(f"""
                SELECT EVENT_ID, MAIN_ACTOR, ACTOR_COUNTRY_CODE, HEADLINE
                FROM {TARGET_TABLE}
                WHERE EMBEDDING IS NULL
                  AND ARTICLE_COUNT >= {MIN_ARTICLE_COUNT}
                  AND HEADLINE IS NOT NULL
                ORDER BY DATE DESC
                LIMIT {EVENTS_PER_RUN}
            """).fetchdf()
            
            if events.empty:
                context.log.info("No events need embedding")
                return Output("No events to embed", metadata={"status": "Complete", "embedded": 0})
            
            logger.info(f"🔢 Computing embeddings for {len(events)} events...")
            
            # Create embedding texts
            embedding_texts = []
            for _, row in events.iterrows():
                text = create_embedding_text(row.to_dict())
                embedding_texts.append(text)
            
            # Get embeddings
            embeddings = get_embeddings_batch(embedding_texts, voyage_key)
            
            # Update database
            updated = 0
            for i, (_, row) in enumerate(events.iterrows()):
                if embeddings[i] is not None:
                    event_id = row['EVENT_ID']
                    embedding = embeddings[i]
                    con.execute(f"""
                        UPDATE {TARGET_TABLE}
                        SET EMBEDDING = ?
                        WHERE EVENT_ID = ?
                    """, [embedding, event_id])
                    updated += 1
            
            # Get stats
            total_embedded = con.execute(f"""
                SELECT COUNT(*) FROM {TARGET_TABLE} WHERE EMBEDDING IS NOT NULL
            """).fetchone()[0]
            
            context.log.info(f"✅ Updated {updated} embeddings. Total embedded: {total_embedded}")
            return Output(
                f"Embedded {updated} events",
                metadata={"embedded_this_run": updated, "total_embedded": total_embedded}
            )
            
    except Exception as e:
        context.log.error(f"❌ Embedding error: {e}")
        return Output(None, metadata={"status": "Error", "message": str(e)})


# Job definition
gdelt_embedding_job = define_asset_job(
    name="gdelt_embedding_job",
    selection=["gdelt_embeddings_job"],
    description="Compute embeddings every 12 hours"
)

# Schedule: Every 12 hours
gdelt_embedding_schedule = ScheduleDefinition(
    job=gdelt_embedding_job,
    cron_schedule="0 */12 * * *",  # At minute 0 of every 12th hour
    execution_timezone="UTC",
    description="Run embedding generation every 12 hours"
)

defs = Definitions(
    assets=[gdelt_embeddings_job],
    jobs=[gdelt_embedding_job],
    schedules=[gdelt_embedding_schedule]
)
