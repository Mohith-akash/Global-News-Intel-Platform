"""
Backfill script to compute embeddings for existing headlines.
Run this once after deploying RAG to embed historical data.

Usage:
    python etl/backfill_embeddings.py

This script will:
1. Connect to MotherDuck
2. Find headlines without embeddings (last 14 days by default)
3. Compute embeddings using Voyage AI
4. Update the database with embeddings
"""

import os
import sys
import datetime
import duckdb
import requests
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration
TARGET_TABLE = "events_dagster"
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = "voyage-3.5-lite"
BATCH_SIZE = 50
BACKFILL_DAYS = 14  # 2 weeks


def get_embeddings_batch(headlines: list) -> list:
    """Get embeddings for a batch of headlines."""
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: VOYAGE_API_KEY not set")
        return [None] * len(headlines)
    
    # Clean headlines
    clean_batch = []
    valid_indices = []
    for i, h in enumerate(headlines):
        if h and isinstance(h, str) and len(h.strip()) >= 15:
            clean_batch.append(h.strip()[:512])
            valid_indices.append(i)
    
    if not clean_batch:
        return [None] * len(headlines)
    
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
            # Map back to original indices
            results = [None] * len(headlines)
            for i, emb in zip(valid_indices, embeddings):
                results[i] = emb
            return results
        else:
            print(f"Voyage API error: {response.status_code}")
            return [None] * len(headlines)
    except Exception as e:
        print(f"Error: {e}")
        return [None] * len(headlines)


def main():
    print("=" * 60)
    print("GDELT RAG Backfill - Embedding Historical Headlines")
    print("=" * 60)
    
    # Check environment
    token = os.getenv("MOTHERDUCK_TOKEN")
    voyage_key = os.getenv("VOYAGE_API_KEY")
    
    if not token:
        print("ERROR: MOTHERDUCK_TOKEN not set")
        sys.exit(1)
    if not voyage_key:
        print("ERROR: VOYAGE_API_KEY not set")
        sys.exit(1)
    
    print(f"\n✅ Environment configured")
    print(f"📅 Backfilling last {BACKFILL_DAYS} days")
    
    # Calculate date range
    min_date = (datetime.datetime.now() - datetime.timedelta(days=BACKFILL_DAYS)).strftime('%Y%m%d')
    
    try:
        print(f"\n🔌 Connecting to MotherDuck...")
        con = duckdb.connect(f'md:gdelt_db?motherduck_token={token}')
        
        # Check if EMBEDDING column exists
        try:
            con.execute(f"SELECT EMBEDDING FROM {TARGET_TABLE} LIMIT 1")
        except:
            print("➕ Adding EMBEDDING column...")
            con.execute(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN EMBEDDING DOUBLE[]")
        
        # Get headlines needing embeddings
        print(f"🔍 Finding headlines without embeddings (since {min_date})...")
        
        df = con.execute(f"""
            SELECT EVENT_ID, HEADLINE 
            FROM {TARGET_TABLE} 
            WHERE DATE >= '{min_date}' 
              AND HEADLINE IS NOT NULL 
              AND LENGTH(HEADLINE) > 15
              AND EMBEDDING IS NULL
            ORDER BY DATE DESC
        """).df()
        
        total = len(df)
        if total == 0:
            print("✅ No headlines need embedding!")
            con.close()
            return
        
        print(f"📝 Found {total:,} headlines to embed\n")
        
        # Process in batches
        embedded = 0
        failed = 0
        
        for i in range(0, total, BATCH_SIZE):
            batch_df = df.iloc[i:i + BATCH_SIZE]
            headlines = batch_df['HEADLINE'].tolist()
            event_ids = batch_df['EVENT_ID'].tolist()
            
            # Get embeddings
            embeddings = get_embeddings_batch(headlines)
            
            # Update database
            for event_id, embedding in zip(event_ids, embeddings):
                if embedding:
                    try:
                        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                        con.execute(f"""
                            UPDATE {TARGET_TABLE} 
                            SET EMBEDDING = {embedding_str}::DOUBLE[]
                            WHERE EVENT_ID = '{event_id}'
                        """)
                        embedded += 1
                    except Exception as e:
                        print(f"  Error updating {event_id}: {e}")
                        failed += 1
                else:
                    failed += 1
            
            # Progress
            progress = (i + len(batch_df)) / total * 100
            print(f"  Progress: {progress:.1f}% ({embedded:,} embedded, {failed:,} skipped)")
            
            # Rate limiting
            time.sleep(0.6)
        
        print(f"\n{'=' * 60}")
        print(f"✅ Backfill complete!")
        print(f"   Embedded: {embedded:,}")
        print(f"   Skipped:  {failed:,}")
        
        # Final stats
        stats = con.execute(f"SELECT COUNT(*) as total, SUM(CASE WHEN EMBEDDING IS NOT NULL THEN 1 ELSE 0 END) as embedded FROM {TARGET_TABLE}").df()
        print(f"\n📊 Database Status:")
        print(f"   Total events:    {stats.iloc[0]['total']:,}")
        print(f"   With embeddings: {stats.iloc[0]['embedded']:,}")
        
        con.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
