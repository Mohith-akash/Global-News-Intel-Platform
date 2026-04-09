"""
RAG (Retrieval-Augmented Generation) engine for GDELT platform.
Uses Voyage AI for embeddings and DuckDB for vector similarity search.

Optimized for MotherDuck (which doesn't support the vss/HNSW extension):
- Two-stage search: pre-filter by metadata → compute similarity on reduced set
- Hybrid mode: keyword pre-filter → vector reranking  
- Similarity thresholds to skip irrelevant results
- Proper ARRAY type casting for DuckDB 1.5.x
"""

import os
import logging
import requests
import time
from typing import Optional
import pandas as pd

logger = logging.getLogger("gdelt.rag")

from src.config import VOYAGE_MODEL, EMBEDDING_DIMENSIONS

# Configuration
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
TARGET_TABLE = "events_dagster"  # Default, can be overridden per-call

# Search tuning
SIMILARITY_THRESHOLD = 0.20  # Skip results below this cosine similarity
PRE_FILTER_LIMIT = 2000  # Max rows to scan for similarity (MotherDuck perf guard)
MIN_ARTICLE_COUNT_FOR_SEARCH = 2  # Pre-filter: only search events with 2+ articles


def get_voyage_api_key() -> Optional[str]:
    """Get Voyage API key from environment or Streamlit secrets."""
    key = os.getenv("VOYAGE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("VOYAGE_API_KEY")
    except Exception:
        return None


def get_embedding(text: str) -> Optional[list]:
    """Get embedding vector from Voyage AI."""
    api_key = get_voyage_api_key()
    if not api_key:
        logger.warning("VOYAGE_API_KEY not found")
        return None

    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return None

    try:
        response = requests.post(
            VOYAGE_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "input": [text.strip()[:512]],
                "model": VOYAGE_MODEL
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            return data["data"][0]["embedding"]
        else:
            logger.warning(f"Voyage API error: {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        logger.warning("Voyage API timeout")
        return None
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


def get_embeddings_batch(texts: list, batch_size: int = 50) -> list:
    """Get embeddings for multiple texts in batches."""
    api_key = get_voyage_api_key()
    if not api_key:
        logger.warning("VOYAGE_API_KEY not found")
        return [None] * len(texts)

    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        clean_batch = [
            t.strip()[:512] if t and isinstance(t, str) and len(t.strip()) >= 10 else ""
            for t in batch
        ]

        valid_indices = [j for j, t in enumerate(clean_batch) if t]
        valid_texts = [clean_batch[j] for j in valid_indices]

        if not valid_texts:
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
                    "input": valid_texts,
                    "model": VOYAGE_MODEL
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                embeddings = [d["embedding"] for d in data["data"]]

                batch_results = [None] * len(batch)
                for j, emb in zip(valid_indices, embeddings):
                    batch_results[j] = emb
                results.extend(batch_results)
            else:
                logger.warning(f"Voyage batch API error: {response.status_code}")
                results.extend([None] * len(batch))

        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            results.extend([None] * len(batch))

    return results


def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a query string, sanitized for SQL."""
    stopwords = {
        'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'about', 'tell', 'me',
        'show', 'find', 'get', 'give', 'any', 'some', 'all', 'most', 'many',
        'much', 'more', 'than', 'that', 'this', 'these', 'those', 'there',
        'here', 'between', 'during', 'recent', 'latest', 'events', 'news',
        'happening', 'going', 'related',
    }
    words = [w.strip().lower() for w in query.split() if len(w.strip()) > 2]
    # Sanitize: keep only alphanumeric chars to prevent SQL injection
    words = [''.join(c for c in w if c.isalnum()) for w in words]
    keywords = [w for w in words if w and len(w) > 2 and w not in stopwords]
    return keywords[:8]  # Cap at 8 keywords


def search_similar_headlines(
    query: str,
    conn,
    top_k: int = 10,
    min_date: str = None,
    table_name: str = None
) -> pd.DataFrame:
    """
    Find semantically similar headlines using optimized vector similarity search.
    
    Optimized for MotherDuck (no HNSW/vss extension):
    1. Pre-filters by date + article count to reduce scan size
    2. Computes cosine similarity only on the filtered candidate set
    3. Applies similarity threshold to skip irrelevant noise
    4. Falls back to hybrid search (keyword + vector rerank) if pure vector fails
    """
    tbl = table_name or TARGET_TABLE
    
    # Step 1: Get query embedding
    t0 = time.time()
    query_embedding = get_embedding(query)
    embed_time = time.time() - t0
    
    if query_embedding is None:
        logger.warning("Failed to get query embedding, falling back to keyword search")
        return _fallback_keyword_search(query, conn, top_k, min_date, table_name=tbl)

    logger.info(f"Query embedding retrieved in {embed_time:.2f}s")

    # Step 2: Build the embedding literal string
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    date_filter = f"AND DATE >= '{min_date}'" if min_date else ""

    # Step 3: Run BOTH hybrid and vector search, merge results
    # Hybrid is fast but misses semantic matches (e.g. "Asia" won't find "India" headlines)
    # Vector search is slower but finds semantically related events
    keywords = _extract_keywords(query)
    all_results = []

    if keywords:
        hybrid = _hybrid_search(
            keywords, embedding_str, conn, tbl,
            top_k, date_filter
        )
        if hybrid is not None and not hybrid.empty:
            logger.info(f"Hybrid search returned {len(hybrid)} results")
            all_results.append(hybrid)

    vector = _prefiltered_vector_search(
        embedding_str, conn, tbl,
        top_k, date_filter
    )
    if vector is not None and not vector.empty:
        logger.info(f"Vector search returned {len(vector)} results")
        all_results.append(vector)

    if all_results:
        import pandas as pd
        merged = pd.concat(all_results, ignore_index=True)
        # Deduplicate by headline, keep the one with highest similarity
        merged = merged.sort_values('similarity', ascending=False)
        merged = merged.drop_duplicates(subset=['HEADLINE'], keep='first')
        merged = merged.head(top_k)
        return merged

    # Last resort — keyword search without vectors
    logger.info("Vector searches returned no results, falling back to keyword search")
    return _fallback_keyword_search(query, conn, top_k, min_date, table_name=tbl)


def _hybrid_search(
    keywords: list[str],
    embedding_str: str,
    conn,
    tbl: str,
    top_k: int,
    date_filter: str,
) -> Optional[pd.DataFrame]:
    """
    Two-stage hybrid search:
    Stage 1: Use keyword LIKE filters to narrow candidates (~100-500 rows)
    Stage 2: Compute vector cosine similarity only on those candidates, rerank
    
    This avoids scanning the entire table's embeddings on MotherDuck.
    """
    # Build keyword filter — require ANY keyword to match headline
    like_clauses = " OR ".join([f"LOWER(HEADLINE) LIKE '%' || ? || '%'" for _ in keywords])
    
    sql = f"""
        WITH keyword_candidates AS (
            SELECT 
                EVENT_ID, DATE, HEADLINE, ACTOR_COUNTRY_CODE,
                MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK,
                EMBEDDING
            FROM {tbl}
            WHERE EMBEDDING IS NOT NULL 
              AND HEADLINE IS NOT NULL
              AND LENGTH(HEADLINE) > 15
              AND ({like_clauses})
              {date_filter}
            ORDER BY ARTICLE_COUNT DESC
            LIMIT {PRE_FILTER_LIMIT}
        )
        SELECT 
            MAX(DATE) as DATE,
            HEADLINE,
            MAX(ACTOR_COUNTRY_CODE) as ACTOR_COUNTRY_CODE,
            MAX(MAIN_ACTOR) as MAIN_ACTOR,
            MAX(IMPACT_SCORE) as IMPACT_SCORE,
            SUM(ARTICLE_COUNT) as ARTICLE_COUNT,
            MAX(NEWS_LINK) as NEWS_LINK,
            MAX(array_cosine_similarity(
                EMBEDDING::DOUBLE[{EMBEDDING_DIMENSIONS}], 
                {embedding_str}::DOUBLE[{EMBEDDING_DIMENSIONS}]
            )) as similarity
        FROM keyword_candidates
        GROUP BY HEADLINE
        HAVING similarity >= {SIMILARITY_THRESHOLD}
        ORDER BY similarity DESC
        LIMIT {top_k}
    """
    
    try:
        t0 = time.time()
        result = conn.execute(sql, keywords).df()
        query_time = time.time() - t0
        logger.info(f"Hybrid search completed in {query_time:.2f}s, {len(result)} results")
        return result if not result.empty else None
    except Exception as e:
        logger.warning(f"Hybrid search error: {e}")
        return None


def _prefiltered_vector_search(
    embedding_str: str,
    conn,
    tbl: str,
    top_k: int,
    date_filter: str,
) -> Optional[pd.DataFrame]:
    """
    Pre-filtered vector search: narrows by article count before computing similarity.
    Only scans events with ARTICLE_COUNT >= threshold, drastically reducing work.
    """
    sql = f"""
        WITH candidates AS (
            SELECT 
                EVENT_ID, DATE, HEADLINE, ACTOR_COUNTRY_CODE,
                MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK, 
                EMBEDDING
            FROM {tbl}
            WHERE EMBEDDING IS NOT NULL 
              AND HEADLINE IS NOT NULL
              AND LENGTH(HEADLINE) > 15
              AND ARTICLE_COUNT >= {MIN_ARTICLE_COUNT_FOR_SEARCH}
              {date_filter}
            ORDER BY ARTICLE_COUNT DESC
            LIMIT {PRE_FILTER_LIMIT}
        )
        SELECT 
            MAX(DATE) as DATE,
            HEADLINE,
            MAX(ACTOR_COUNTRY_CODE) as ACTOR_COUNTRY_CODE,
            MAX(MAIN_ACTOR) as MAIN_ACTOR,
            MAX(IMPACT_SCORE) as IMPACT_SCORE,
            SUM(ARTICLE_COUNT) as ARTICLE_COUNT,
            MAX(NEWS_LINK) as NEWS_LINK,
            MAX(array_cosine_similarity(
                EMBEDDING::DOUBLE[{EMBEDDING_DIMENSIONS}], 
                {embedding_str}::DOUBLE[{EMBEDDING_DIMENSIONS}]
            )) as similarity
        FROM candidates
        GROUP BY HEADLINE
        HAVING similarity >= {SIMILARITY_THRESHOLD}
        ORDER BY similarity DESC
        LIMIT {top_k}
    """
    
    try:
        t0 = time.time()
        result = conn.execute(sql).df()
        query_time = time.time() - t0
        logger.info(f"Pre-filtered vector search completed in {query_time:.2f}s, {len(result)} results")
        return result if not result.empty else None
    except Exception as e:
        logger.warning(f"Pre-filtered vector search error: {e}")
        return None


def _fallback_keyword_search(
    query: str,
    conn,
    top_k: int = 10,
    min_date: str = None,
    table_name: str = None
) -> pd.DataFrame:
    """
    Fallback keyword search when vector search fails.
    Uses relevance scoring based on keyword match count for better ranking.
    """
    tbl = table_name or TARGET_TABLE
    date_filter = f"AND DATE >= '{min_date}'" if min_date else ""
    keywords = _extract_keywords(query)
    
    if not keywords:
        return pd.DataFrame()
    
    # Build parameterized LIKE clauses for each keyword
    like_clauses = " OR ".join(["LOWER(HEADLINE) LIKE '%' || ? || '%'" for _ in keywords])

    # Score by how many keywords match (poor man's relevance ranking)
    score_parts = " + ".join([
        "CASE WHEN LOWER(HEADLINE) LIKE '%' || ? || '%' THEN 1 ELSE 0 END"
        for _ in keywords
    ])
    
    sql = f"""
        SELECT 
            DATE,
            HEADLINE,
            ACTOR_COUNTRY_CODE,
            MAIN_ACTOR,
            IMPACT_SCORE,
            ARTICLE_COUNT,
            NEWS_LINK,
            ({score_parts}) as keyword_score
        FROM {tbl}
        WHERE HEADLINE IS NOT NULL
          AND LENGTH(HEADLINE) > 15
          AND ({like_clauses})
          {date_filter}
        ORDER BY keyword_score DESC, ARTICLE_COUNT DESC
        LIMIT {top_k}
    """
    
    try:
        # score_parts params first (SELECT position), then like_clauses params (WHERE position)
        result = conn.execute(sql, keywords + keywords).df()
        # Drop the scoring column before returning
        if 'keyword_score' in result.columns:
            result = result.drop(columns=['keyword_score'])
        return result
    except Exception as e:
        logger.error(f"Fallback search error: {e}")
        return pd.DataFrame()


def rag_query(question: str, conn, llm, top_k: int = 10, min_date: str = None, table_name: str = None) -> dict:
    """Full RAG pipeline: embed query → search → synthesize answer."""
    t0 = time.time()
    headlines_df = search_similar_headlines(question, conn, top_k=top_k, min_date=min_date, table_name=table_name)
    search_time = time.time() - t0
    logger.info(f"RAG search completed in {search_time:.2f}s")

    if headlines_df.empty:
        return {
            "answer": "I couldn't find any relevant events. Try a different question or use SQL mode for precise queries.",
            "headlines": pd.DataFrame(),
            "sql": None
        }

    # Clean headlines and drop slugs/garbage before anything else
    from src.headline_utils import clean_headline, score_headline_quality
    valid_rows = []
    for idx, row in headlines_df.iterrows():
        raw = row.get('HEADLINE', '')
        url = row.get('NEWS_LINK', '')
        cleaned = clean_headline(str(raw)) if raw else None
        quality = score_headline_quality(cleaned, str(url)) if cleaned else 0
        if cleaned and quality >= 0.3:
            headlines_df.at[idx, 'HEADLINE'] = cleaned
            valid_rows.append(idx)
    headlines_df = headlines_df.loc[valid_rows].copy()

    if headlines_df.empty:
        return {
            "answer": "I found some results but the headlines were too low quality to display. Try a more specific question.",
            "headlines": pd.DataFrame(),
            "sql": None
        }

    # GDELT MAIN_ACTOR codes that are entity-type labels, not real actors — skip these
    _GDELT_NOISE_ACTORS = {
        'army', 'government', 'police', 'military', 'industry', 'business',
        'media', 'economist', 'official', 'minister', 'rebel', 'opposition',
        'protester', 'civilian', 'refugee', 'journalist', 'activist',
        'diplomat', 'court', 'parliament', 'congress', 'president',
        'citizen', 'citizen', 'company', 'agency', 'spokesperson',
        # Country names that appear as MAIN_ACTOR in GDELT
        'russia', 'china', 'iran', 'israel', 'india', 'ukraine',
        'united states', 'united kingdom', 'france', 'germany',
    }

    context_parts = []
    _seen_headlines = set()
    for _, row in headlines_df.iterrows():
        headline = row.get('HEADLINE', '')
        headline = str(headline).strip()
        if not headline or len(headline) < 15:
            continue

        # Skip garbage headlines with random character sequences (e.g. "Bnzqvyw5 Xes7")
        import re
        words = headline.split()
        garbage_words = sum(1 for w in words if re.search(r'[a-zA-Z]\d|\d[a-zA-Z]', w) or
                           (len(w) > 4 and not re.search(r'[aeiouAEIOU]', w)))
        if garbage_words >= 2:
            continue

        date_raw = str(row.get('DATE', ''))
        try:
            import datetime
            date_fmt = datetime.datetime.strptime(date_raw, '%Y%m%d').strftime('%b %d')
        except Exception:
            date_fmt = date_raw

        country_code = str(row.get('ACTOR_COUNTRY_CODE', '') or '').strip()
        try:
            from src.utils import get_country
            country = get_country(country_code) or ''
        except Exception:
            country = ''

        score = row.get('IMPACT_SCORE')
        severity = ''
        if score is not None:
            try:
                s = float(score)
                if s <= -7:
                    severity = 'severe crisis'
                elif s <= -4:
                    severity = 'crisis'
                elif s <= -2:
                    severity = 'negative'
                elif s >= 5:
                    severity = 'positive'
            except Exception:
                pass

        # Build context line — no actor field, it's almost always noise from GDELT
        parts = [f"[{date_fmt}]", headline]
        if country:
            parts.append(f"— {country}")
        if severity:
            parts.append(f"[{severity}]")

        line = ' '.join(parts)
        # Deduplicate by headline text
        if headline not in _seen_headlines:
            _seen_headlines.add(headline)
            context_parts.append(line)

    if not context_parts:
        return {
            "answer": "I couldn't find any relevant events. Try a different question or use SQL mode.",
            "headlines": headlines_df,
            "sql": None
        }

    # Cap context to 8 best events for LLM (more would dilute quality)
    context_parts = context_parts[:8]
    context = "\n".join(context_parts)
    n = len(context_parts)

    prompt = f"""You are a geopolitical news analyst. Below are {n} recent events from the GDELT global news database that are relevant to the question.

{context}

Question: {question}

Write a concise analytical answer (3-5 sentences) that:
- Directly answers the question based on the events above
- Groups related events by country or theme where relevant
- Uses the dates and severity labels from the data to ground your points
- Does not simply repeat each headline one by one

After the paragraph, list the key events using this EXACT markdown format (one per line, with a blank line before the list):

**Key events:**
- [Date] Headline — Country [severity if any]
- [Date] Headline — Country [severity if any]

Start with the analytical paragraph:"""

    try:
        answer = str(llm.complete(prompt))
    except Exception as e:
        logger.error(f"LLM error: {e}")
        answer = "Error generating response. Please try again."

    return {
        "answer": answer,
        "headlines": headlines_df,
        "sql": f"Hybrid vector search ({search_time:.1f}s)"
    }