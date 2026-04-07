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
SIMILARITY_THRESHOLD = 0.25  # Skip results below this cosine similarity
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

    # Step 3: Try hybrid search first (keyword pre-filter → vector rerank)
    # This is much faster than full-table vector scan on MotherDuck
    keywords = _extract_keywords(query)
    
    if keywords:
        result = _hybrid_search(
            keywords, embedding_str, conn, tbl, 
            top_k, date_filter
        )
        if result is not None and not result.empty:
            logger.info(f"Hybrid search returned {len(result)} results")
            return result
    
    # Step 4: Fall back to pre-filtered vector scan
    # Scan only high-article-count events with embeddings (much smaller set)
    result = _prefiltered_vector_search(
        embedding_str, conn, tbl,
        top_k, date_filter
    )
    if result is not None and not result.empty:
        logger.info(f"Pre-filtered vector search returned {len(result)} results")
        return result
    
    # Step 5: Last resort — keyword search without vectors
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
    like_clauses = " OR ".join([f"LOWER(HEADLINE) LIKE '%{kw}%'" for kw in keywords])
    
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
        result = conn.execute(sql).df()
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
    
    # Build LIKE clauses for each keyword  
    like_clauses = " OR ".join([f"LOWER(HEADLINE) LIKE '%{kw}%'" for kw in keywords])
    
    # Score by how many keywords match (poor man's relevance ranking)
    score_parts = " + ".join([
        f"CASE WHEN LOWER(HEADLINE) LIKE '%{kw}%' THEN 1 ELSE 0 END"
        for kw in keywords
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
        result = conn.execute(sql).df()
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

    context_parts = []
    for _, row in headlines_df.iterrows():
        headline = row.get('HEADLINE', 'Unknown event')
        context_parts.append(f"• {headline}")

    context = "\n".join(context_parts)

    prompt = f"""Here are the top 5 news events related to the question:

{context}

Question: {question}

For each event, write a 2-3 sentence summary that:
- Explains WHY this event matters (not just repeating the headline)
- Provides context or background that isn't obvious
- Uses full country names

Format your response like this:
**1. [First Event Title]**
[2-3 sentences of insight]

**2. [Second Event Title]**
[2-3 sentences of insight]

(Continue for all events)

Start directly with the numbered list, no introduction needed:"""

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