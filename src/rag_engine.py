"""
RAG (Retrieval-Augmented Generation) engine for GDELT platform.
Uses Voyage AI for embeddings and DuckDB for vector similarity search.
"""

import os
import logging
import requests
from typing import Optional
import pandas as pd

logger = logging.getLogger("gdelt.rag")

# Voyage AI Configuration
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = "voyage-3.5-lite"  # 200M free tokens, best value
EMBEDDING_DIMENSIONS = 1024  # voyage-3.5-lite outputs 1024 dimensions


def get_voyage_api_key() -> Optional[str]:
    """Get Voyage API key from environment or Streamlit secrets."""
    key = os.getenv("VOYAGE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("VOYAGE_API_KEY")
    except:
        return None


def get_embedding(text: str) -> Optional[list]:
    """
    Get embedding vector from Voyage AI.
    
    Args:
        text: Text to embed (headline)
        
    Returns:
        List of floats (1024 dimensions) or None if failed
    """
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
                "input": [text.strip()[:512]],  # Limit to 512 chars for efficiency
                "model": VOYAGE_MODEL
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["data"][0]["embedding"]
        else:
            logger.warning(f"Voyage API error: {response.status_code} - {response.text[:100]}")
            return None
            
    except requests.exceptions.Timeout:
        logger.warning("Voyage API timeout")
        return None
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


def get_embeddings_batch(texts: list, batch_size: int = 50) -> list:
    """
    Get embeddings for multiple texts in batches.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call (max 128 for Voyage)
        
    Returns:
        List of embeddings (same order as input, None for failed)
    """
    api_key = get_voyage_api_key()
    if not api_key:
        logger.warning("VOYAGE_API_KEY not found")
        return [None] * len(texts)
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Clean and filter texts
        clean_batch = [t.strip()[:512] if t and isinstance(t, str) and len(t.strip()) >= 10 else "" for t in batch]
        
        # Find valid indices
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
                
                # Map back to original positions
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


def search_similar_headlines(query: str, conn, top_k: int = 10, min_date: str = None) -> pd.DataFrame:
    """
    Find semantically similar headlines using vector similarity search.
    
    Args:
        query: User's question
        conn: DuckDB/MotherDuck connection
        top_k: Number of results to return
        min_date: Optional minimum date filter (YYYYMMDD format)
        
    Returns:
        DataFrame with similar headlines and metadata
    """
    # Get query embedding
    query_embedding = get_embedding(query)
    if query_embedding is None:
        logger.warning("Failed to get query embedding")
        return pd.DataFrame()
    
    # Build date filter
    date_filter = f"AND DATE >= '{min_date}'" if min_date else ""
    
    # Convert embedding to SQL array format
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    
    # DuckDB vector similarity search with deduplication
    sql = f"""
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
        FROM events_dagster
        WHERE EMBEDDING IS NOT NULL 
          AND HEADLINE IS NOT NULL
          AND LENGTH(HEADLINE) > 15
          {date_filter}
        GROUP BY HEADLINE
        ORDER BY similarity DESC
        LIMIT {top_k}
    """
    
    try:
        result = conn.execute(sql).df()
        return result
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return pd.DataFrame()


def rag_query(question: str, conn, llm, top_k: int = 10) -> dict:
    """
    Full RAG pipeline: embed query → search → synthesize answer.
    
    Args:
        question: User's question
        conn: Database connection
        llm: Cerebras LLM instance
        top_k: Number of headlines to retrieve
        
    Returns:
        dict with 'answer', 'headlines' DataFrame, and 'sql' query
    """
    # Search for similar headlines
    headlines_df = search_similar_headlines(question, conn, top_k=top_k)
    
    if headlines_df.empty:
        return {
            "answer": "I couldn't find any relevant events. Try a different question or use SQL mode for precise queries.",
            "headlines": pd.DataFrame(),
            "sql": None
        }
    
    # Build context from retrieved headlines (human-friendly, no technical junk)
    context_parts = []
    for _, row in headlines_df.iterrows():
        headline = row.get('HEADLINE', 'Unknown event')
        # Convert country code to readable name
        country_code = row.get('ACTOR_COUNTRY_CODE', '')
        date_str = str(row.get('DATE', ''))
        
        # Format date nicely
        if len(date_str) == 8:
            try:
                formatted_date = f"{date_str[4:6]}/{date_str[6:8]}"
            except:
                formatted_date = date_str
        else:
            formatted_date = date_str
        
        context_parts.append(f"• {headline}")
    
    context = "\n".join(context_parts)
    
    # Build prompt for LLM - individual event summaries
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
        "sql": "Vector similarity search (RAG mode)"
    }
