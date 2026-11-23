"""
RAG chat: embed the question with Voyage AI, find similar headlines in
MotherDuck, and let the LLM answer from those headlines only.

MotherDuck has no vector index (no vss/HNSW), so similarity is computed on a
pre-filtered candidate set. Two searches run together in one worker call:
- keyword: headlines containing a question word, reranked by similarity
- vector: the most-covered events in the window, ranked by similarity
Keyword search alone misses semantic matches ("Asia" never finds "India"),
vector search alone misses rare stories, so the results are merged. If the
embedding API is down, a plain keyword search answers instead.
"""

import datetime
import logging
import re
import time
from typing import Optional

import pandas as pd
import requests

from src.config import VOYAGE_MODEL, EMBEDDING_DIMENSIONS, get_secret
from src.database import safe_queries, safe_query
from src.headline_utils import clean_headline, score_headline_quality
from src.utils import get_country

logger = logging.getLogger("gdelt.rag")

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
TARGET_TABLE = "events_dagster"
SIMILARITY_THRESHOLD = 0.20  # drop matches below this cosine similarity
PRE_FILTER_LIMIT = 2000      # candidates per search (MotherDuck cost guard)
MIN_ARTICLE_COUNT_FOR_SEARCH = 2
MAX_CONTEXT_EVENTS = 8       # more than this dilutes the answer

STOPWORDS = {
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
RESULT_COLS = """MAX(DATE) AS DATE, HEADLINE, MAX(ACTOR_COUNTRY_CODE) AS ACTOR_COUNTRY_CODE,
    MAX(MAIN_ACTOR) AS MAIN_ACTOR, MAX(IMPACT_SCORE) AS IMPACT_SCORE,
    SUM(ARTICLE_COUNT) AS ARTICLE_COUNT, MAX(NEWS_LINK) AS NEWS_LINK"""
LIKE = "LOWER(HEADLINE) LIKE '%' || ? || '%'"


def get_voyage_api_key() -> Optional[str]:
    return get_secret("VOYAGE_API_KEY")


def get_embedding(text: str) -> Optional[list]:
    """Embedding for one question, or None if the API is unavailable."""
    api_key = get_voyage_api_key()
    if not api_key or not isinstance(text, str) or len(text.strip()) < 10:
        return None
    try:
        r = requests.post(VOYAGE_API_URL, timeout=10,
                          headers={"Authorization": f"Bearer {api_key}"},
                          json={"input": [text.strip()[:512]], "model": VOYAGE_MODEL})
    except requests.exceptions.RequestException as e:
        logger.warning("Voyage request failed: %s", e)
        return None
    if r.status_code != 200:
        logger.warning("Voyage API error: %s", r.status_code)
        return None
    return r.json()["data"][0]["embedding"]


def _extract_keywords(query: str) -> list[str]:
    """Up to 8 question words, letters and digits only (they are bound, not inlined)."""
    words = (''.join(c for c in w.strip().lower() if c.isalnum()) for w in query.split())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS][:8]


def _date_clause(min_date, max_date):
    sql, params = "", []
    if min_date:
        sql, params = " AND DATE >= ?", [min_date]
    if max_date:
        sql, params = sql + " AND DATE <= ?", params + [max_date]
    return sql, params


def _similarity_search(tbl, candidate_sql, candidate_params, embedding, date_sql, date_params, top_k):
    """(sql, params): best `top_k` headlines among the filtered candidates."""
    dim = EMBEDDING_DIMENSIONS
    sql = f"""
        WITH candidates AS (
            SELECT EVENT_ID, DATE, HEADLINE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE,
                   ARTICLE_COUNT, NEWS_LINK, EMBEDDING
            FROM {tbl}
            WHERE EMBEDDING IS NOT NULL AND HEADLINE IS NOT NULL AND LENGTH(HEADLINE) > 15
              AND {candidate_sql}{date_sql}
            ORDER BY ARTICLE_COUNT DESC, EVENT_ID
            LIMIT {PRE_FILTER_LIMIT}
        )
        SELECT {RESULT_COLS},
               MAX(array_cosine_similarity(EMBEDDING::DOUBLE[{dim}], ?::DOUBLE[{dim}])) AS similarity
        FROM candidates
        GROUP BY HEADLINE
        HAVING similarity >= {SIMILARITY_THRESHOLD}
        ORDER BY similarity DESC, HEADLINE
        LIMIT {top_k}
    """
    return sql, candidate_params + date_params + [embedding]


def search_similar_headlines(query: str, top_k: int = 10, min_date: str = None,
                             max_date: str = None, table_name: str = None) -> pd.DataFrame:
    """Keyword and vector search in one worker call, merged by best similarity."""
    tbl = table_name or TARGET_TABLE
    embedding = get_embedding(query)
    if embedding is None:
        logger.warning("No query embedding, falling back to keyword search")
        return _keyword_search(query, top_k, min_date, max_date, tbl)

    date_sql, date_params = _date_clause(min_date, max_date)
    keywords = _extract_keywords(query)
    searches = {"vector": _similarity_search(
        tbl, f"ARTICLE_COUNT >= {MIN_ARTICLE_COUNT_FOR_SEARCH}", [],
        embedding, date_sql, date_params, top_k)}
    if keywords:
        searches = {"keyword": _similarity_search(
            tbl, "(" + " OR ".join([LIKE] * len(keywords)) + ")", keywords,
            embedding, date_sql, date_params, top_k), **searches}

    t0 = time.time()
    found = [df for df in safe_queries(searches).values() if not df.empty]
    logger.info("Similarity search took %.2fs, %s result sets", time.time() - t0, len(found))
    if not found:
        return _keyword_search(query, top_k, min_date, max_date, tbl)
    merged = pd.concat(found, ignore_index=True).sort_values(
        ['similarity', 'HEADLINE'], ascending=[False, True], kind='stable')
    return merged.drop_duplicates(subset=['HEADLINE'], keep='first').head(top_k)


def _keyword_search(query, top_k, min_date, max_date, tbl) -> pd.DataFrame:
    """Ranks headlines by how many question words they contain."""
    keywords = _extract_keywords(query)
    if not keywords:
        return pd.DataFrame()
    date_sql, date_params = _date_clause(min_date, max_date)
    score = " + ".join([f"CASE WHEN {LIKE} THEN 1 ELSE 0 END"] * len(keywords))
    sql = f"""
        SELECT DATE, HEADLINE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK
        FROM {tbl}
        WHERE HEADLINE IS NOT NULL AND LENGTH(HEADLINE) > 15
          AND ({" OR ".join([LIKE] * len(keywords))}){date_sql}
        ORDER BY ({score}) DESC, ARTICLE_COUNT DESC, HEADLINE
        LIMIT {top_k}
    """
    # placeholders in text order: WHERE keywords, dates, then the ORDER BY score
    return safe_query(sql, keywords + date_params + keywords)


def _looks_garbled(headline: str) -> bool:
    """Two or more words like 'Bnzqvyw5' or 'Xes7' (letters mixed with digits, no vowels)."""
    bad = sum(1 for w in headline.split()
              if re.search(r'[a-zA-Z]\d|\d[a-zA-Z]', w) or (len(w) > 4 and not re.search(r'[aeiouAEIOU]', w)))
    return bad >= 2


def _severity_word(score) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return ''
    if s <= -7:
        return 'severe crisis'
    if s <= -4:
        return 'crisis'
    if s <= -2:
        return 'negative'
    return 'positive' if s >= 5 else ''


def _context_line(row) -> Optional[str]:
    headline = str(row.get('HEADLINE', '')).strip()
    if len(headline) < 15 or _looks_garbled(headline):
        return None
    date_raw = str(row.get('DATE', ''))
    try:
        date = datetime.datetime.strptime(date_raw, '%Y%m%d').strftime('%b %d')
    except ValueError:
        date = date_raw
    country = get_country(str(row.get('ACTOR_COUNTRY_CODE', '') or '').strip()) or ''
    severity = _severity_word(row.get('IMPACT_SCORE'))
    # no actor field: GDELT's MAIN_ACTOR is mostly noise
    return ' '.join(filter(None, [f"[{date}]", headline,
                                  f"— {country}" if country else '',
                                  f"[{severity}]" if severity else '']))


def _no_answer(text, headlines=None):
    return {"answer": text, "headlines": pd.DataFrame() if headlines is None else headlines, "sql": None}


def rag_query(question: str, llm, top_k: int = 10, min_date: str = None,
              max_date: str = None, table_name: str = None) -> dict:
    """Search, clean the headlines, and answer from at most 8 of them."""
    t0 = time.time()
    found = search_similar_headlines(question, top_k=top_k, min_date=min_date,
                                     max_date=max_date, table_name=table_name)
    search_time = time.time() - t0
    if found.empty:
        return _no_answer("I couldn't find any relevant events. Try a different question "
                          "or use SQL mode for precise queries.")

    found = found.copy()
    found['HEADLINE'] = [clean_headline(str(h)) if h else None for h in found['HEADLINE']]
    keep = [bool(h) and score_headline_quality(h, str(url)) >= 0.3
            for h, url in zip(found['HEADLINE'], found['NEWS_LINK'])]
    found = found[keep]
    if found.empty:
        return _no_answer("I found some results but the headlines were too low quality "
                          "to display. Try a more specific question.")

    lines, seen = [], set()
    for _, row in found.iterrows():
        line = _context_line(row)
        headline = str(row['HEADLINE']).strip()
        if line and headline not in seen:
            seen.add(headline)
            lines.append(line)
    if not lines:
        return _no_answer("I couldn't find any relevant events. Try a different question or use SQL mode.", found)

    lines = lines[:MAX_CONTEXT_EVENTS]
    context = "\n".join(lines)
    prompt = f"""You are a geopolitical news analyst. Below are {len(lines)} recent events from the GDELT global news database that are relevant to the question.

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

    return {"answer": str(llm.complete(prompt)), "headlines": found,
            "sql": f"Hybrid vector search ({search_time:.1f}s)"}
