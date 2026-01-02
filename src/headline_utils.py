"""
Centralized headline extraction and cleaning utilities.
Single source of truth for all headline processing.
"""

import re
from urllib.parse import urlparse, unquote
from typing import Optional

# Trusted news domains get priority in quality scoring
TRUSTED_DOMAINS = {
    'reuters.com', 'bbc.com', 'apnews.com', 'nytimes.com', 'wsj.com',
    'theguardian.com', 'bloomberg.com', 'ft.com', 'economist.com',
    'washingtonpost.com', 'cnn.com', 'npr.org', 'aljazeera.com'
}

# Patterns that indicate garbage/invalid headlines
REJECT_PATTERNS = [
    r'^[a-f0-9]{8}[-][a-f0-9]{4}',  # UUID-like
    r'^[a-f0-9\-]{20,}$',            # Long hex strings
    r'^(article|post|item|id)[-_]?[a-f0-9]{6,}',
    r'^\d{10,}$',                    # Long numbers
    r'^\d+$',                        # Pure numbers
    r'^[A-Z]{2,5}\s*\d{5,}',         # Code + numbers
]

# Generic segments to skip
SKIP_SEGMENTS = {
    'index', 'home', 'page', 'article', 'news', 'post', 'featured',
    'content', 'story', 'read', 'view', 'watch', 'amp', 'mobile'
}

# Generic/incomplete headline patterns to reject
GENERIC_PATTERNS = [
    'business online', 'full list', 'read more', 'click here',
    'view gallery', 'photo gallery', 'see also', 'related',
    'breaking news', 'just in', 'developing story'
]


def extract_headline_from_url(url: str) -> Optional[str]:
    """
    Extract headline from URL path.
    Returns cleaned headline or None if extraction fails.
    """
    if not url or not isinstance(url, str):
        return None

    try:
        parsed = urlparse(str(url))
        path = unquote(parsed.path)
        segments = [s for s in path.split('/') if s]

        if not segments:
            return None

        best_headline = None
        best_score = 0

        for seg in reversed(segments):
            if len(seg) < 5:
                continue
            if seg.lower() in SKIP_SEGMENTS:
                continue

            headline = _clean_url_segment(seg)
            if not headline:
                continue

            score = _score_segment(headline)
            if score > best_score:
                best_score = score
                best_headline = headline

        return best_headline

    except Exception:
        return None


def _clean_url_segment(text: str) -> Optional[str]:
    """Clean a URL path segment into readable text."""
    if not text:
        return None

    text = str(text).strip()

    # Reject garbage patterns
    for pattern in REJECT_PATTERNS:
        if re.match(pattern, text, re.I):
            return None

    # Remove file extensions
    text = re.sub(r'\.(html?|php|aspx?|jsp|shtml|amp)$', '', text, flags=re.I)

    # Remove leading date patterns
    text = re.sub(r'^\d{8}[-_]?', '', text)
    text = re.sub(r'^\d{4}[-/]\d{2}[-/]\d{2}[-_]?', '', text)
    text = re.sub(r'^\d{4}[-_]', '', text)
    text = re.sub(r'^[a-f0-9]{6,8}[-_]', '', text)

    # Replace separators with spaces
    text = re.sub(r'[-_]+', ' ', text)

    # Remove trailing garbage
    text = re.sub(r'\s+\d{5,}$', '', text)
    text = re.sub(r'\s+[a-f0-9]{8,}$', '', text, flags=re.I)
    text = re.sub(r'\s+\d{1,8}$', '', text)

    # Clean whitespace
    text = ' '.join(text.split())

    # Remove leading/trailing punctuation
    text = re.sub(r'^[.,;:\'\"!?\-_\s\.]+', '', text)
    text = re.sub(r'[.,;:\'\"!?\-_\s]+$', '', text)

    return text.strip() if text else None


def _score_segment(text: str) -> int:
    """Score a text segment for headline quality."""
    if not text:
        return 0

    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # Check minimum quality
    text_alpha = ''.join(c for c in text if c.isalpha())
    if len(text_alpha) < 5:
        return 0

    # Score based on length and word count
    if word_count >= 4 and char_count > 20:
        return word_count * 10 + char_count
    elif word_count >= 3 and char_count > 15:
        return word_count * 5 + char_count
    elif word_count >= 2 and char_count > 10:
        return word_count * 2 + char_count

    return 0


def clean_headline(text: str) -> Optional[str]:
    """
    Clean and normalize a headline for display.
    Single source of truth for headline cleaning.
    """
    if not text or not isinstance(text, str):
        return None

    text = str(text).strip()

    # Remove leading dots and punctuation (multiple passes)
    for _ in range(3):
        text = re.sub(r'^[.,;:\'\"!?\-_\s\.]+', '', text)

    # Remove embedded timestamps (8+ digits)
    text = re.sub(r'\d{8,}', '', text)

    # Remove trailing alphanumeric garbage
    text = re.sub(r'[A-Za-z]?\d{6,}[A-Za-z]*$', '', text)
    text = re.sub(r'\d+[A-Za-z]?$', '', text)
    text = re.sub(r'\s+\d+$', '', text)

    # Fix camelCase: insert spaces before uppercase letters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    # Fix common URL artifacts
    text = re.sub(r'Apos(?=[a-z])', "'", text)
    text = re.sub(r'\bApos\b', "'", text, flags=re.I)
    text = re.sub(r"''", "'", text)
    text = re.sub(r"^'", "", text)

    # Fix number spacing (6 5 -> 6.5)
    text = re.sub(r'(\d)\s+(\d)', r'\1.\2', text)

    # Merge single letters (U S -> US)
    text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', text)
    text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', text)

    # Remove truncated endings
    text = re.sub(r'\s+(for|on|in|to|of|with)\s+\w{1,4}$', '', text)

    # Remove short trailing words
    words = text.strip().split()
    while words and len(words[-1]) <= 2:
        words.pop()
    text = ' '.join(words)

    # Truncate if too long
    if len(text) > 100:
        text = text[:100].rsplit(' ', 1)[0]

    text = text.strip()

    # Remove trailing punctuation
    text = re.sub(r'[.,;:\'\"!?\-_\s]+$', '', text)

    # Quality check: require 4+ words, 20+ chars
    if len(text) < 20 or len(text.split()) < 4:
        return None

    # Check for generic patterns
    text_lower = text.lower()
    for pattern in GENERIC_PATTERNS:
        if text_lower.startswith(pattern):
            return None

    # Title case
    return _title_case(text)


def _title_case(text: str) -> str:
    """Convert text to title case with smart handling of small words."""
    small_words = {
        'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at',
        'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be'
    }

    words = text.lower().split()
    result = []

    for i, word in enumerate(words):
        if i == 0:
            result.append(word.capitalize())
        elif word in small_words:
            result.append(word)
        else:
            result.append(word.capitalize())

    return ' '.join(result)


def score_headline_quality(headline: str, url: str) -> float:
    """
    Score headline quality from 0.0 to 1.0.
    Higher scores indicate better quality headlines.
    """
    if not headline:
        return 0.0

    score = 0.5

    # Domain trust bonus
    if url:
        try:
            domain = urlparse(url).netloc.replace('www.', '').lower()
            if any(d in domain for d in TRUSTED_DOMAINS):
                score += 0.25
        except Exception:
            pass

    # Length sweet spot (40-80 chars)
    if 40 <= len(headline) <= 80:
        score += 0.1
    elif len(headline) < 25 or len(headline) > 120:
        score -= 0.1

    # Word count sweet spot (5-12 words)
    words = len(headline.split())
    if 5 <= words <= 12:
        score += 0.1
    elif words < 4 or words > 15:
        score -= 0.1

    # Penalize garbage patterns
    if re.search(r'\d{6,}', headline):
        score -= 0.3
    if headline.count("'") > 2:
        score -= 0.1
    if re.search(r'[A-Z]{5,}', headline):  # All caps words
        score -= 0.1

    return max(0.0, min(1.0, score))


def dedupe_headlines_semantic(headlines: list) -> list:
    """
    Return indices of unique headlines using similarity detection.
    Removes near-duplicates that differ only slightly.
    """
    from difflib import SequenceMatcher

    keep_indices = []
    seen_normalized = []

    for i, h in enumerate(headlines):
        if not h:
            continue

        # Normalize: lowercase, remove punctuation, get key words
        normalized = ' '.join(sorted(
            re.sub(r'[^\w\s]', '', str(h).lower()).split()
        ))

        # Check similarity against kept headlines
        is_dupe = False
        for seen in seen_normalized:
            ratio = SequenceMatcher(None, normalized, seen).ratio()
            if ratio > 0.7:
                is_dupe = True
                break

        if not is_dupe:
            keep_indices.append(i)
            seen_normalized.append(normalized)

    return keep_indices


def dedupe_headlines_simple(headlines: list) -> list:
    """
    Return indices of unique headlines using simple word-based dedup.
    Faster than semantic dedup, skips first word to catch variations.
    """
    keep_indices = []
    seen_keys = set()

    for i, h in enumerate(headlines):
        if not h:
            continue

        # Key: words 2-6 lowercase (skip first word to catch variations)
        words = str(h).lower().split()
        if len(words) > 1:
            key = ' '.join(words[1:6])
        else:
            key = h.lower()

        if key not in seen_keys:
            keep_indices.append(i)
            seen_keys.add(key)

    return keep_indices


def get_best_headline(db_headline: str, url: str, impact_score: float = None) -> Optional[str]:
    """
    Get the best available headline from DB or URL extraction.
    Prefers DB headline if valid, falls back to URL extraction.
    """
    # Try DB headline first
    if db_headline and isinstance(db_headline, str) and len(db_headline.strip()) > 15:
        cleaned = clean_headline(db_headline)
        if cleaned:
            return cleaned

    # Fall back to URL extraction
    extracted = extract_headline_from_url(url)
    if extracted:
        cleaned = clean_headline(extracted)
        if cleaned:
            return cleaned

    return None