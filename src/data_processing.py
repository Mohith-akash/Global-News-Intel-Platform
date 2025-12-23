"""
Data processing functions for headline extraction and cleaning.
"""

import re
import pandas as pd
from urllib.parse import urlparse, unquote

from src.utils import get_country, get_intensity_label


def clean_headline(text):
    """Remove garbage patterns and dates from headlines."""
    if not text: 
        return None
    
    text = str(text).strip()
    
    # Remove leading/trailing punctuation
    text = re.sub(r'^[.,;:\'\"!?\-_\s]+', '', text)
    text = re.sub(r'[.,;:\'\"!?\-_\s]+$', '', text)
    
    # Reject if too short or single word (lenient: 10 chars, 2 words)
    if len(text) < 10 or ' ' not in text:
        return None
    
    words = text.split()
    if len(words) < 2:
        return None
    
    # Reject if it's just a country/city/entity name (all caps single concept)
    if text.isupper() and len(words) <= 3:
        return None
    
    # Reject common garbage patterns
    reject_patterns = [
        r'^[a-f0-9]{8}[-\s][a-f0-9]{4}',
        r'^[a-f0-9\s\-]{20,}$',
        r'^(article|post|item|id)[\s\-_]*[a-f0-9]{8}',
        r'^\d+$',
        r'^[A-Z]{2,5}\s*\d{5,}',
    ]
    
    for pattern in reject_patterns:
        if re.match(pattern, text.lower()): 
            return None
    
    # Remove date patterns at start
    for _ in range(5):
        text = re.sub(r'^\d{4}\s*\d{1,2}\s*\d{1,2}\s+', '', text)
        text = re.sub(r'^\d{8}\s*', '', text)
        text = re.sub(r'^\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\s*', '', text)
        text = re.sub(r'^\d{4}\s+', '', text)
        text = re.sub(r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\s*', '', text)
    
    # Remove garbage patterns anywhere
    text = re.sub(r'\s+\d{1,2}\.\d{5,}', ' ', text)
    text = re.sub(r'\s+\d{4,}', ' ', text)
    text = re.sub(r'\s+[a-z]{1,3}\d[a-z\d]{3,}', ' ', text, flags=re.I)
    text = re.sub(r'\s+[a-z0-9]{10,}(?=\s|$)', ' ', text, flags=re.I)
    text = re.sub(r'\.(html?|php|aspx?|jsp|shtml)$', '', text, flags=re.I)
    text = re.sub(r'[-_]+', ' ', text)
    
    # Remove trailing junk
    text = re.sub(r'\s+\d{1,8}$', '', text)
    text = re.sub(r'\s+[A-Za-z]\d[A-Za-z0-9]{1,5}$', '', text)
    text = re.sub(r'\s+[A-Z]{1,3}\d+$', '', text)
    text = re.sub(r'[\s,]+\d{1,6}$', '', text)
    text = re.sub(r'^[A-Za-z]{1,2}\d+\s+', '', text)
    
    text = ' '.join(text.split())
    
    # Quality checks (lenient: 10 chars, 2 words)
    words = text.split()
    if len(text) < 10 or len(words) < 2:
        return None
    
    text_no_spaces = text.replace(' ', '')
    if text_no_spaces:
        num_count = sum(c.isdigit() for c in text_no_spaces)
        if num_count > len(text_no_spaces) * 0.15:
            return None
    
    hex_count = sum(c in '0123456789abcdefABCDEF' for c in text_no_spaces)
    if hex_count > len(text_no_spaces) * 0.3:
        return None
    
    # Reject if last word looks like a code
    last_word = words[-1]
    if re.match(r'^[A-Za-z]{0,2}\d+[A-Za-z]*$', last_word) and len(last_word) < 8:
        words = words[:-1]
        text = ' '.join(words)
        if len(words) < 2:
            return None
    
    # Truncate to 100 chars
    if len(text) > 100:
        text = text[:100].rsplit(' ', 1)[0]
    
    # Remove trailing junk words
    trailing_junk = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                     'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be', 'if',
                     'with', 'from', 'into', 'that', 'this', 'than', 'when', 'where',
                     'n', 'b', 'na', 'th', 'wh', 's', 't'}
    
    words = text.split()
    while words and (words[-1].lower() in trailing_junk or len(words[-1]) <= 2):
        words.pop()
        if len(words) < 2:
            return None
    
    text = ' '.join(words)
    
    if len(text) < 10 or len(text.split()) < 2:
        return None
    
    return text


def enhance_headline(text, impact_score=None, actor=None):
    """Proper title case - capitalize first letter of each significant word."""
    if not text: 
        return None
    
    text = re.sub(r'^[.,;:\'\"!?\-_\s]+', '', text)
    text = re.sub(r'[.,;:\'\"!?\-_\s]+$', '', text)
    text = re.sub(r'\s+\d{3,}$', '', text)
    
    if not text or len(text) < 10:
        return None
    
    words = text.split()
    if len(words) < 2:
        return None
    
    trailing_junk = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                     'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be', 'if',
                     'with', 'from', 'into', 'that', 'this', 'than', 'when', 'where',
                     'n', 'b', 'na', 'th', 'wh', 's', 't'}
    
    while words and (words[-1].lower() in trailing_junk or len(words[-1]) <= 2):
        words.pop()
        if len(words) < 2:
            return None
    
    if len(words) < 2:
        return None
    
    result = []
    small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                   'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be'}
    
    for i, word in enumerate(words):
        word_lower = word.lower()
        if i == 0:
            result.append(word_lower.capitalize())
        elif word_lower in small_words:
            result.append(word_lower)
        else:
            result.append(word_lower.capitalize())
    
    return ' '.join(result)


def extract_headline(url, actor=None, impact_score=None):
    """Extract a readable headline from a news article URL - very lenient."""
    if not url: 
        return None
    
    try:
        parsed = urlparse(str(url))
        path = unquote(parsed.path)
        
        # Try ALL segments, not just long ones
        segments = [s for s in path.split('/') if s]
        
        best_headline = None
        best_score = 0
        
        for seg in reversed(segments):
            # Skip obviously bad segments
            if len(seg) < 5:
                continue
            if seg.lower() in ('index', 'home', 'page', 'article', 'news', 'post', 'featured', 'content'):
                continue
            
            # Simple cleaning: replace hyphens/underscores with spaces, remove file extensions
            text = re.sub(r'\.(html?|php|aspx?|jsp|shtml|amp)$', '', seg, flags=re.I)
            text = re.sub(r'[-_]+', ' ', text)
            text = text.strip()
            
            # Skip if it looks like a UUID or ID
            if re.match(r'^[a-f0-9]{8,}$', text, re.I):
                continue
            if re.match(r'^\d+$', text):
                continue
            
            # Skip if mostly numbers
            text_alpha = ''.join(c for c in text if c.isalpha())
            if len(text_alpha) < 5:
                continue
            
            words = text.split()
            word_count = len(words)
            
            # Score this segment (prefer longer, multi-word segments)
            if word_count >= 3 and len(text) > 15:
                score = word_count * 10 + len(text)
                if score > best_score:
                    best_score = score
                    best_headline = text
            elif word_count >= 2 and len(text) > 10:
                score = word_count * 5 + len(text)
                if score > best_score:
                    best_score = score
                    best_headline = text
        
        if best_headline:
            text = best_headline
            
            # Reject obvious garbage patterns upfront
            if re.match(r'^article\s+[a-f0-9]{6,}', text, re.I):
                return None
            if re.match(r'^[a-f0-9]{8,}', text, re.I):
                return None
            
            # Remove ALL leading dots and punctuation (multiple passes)
            for _ in range(3):
                text = re.sub(r'^[.,;:\'"!?\-_\s\.]+', '', text)
            
            # Remove leading date patterns
            text = re.sub(r'^\d{4}\s*\d{1,2}\s*\d{1,2}\s+', '', text)
            text = re.sub(r'^\d{8}\s*', '', text)
            text = re.sub(r'^\d+\.?\s*', '', text)
            
            # Remove trailing numbers (any digits at end)
            for _ in range(3):
                text = re.sub(r'\s+\d+$', '', text)
                text = re.sub(r'\s+\d{1,2}\s+\d{1,2}\s+\w{3}$', '', text)  # "14 23 Nov" pattern
            
            # Remove trailing garbage words
            text = re.sub(r'\s+[A-Z][a-z]?\s+[A-Z]$', '', text)
            text = re.sub(r'\s+[A-Za-z]{1,2}$', '', text)
            text = re.sub(r'\s+Today$', '', text, flags=re.I)
            
            # Remove trailing punctuation
            text = re.sub(r'[.,;:\'"!?\-_\s]+$', '', text)
            
            text = text.strip()
            
            # Quality check: require 3+ words and 15+ chars for meaningful headlines
            words = text.split()
            if len(text) < 15 or len(words) < 3:
                return None
            
            # Reject if still looks like garbage (too many numbers/hex chars)
            text_alpha = ''.join(c for c in text if c.isalpha())
            if len(text_alpha) < len(text) * 0.6:
                return None
            
            # Title case it
            words = text.lower().split()
            small = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'by', 'in', 'of', 'as', 'is'}
            result = []
            for i, w in enumerate(words):
                if i == 0 or w not in small:
                    result.append(w.capitalize())
                else:
                    result.append(w)
            return ' '.join(result)
        
        return None
    except:
        return None


def process_df(df):
    """Process raw database results into display-ready format."""
    if df.empty: 
        return df
    
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    
    headlines = []
    for _, row in df.iterrows():
        # Trust DB headlines - they're already cleaned at ingestion
        db_headline = row.get('HEADLINE')
        if db_headline and isinstance(db_headline, str) and len(db_headline.strip()) > 15:
            headline = db_headline
        else:
            # Only extract from URL if DB headline missing/bad
            headline = extract_headline(row.get('NEWS_LINK', ''), None, row.get('IMPACT_SCORE', None))
        
        headlines.append(headline)
    
    df['HEADLINE'] = headlines
    df = df[df['HEADLINE'].notna()]
    
    df['REGION'] = df['ACTOR_COUNTRY_CODE'].apply(
        lambda x: get_country(x) or x if x else 'Global'
    )
    
    try:
        df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d/%m')
    except:
        df['DATE_FMT'] = df['DATE']
    
    df['TONE'] = df['IMPACT_SCORE'].apply(
        lambda x: "🔴" if x and x < -4 else ("🟡" if x and x < -1 else ("🟢" if x and x > 2 else "⚪"))
    )
    
    df['INTENSITY'] = df['IMPACT_SCORE'].apply(get_intensity_label)
    
    df = df.drop_duplicates(subset=['HEADLINE'])
    return df