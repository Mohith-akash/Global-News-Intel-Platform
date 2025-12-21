"""
Data processing functions for headline extraction and cleaning.
"""

import re
import pandas as pd
from urllib.parse import urlparse, unquote

from utils import get_country, get_intensity_label


def clean_headline(text):
    """Remove garbage patterns and dates from headlines."""
    if not text: 
        return None
    
    text = str(text).strip()
    
    # Remove leading/trailing punctuation
    text = re.sub(r'^[.,;:\'\"!?\-_\s]+', '', text)
    text = re.sub(r'[.,;:\'\"!?\-_\s]+$', '', text)
    
    # Reject if too short or single word
    if len(text) < 20 or ' ' not in text:
        return None
    
    words = text.split()
    if len(words) < 4:
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
    text = re.sub(r'\s+\d{4,}', ' ', text)  # Remove 4+ digit numbers anywhere
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
    
    # Quality checks - must have at least 4 words and 20 chars
    words = text.split()
    if len(text) < 20 or len(words) < 4:
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
        if len(words) < 4:
            return None
    
    # Truncate to 100 chars but don't cut mid-word
    if len(text) > 100:
        text = text[:100].rsplit(' ', 1)[0]
    
    # Remove trailing prepositions and fragments
    trailing_junk = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                     'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be', 'if',
                     'with', 'from', 'into', 'that', 'this', 'than', 'when', 'where',
                     'n', 'b', 'na', 'th', 'wh', 's', 't'}
    
    words = text.split()
    while words and (words[-1].lower() in trailing_junk or len(words[-1]) <= 2):
        words.pop()
        if len(words) < 4:
            return None
    
    text = ' '.join(words)
    
    if len(text) < 20 or len(text.split()) < 4:
        return None
    
    return text


def enhance_headline(text, impact_score=None, actor=None):
    """Proper title case - capitalize first letter of each significant word."""
    if not text: 
        return None
    
    # Remove leading/trailing punctuation first
    text = re.sub(r'^[.,;:\'\"!?\-_\s]+', '', text)
    text = re.sub(r'[.,;:\'\"!?\-_\s]+$', '', text)
    
    # Remove trailing numbers (like "1396")
    text = re.sub(r'\s+\d{3,}$', '', text)
    
    if not text or len(text) < 20:
        return None
    
    # Must have at least 4 words
    words = text.split()
    if len(words) < 4:
        return None
    
    # Remove trailing prepositions and fragments
    trailing_junk = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                     'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be', 'if',
                     'with', 'from', 'into', 'that', 'this', 'than', 'when', 'where',
                     'n', 'b', 'na', 'th', 'wh', 's', 't'}
    
    while words and (words[-1].lower() in trailing_junk or len(words[-1]) <= 2):
        words.pop()
        if len(words) < 4:
            return None
    
    if len(words) < 4:
        return None
    
    # Convert to title case
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
    """Extract a readable headline from a news article URL."""
    if not url: 
        return None
    
    try:
        parsed = urlparse(str(url))
        path = unquote(parsed.path)
        
        segments = [s for s in path.split('/') if s and len(s) > 15]
        
        for seg in reversed(segments):
            cleaned = clean_headline(seg)
            if cleaned and len(cleaned) > 20 and len(cleaned.split()) >= 4:
                return enhance_headline(cleaned, impact_score, actor)
        
        return None
    except:
        return None


def process_df(df):
    """Process raw database results into display-ready format."""
    if df.empty: 
        return df
    
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    
    # Use database HEADLINE if available, otherwise extract from URL
    headlines = []
    for _, row in df.iterrows():
        headline = None
        
        # First try database headline - must be at least 25 chars and 4 words
        db_headline = row.get('HEADLINE')
        if db_headline and isinstance(db_headline, str) and len(db_headline.strip()) > 25:
            # Skip if it's just a country/city name (all caps, short)
            if not (db_headline.isupper() and len(db_headline.split()) <= 3):
                cleaned = clean_headline(db_headline)
                if cleaned and len(cleaned.split()) >= 4:
                    headline = enhance_headline(cleaned)
        
        # Fall back to URL extraction only
        if not headline:
            headline = extract_headline(
                row.get('NEWS_LINK', ''), 
                None,  # Don't use actor
                row.get('IMPACT_SCORE', None)
            )
            # Validate extracted headline - must be 4+ words
            if headline and len(headline.split()) < 4:
                headline = None
        
        headlines.append(headline)
    
    df['HEADLINE'] = headlines
    df = df[df['HEADLINE'].notna()]
    
    # Convert country codes to full names
    df['REGION'] = df['ACTOR_COUNTRY_CODE'].apply(
        lambda x: get_country(x) or x if x else 'Global'
    )
    
    # Format dates
    try:
        df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d/%m')
    except:
        df['DATE_FMT'] = df['DATE']
    
    # Add tone indicators
    df['TONE'] = df['IMPACT_SCORE'].apply(
        lambda x: "🔴" if x and x < -4 else ("🟡" if x and x < -1 else ("🟢" if x and x > 2 else "⚪"))
    )
    
    # Add intensity labels
    df['INTENSITY'] = df['IMPACT_SCORE'].apply(get_intensity_label)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['HEADLINE'])
    return df
