"""
Data processing functions for headline extraction and cleaning.
"""

import re
import pandas as pd
from urllib.parse import urlparse, unquote

from src.utils import get_country, get_intensity_label



def extract_headline(url, actor=None, impact_score=None):
    """Extract headline from URL."""
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
            
            # Fix common URL artifacts
            text = re.sub(r'\bApos\b', "'", text, flags=re.I)  # "Apos" -> apostrophe
            text = re.sub(r"''", "'", text)  # Double apostrophe
            text = re.sub(r"'\s*'", "'", text)  # Spaced apostrophe
            
            # Merge single letter words (U S -> US, U K -> UK)
            text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', text)
            text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', text)
            
            # Remove generic/incomplete headlines
            generic_patterns = ['Business Online', 'Full List', 'Read More', 'Click Here', 
                               'View Gallery', 'Photo Gallery', 'See Also', 'Related']
            for pattern in generic_patterns:
                if text.lower().startswith(pattern.lower()):
                    return None
            
            # Quality check: require 4+ words and 20+ chars for meaningful headlines
            words = text.split()
            if len(text) < 20 or len(words) < 4:
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
        # Get DB headline and clean it (remove leading dots, trailing timestamps)
        db_headline = row.get('HEADLINE')
        if db_headline and isinstance(db_headline, str) and len(db_headline.strip()) > 15:
            # Clean the headline
            headline = re.sub(r'^[.,;:\'\"!?\-_\s\.]+', '', str(db_headline))  # Remove leading dots/punctuation
            headline = re.sub(r'\d{8,}', '', headline)  # Remove 8+ digit timestamps
            headline = re.sub(r'\s+\d+$', '', headline)  # Remove trailing numbers
            headline = headline.strip()
            
            # Fix common URL artifacts
            headline = re.sub(r'\bApos\b', "'", headline, flags=re.I)  # "Apos" -> apostrophe
            headline = re.sub(r"''", "'", headline)
            
            # Merge single letter words (U S -> US)
            headline = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', headline)
            headline = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', headline)
            
            # Reject generic/incomplete headlines
            generic_patterns = ['Business Online', 'Full List', 'Read More', 'Click Here']
            is_generic = any(headline.lower().startswith(p.lower()) for p in generic_patterns)
            if is_generic or len(headline) < 20 or len(headline.split()) < 4:
                headline = None
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