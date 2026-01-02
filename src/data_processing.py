"""
Data processing functions for headline extraction and cleaning.
"""

import re
import pandas as pd

from src.headline_utils import (
    get_best_headline,
    clean_headline,
    dedupe_headlines_simple,
    score_headline_quality
)
from src.utils import get_country, get_intensity_label


def extract_headline(url, actor=None, impact_score=None):
    """
    Extract headline from URL.
    Wrapper for backward compatibility - delegates to headline_utils.
    """
    from src.headline_utils import extract_headline_from_url, clean_headline
    
    extracted = extract_headline_from_url(url)
    if extracted:
        return clean_headline(extracted)
    return None


def process_df(df):
    """Process raw database results into display-ready format."""
    if df.empty:
        return df

    df = df.copy()
    df.columns = [c.upper() for c in df.columns]

    # Extract and clean headlines
    headlines = []
    quality_scores = []
    
    for _, row in df.iterrows():
        db_headline = row.get('HEADLINE')
        url = row.get('NEWS_LINK', '')
        impact = row.get('IMPACT_SCORE')
        
        headline = get_best_headline(db_headline, url, impact)
        headlines.append(headline)
        quality_scores.append(score_headline_quality(headline, url) if headline else 0)

    df['HEADLINE'] = headlines
    df['_QUALITY'] = quality_scores
    
    # Filter out rows without valid headlines
    df = df[df['HEADLINE'].notna()]
    
    if df.empty:
        return df

    # Sort by quality score (best headlines first)
    df = df.sort_values('_QUALITY', ascending=False)

    # Deduplicate headlines
    keep_indices = dedupe_headlines_simple(df['HEADLINE'].tolist())
    df = df.iloc[keep_indices].copy()

    # Add region from country code
    df['REGION'] = df['ACTOR_COUNTRY_CODE'].apply(
        lambda x: get_country(x) or x if x else 'Global'
    )

    # Format date
    try:
        df['DATE_FMT'] = pd.to_datetime(
            df['DATE'].astype(str), format='%Y%m%d'
        ).dt.strftime('%d/%m')
    except Exception:
        df['DATE_FMT'] = df['DATE']

    # Add tone indicator
    df['TONE'] = df['IMPACT_SCORE'].apply(
        lambda x: "🔴" if x and x < -4 else (
            "🟡" if x and x < -1 else (
                "🟢" if x and x > 2 else "⚪"
            )
        )
    )

    # Add intensity label
    df['INTENSITY'] = df['IMPACT_SCORE'].apply(get_intensity_label)

    # Drop internal columns
    df = df.drop(columns=['_QUALITY'], errors='ignore')

    return df