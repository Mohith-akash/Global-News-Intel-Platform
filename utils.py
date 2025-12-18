"""
Utility functions for GDELT platform.
"""
import datetime
import re
import pycountry
from config import COUNTRY_ALIASES


def get_country_code(name):
    """Convert country name to ISO 3-letter code."""
    if not name or not isinstance(name, str):
        return None
    
    name_lower = name.lower().strip()
    
    # Skip common English words that might match countries
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
        'now', 'here', 'there', 'then', 'once', 'many', 'any', 'show', 'get',
        'events', 'event', 'total', 'count', 'many', 'much', 'last', 'next',
        'week', 'month', 'year', 'today', 'yesterday', 'happened', 'involving',
        'about', 'between', 'during', 'before', 'after', 'crisis', 'severe',
        'top', 'countries', 'country',
        # Geographic directions (prevent "Middle East", "North Africa", etc. issues)
        'east', 'west', 'north', 'south', 'middle', 'central', 'eastern', 
        'western', 'northern', 'southern', 'asia', 'africa', 'europe',
        'happening', 'going', 'news', 'latest', 'recent', 'new', 'old'
    }
    
    if name_lower in STOPWORDS:
        return None
    
    # Check aliases first (includes common names like 'usa', 'uk', etc.)
    if name_lower in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[name_lower]
    
    # Explicit check for USA variants (backup in case alias fails)
    if name_lower in ('usa', 'us', 'america', 'united states', 'u.s.', 'u.s.a.'):
        return 'USA'
    
    # Skip very short words (likely not country names)
    if len(name_lower) < 3:
        return None
    
    try:
        result = pycountry.countries.search_fuzzy(name_lower)
        if result:
            code = result[0].alpha_3
            # Prevent UMI (US Minor Outlying Islands) from matching "usa" queries
            if code == 'UMI' and 'minor' not in name_lower and 'outlying' not in name_lower:
                return 'USA'
            return code
    except LookupError:
        pass
    
    return None


def get_dates():
    """Get date ranges for queries."""
    now = datetime.datetime.now()
    return {
        'now': now,
        'today': now.strftime('%Y%m%d'),
        'week_ago': (now - datetime.timedelta(days=7)).strftime('%Y%m%d'),
        'month_ago': (now - datetime.timedelta(days=30)).strftime('%Y%m%d'),
        'three_months_ago': (now - datetime.timedelta(days=90)).strftime('%Y%m%d')
    }


def detect_query_type(prompt):
    """Detect query type and time period from user prompt."""
    prompt_lower = prompt.lower()
    result = {
        'is_aggregate': False,
        'is_specific_date': False,
        'specific_date': None,
        'time_period': 'week',
        'period_label': 'the past week'
    }
    
    aggregate_keywords = ['count', 'total', 'how many', 'number of', 'sum', 'overall', 
                          'whole year', 'all time', 'entire', 'so far', 'all events']
    if any(kw in prompt_lower for kw in aggregate_keywords):
        result['is_aggregate'] = True
        result['time_period'] = 'all'
        result['period_label'] = 'all available data'
    
    if 'year' in prompt_lower or 'all data' in prompt_lower:
        result['time_period'] = 'all'
        result['period_label'] = 'all available data'
    elif 'month' in prompt_lower or '30 day' in prompt_lower:
        result['time_period'] = 'month'
        result['period_label'] = 'the past month'
    elif 'today' in prompt_lower:
        result['time_period'] = 'day'
        result['period_label'] = 'today'
    
    now = datetime.datetime.now()
    month_names = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                   'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
                   'january': 1, 'february': 2, 'march': 3, 'april': 4, 'june': 6,
                   'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}
    
    for month_name, month_num in month_names.items():
        pattern = rf'{month_name}\s+(\d{{1,2}})'
        match = re.search(pattern, prompt_lower)
        if match:
            day = int(match.group(1))
            try:
                date_obj = datetime.datetime(now.year, month_num, day)
                result['is_specific_date'] = True
                result['specific_date'] = date_obj.strftime('%Y%m%d')
                result['time_period'] = 'specific'
                result['period_label'] = date_obj.strftime('%B %d, %Y')
                return result
            except ValueError:
                pass
    
    return result


def get_country(code):
    """Convert country code to full name."""
    if not code or not isinstance(code, str): 
        return None
    
    code = code.strip().upper()
    if len(code) < 2: 
        return None
    
    try:
        if len(code) == 2:
            country = pycountry.countries.get(alpha_2=code)
            if country: 
                return country.name
        
        if len(code) == 3:
            country = pycountry.countries.get(alpha_3=code)
            if country: 
                return country.name
        
        return None
    except:
        return None


def get_impact_label(score):
    """Convert impact score to readable label."""
    if score is None: 
        return "Neutral"
    
    score = float(score)
    
    if score <= -8: return "🔴 Severe Crisis"
    if score <= -5: return "🔴 Major Conflict"
    if score <= -3: return "🟠 Rising Tensions"
    if score <= -1: return "🟡 Minor Dispute"
    if score < 1: return "⚪ Neutral"
    if score < 3: return "🟢 Cooperation"
    if score < 5: return "🟢 Partnership"
    return "✨ Major Agreement"


def get_intensity_label(score):
    """Get intensity description for events."""
    if score is None: 
        return "⚪ Neutral Event"
    
    score = float(score)
    
    if score <= -8: return "⚔️ Armed Conflict"
    if score <= -6: return "🔴 Major Crisis"
    if score <= -4: return "🟠 Serious Tension"
    if score <= -2: return "🟡 Verbal Dispute"
    if score < 2: return "⚪ Neutral Event"
    if score < 4: return "🟢 Diplomatic Talk"
    if score < 6: return "🤝 Active Partnership"
    return "✨ Peace Agreement"