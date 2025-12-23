"""
Utility functions for GDELT platform.
"""
import datetime
import re
import pycountry
try:
    from src.config import COUNTRY_ALIASES
except ImportError:
    from config import COUNTRY_ALIASES


# STOPWORDS - common words that should NOT be matched to countries
STOPWORDS = {
    # Common English words
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'up', 'down', 'out', 'off',
    
    # Question words
    'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
    
    # Common verbs and nouns
    'show', 'get', 'give', 'go', 'come', 'make', 'take', 'see', 'know', 'think',
    'want', 'tell', 'find', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call',
    'events', 'event', 'news', 'data', 'results', 'result', 'query', 'search',
    'list', 'all', 'any', 'some', 'many', 'much', 'more', 'most', 'other',
    
    # Time-related words
    'today', 'yesterday', 'tomorrow', 'week', 'month', 'year', 'day', 'time',
    'now', 'then', 'before', 'after', 'during', 'while', 'since', 'until',
    'recent', 'latest', 'last', 'next', 'first', 'second', 'past', 'current',
    
    # Month names (should not match countries)
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september',
    'october', 'november', 'december',
    
    # Numbers and ordinals
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    
    # Direction words (not countries)
    'east', 'west', 'north', 'south', 'eastern', 'western', 'northern', 'southern',
    'middle', 'central',
    
    # Query-specific words
    'top', 'bottom', 'best', 'worst', 'major', 'minor', 'big', 'small', 'large',
    'high', 'low', 'important', 'significant', 'trending', 'popular', 'crisis',
    'severe', 'critical', 'level', 'count', 'total', 'number', 'amount',
    'countries', 'country', 'region', 'regions', 'global', 'world', 'worldwide',
    'international', 'local', 'national', 'foreign', 'domestic',
    
    # Common adjectives
    'new', 'old', 'good', 'bad', 'great', 'little', 'own', 'same', 'different',
    'right', 'wrong', 'long', 'short', 'early', 'late',
    
    # Articles and pronouns
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
    
    # Conjunctions and prepositions
    'about', 'above', 'across', 'against', 'along', 'among', 'around', 'because',
    'between', 'beyond', 'during', 'except', 'inside', 'into', 'like', 'near',
    'over', 'through', 'toward', 'under', 'upon', 'within', 'without',
    
    # Misc words that shouldn't match
    'only', 'just', 'also', 'very', 'even', 'still', 'already', 'always', 'never',
    'often', 'sometimes', 'usually', 'really', 'actually', 'probably', 'maybe',
    'please', 'thanks', 'thank', 'help', 'hello', 'hi', 'hey',
}


def get_country_code(name):
    """Convert country name to ISO 3-letter code."""
    if not name or not isinstance(name, str):
        return None
    
    name_lower = name.lower().strip()
    
    # Skip if too short
    if len(name_lower) < 2:
        return None
    
    # Skip stopwords
    if name_lower in STOPWORDS:
        return None
    
    # Check aliases first
    if name_lower in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[name_lower]
    
    # Skip if it's a number
    if name_lower.isdigit():
        return None
    
    # Try pycountry fuzzy search
    try:
        result = pycountry.countries.search_fuzzy(name)
        if result:
            # Extra validation: fuzzy match should be reasonably close
            matched_name = result[0].name.lower()
            # Only accept if the input is at least 3 chars and somewhat matches
            if len(name_lower) >= 3 and (
                name_lower in matched_name or 
                matched_name.startswith(name_lower[:3]) or
                name_lower.startswith(matched_name[:3])
            ):
                return result[0].alpha_3
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
    
    # Check for specific date patterns
    now = datetime.datetime.now()
    
    # Month name mapping (must be checked in order to avoid partial matches)
    month_patterns = [
        ('january', 1), ('february', 2), ('march', 3), ('april', 4),
        ('may', 5), ('june', 6), ('july', 7), ('august', 8),
        ('september', 9), ('october', 10), ('november', 11), ('december', 12),
        ('jan', 1), ('feb', 2), ('mar', 3), ('apr', 4),
        ('jun', 6), ('jul', 7), ('aug', 8), ('sep', 9),
        ('oct', 10), ('nov', 11), ('dec', 12)
    ]
    
    # Try to find a date pattern
    for month_name, month_num in month_patterns:
        # Pattern: month day (e.g., "oct 30", "october 15")
        pattern = rf'\b{month_name}\s+(\d{{1,2}})\b'
        match = re.search(pattern, prompt_lower)
        if match:
            day = int(match.group(1))
            if 1 <= day <= 31:
                try:
                    date_obj = datetime.datetime(now.year, month_num, day)
                    result['is_specific_date'] = True
                    result['specific_date'] = date_obj.strftime('%Y%m%d')
                    result['time_period'] = 'specific'
                    result['period_label'] = date_obj.strftime('%B %d, %Y')
                    return result
                except ValueError:
                    pass
        
        # Pattern: day month (e.g., "30 oct", "15 october")
        pattern2 = rf'\b(\d{{1,2}})\s+{month_name}\b'
        match2 = re.search(pattern2, prompt_lower)
        if match2:
            day = int(match2.group(1))
            if 1 <= day <= 31:
                try:
                    date_obj = datetime.datetime(now.year, month_num, day)
                    result['is_specific_date'] = True
                    result['specific_date'] = date_obj.strftime('%Y%m%d')
                    result['time_period'] = 'specific'
                    result['period_label'] = date_obj.strftime('%B %d, %Y')
                    return result
                except ValueError:
                    pass
    
    # Check for month-only queries (e.g., "events in october", "october events")
    for month_name, month_num in month_patterns:
        # Pattern: just the month name (e.g., "in october", "october")
        if month_name in prompt_lower:
            # Make sure it's not already matched as a specific date
            if not result['is_specific_date']:
                try:
                    # Create date range for the entire month
                    import calendar
                    year = now.year
                    # If the month is in the future, assume last year
                    if month_num > now.month:
                        year = now.year - 1
                    first_day = datetime.datetime(year, month_num, 1)
                    last_day_num = calendar.monthrange(year, month_num)[1]
                    last_day = datetime.datetime(year, month_num, last_day_num)
                    
                    result['is_month_range'] = True
                    result['month_start'] = first_day.strftime('%Y%m%d')
                    result['month_end'] = last_day.strftime('%Y%m%d')
                    result['time_period'] = 'month_range'
                    result['period_label'] = first_day.strftime('%B %Y')
                    return result
                except ValueError:
                    pass
    
    return result


# Regional codes mapping
REGIONAL_CODES = {
    'AFR': 'Africa', 'EUR': 'Europe', 'SAS': 'South Asia',
    'EAS': 'East Asia', 'MDE': 'Middle East', 'OCE': 'Oceania',
    'NAF': 'North Africa', 'WAF': 'West Africa', 'SAF': 'Southern Africa',
    'EAF': 'East Africa', 'CAF': 'Central Africa', 'NAM': 'North America',
    'SAM': 'South America', 'CAM': 'Central America', 'CAR': 'Caribbean',
    'SEA': 'Southeast Asia', 'CAS': 'Central Asia', 'WEU': 'Western Europe',
    'EEU': 'Eastern Europe', 'NEU': 'Northern Europe', 'SEU': 'Southern Europe'
}


def get_country(code):
    """Convert country code to full name."""
    if not code or not isinstance(code, str): 
        return None
    
    code = code.strip().upper()
    if len(code) < 2: 
        return None
    
    # Check regional codes first
    if code in REGIONAL_CODES:
        return REGIONAL_CODES[code]
    
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