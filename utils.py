"""
Utility functions for the GDELT News Intelligence Platform.
"""
import datetime
import pycountry
from config import COUNTRY_ALIASES


def get_country_code(name):
    """Convert country name to 3-letter ISO code using pycountry."""
    name_lower = name.lower().strip()
    
    # Check aliases first (faster for common cases)
    if name_lower in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[name_lower]
    
    # Try pycountry lookup
    try:
        result = pycountry.countries.search_fuzzy(name)
        if result:
            return result[0].alpha_3
    except LookupError:
        pass
    
    return None


def get_dates():
    """
    Get current date and calculated date ranges.
    Returns dict with: now, week_ago, month_ago (YYYYMMDD format strings)
    """
    now = datetime.datetime.now()
    week_ago = (now - datetime.timedelta(days=7)).strftime('%Y%m%d')
    month_ago = (now - datetime.timedelta(days=30)).strftime('%Y%m%d')
    three_months_ago = (now - datetime.timedelta(days=90)).strftime('%Y%m%d')
    return {
        'now': now,
        'today': now.strftime('%Y%m%d'),
        'week_ago': week_ago,
        'month_ago': month_ago,
        'three_months_ago': three_months_ago
    }


def detect_query_type(prompt):
    """
    Detect what type of query the user is asking and what time period.
    
    Returns dict with:
        - is_aggregate: True if asking for counts/totals (use all data)
        - is_specific_date: True if asking about a specific date
        - specific_date: The date in YYYYMMDD format if detected
        - time_period: 'all', 'month', 'week', or 'day'
        - period_label: Human-readable label for the AI summary
    """
    import re
    
    prompt_lower = prompt.lower()
    result = {
        'is_aggregate': False,
        'is_specific_date': False,
        'specific_date': None,
        'time_period': 'week',  # Default to week
        'period_label': 'the past week'
    }
    
    # Detect aggregate queries (counts, totals, "how many")
    aggregate_keywords = ['count', 'total', 'how many', 'number of', 'sum', 'overall', 
                          'whole year', 'all time', 'entire', 'so far', 'all events']
    if any(kw in prompt_lower for kw in aggregate_keywords):
        result['is_aggregate'] = True
        result['time_period'] = 'all'
        result['period_label'] = 'all available data (past 3+ months)'
    
    # Detect time periods
    if 'year' in prompt_lower or 'all data' in prompt_lower:
        result['time_period'] = 'all'
        result['period_label'] = 'all available data (past 3+ months)'
    elif 'month' in prompt_lower or '30 day' in prompt_lower:
        result['time_period'] = 'month'
        result['period_label'] = 'the past month'
    elif 'today' in prompt_lower:
        result['time_period'] = 'day'
        result['period_label'] = 'today'
    
    # Detect specific dates (e.g., "October 15", "10/15", "2024-10-15", "dec 5")
    now = datetime.datetime.now()
    
    # Pattern: Month name + day (e.g., "october 15", "dec 5")
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
                # Assume current year
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
    """Convert country codes to full country names."""
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
    """Convert numeric impact scores to human-readable labels."""
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
    """Get detailed intensity description for events."""
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
