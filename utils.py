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
    return {
        'now': now,
        'week_ago': week_ago,
        'month_ago': month_ago
    }


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
