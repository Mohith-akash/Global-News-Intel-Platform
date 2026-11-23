"""
Utility functions for GDELT platform.
"""
import calendar
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


AGGREGATE_WORDS = ('count', 'total', 'how many', 'number of', 'sum', 'overall',
                   'whole year', 'all time', 'entire', 'so far', 'all events')
# Full names before abbreviations: the first month found wins, and the full
# name must win in "september 5" over "sep". Abbreviated "may" is the full name.
_MONTH_NAMES = ('january february march april may june july august '
                'september october november december').split()
MONTHS = ([(m, i) for i, m in enumerate(_MONTH_NAMES, 1)]
          + [(m[:3], i) for i, m in enumerate(_MONTH_NAMES, 1) if m != 'may'])


def _week_range(now, weeks_back):
    """Monday-Sunday of the current week (0) or the one before (1)."""
    monday = now - datetime.timedelta(days=now.weekday() + 7 * weeks_back)
    sunday = monday + datetime.timedelta(days=6)
    return monday, sunday


def detect_query_type(prompt):
    """Read the time window out of a question: 'this week', 'oct 30', 'in march', ..."""
    p = prompt.lower()
    now = datetime.datetime.now()
    result = {'is_aggregate': False, 'is_specific_date': False, 'specific_date': None,
              'time_period': 'week', 'period_label': 'the past week'}

    if any(w in p for w in AGGREGATE_WORDS):
        result.update(is_aggregate=True, time_period='all', period_label='all available data')

    if 'year' in p or 'all data' in p:
        result.update(time_period='all', period_label='all available data')
    elif 'last week' in p or 'this week' in p:
        which = 'last week' if 'last week' in p else 'this week'
        monday, sunday = _week_range(now, 1 if which == 'last week' else 0)
        result.update(is_week_range=True, week_start=monday.strftime('%Y%m%d'),
                      week_end=sunday.strftime('%Y%m%d'), time_period=which.replace(' ', '_'),
                      period_label=f"{which} ({monday:%b %d} - {sunday:%b %d})")
    elif 'month' in p or '30 day' in p:
        result.update(time_period='month', period_label='the past month')
    elif 'today' in p:
        result.update(time_period='day', period_label='today')

    # A day of a month: "oct 30" or "30 october" (word boundaries keep 'mar'
    # out of 'summarize')
    for name, month in MONTHS:
        for rx in (rf'(?:^|\s){name}\s+(\d{{1,2}})(?:\s|$|[^a-z])', rf'\b(\d{{1,2}})\s+{name}\b'):
            m = re.search(rx, p)
            if not m:
                continue
            try:
                day = datetime.datetime(now.year, month, int(m.group(1)))
            except ValueError:
                continue
            result.update(is_specific_date=True, specific_date=day.strftime('%Y%m%d'),
                          time_period='specific', period_label=day.strftime('%B %d, %Y'))
            return result

    # A whole month: "events in october"; a month still ahead means last year's
    for name, month in MONTHS:
        if re.search(rf'(?:^|\s){name}(?:\s|$|[^a-z])', p):
            year = now.year - 1 if month > now.month else now.year
            first = datetime.datetime(year, month, 1)
            last = first.replace(day=calendar.monthrange(year, month)[1])
            result.update(is_month_range=True, month_start=first.strftime('%Y%m%d'),
                          month_end=last.strftime('%Y%m%d'), time_period='month_range',
                          period_label=first.strftime('%B %Y'))
            return result

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
    except (KeyError, LookupError):
        return None


# One severity scale for the feed, the chat and the guide on the Feed tab:
# (label, lowest score, highest score, upper bound is inclusive)
INTENSITY_SCALE = [
    ("🔴 Armed conflict", -10, -8, True),
    ("🔴 Major crisis", -8, -6, True),
    ("🟠 Serious tension", -6, -4, True),
    ("🟡 Verbal dispute", -4, -2, True),
    ("⚪ Neutral", -2, 2, False),
    ("🟢 Diplomatic talks", 2, 4, False),
    ("🟢 Partnership", 4, 6, False),
    ("🟢 Peace agreement", 6, 10, True),
]


def get_intensity_label(score):
    """Severity label for an impact score (-10 conflict .. +10 cooperation)."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "⚪ Neutral"
    if s != s:  # NaN
        return "⚪ Neutral"
    for label, _, high, inclusive in INTENSITY_SCALE[:-1]:
        if s < high or (inclusive and s == high):
            return label
    return INTENSITY_SCALE[-1][0]
