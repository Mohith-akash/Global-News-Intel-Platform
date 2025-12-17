"""
Configuration for GDELT platform.
"""

CEREBRAS_MODEL = "llama3.1-8b"

COUNTRY_ALIASES = {
    'usa': 'USA', 'us': 'USA', 'america': 'USA',
    'uk': 'GBR', 'britain': 'GBR', 'british': 'GBR',
    'russia': 'RUS', 'russian': 'RUS',
    'korea': 'KOR', 'korean': 'KOR',
    'iran': 'IRN', 'iranian': 'IRN',
}

REQUIRED_ENVS = ["MOTHERDUCK_TOKEN", "CEREBRAS_API_KEY"]