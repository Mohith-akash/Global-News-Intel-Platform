"""
Configuration constants for the GDELT News Intelligence Platform.
"""

# AI Model
CEREBRAS_MODEL = "llama3.1-8b"

# Common country aliases not handled by pycountry
COUNTRY_ALIASES = {
    'usa': 'USA', 'us': 'USA', 'america': 'USA',
    'uk': 'GBR', 'britain': 'GBR', 'british': 'GBR',
    'russia': 'RUS', 'russian': 'RUS',
    'korea': 'KOR', 'korean': 'KOR',
    'iran': 'IRN', 'iranian': 'IRN',
}

# Required environment variables
REQUIRED_ENVS = ["MOTHERDUCK_TOKEN", "CEREBRAS_API_KEY"]
