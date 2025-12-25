"""
Configuration for GDELT platform.
"""

CEREBRAS_MODEL = "llama3.1-8b"

# RAG Configuration (Voyage AI)
VOYAGE_MODEL = "voyage-3.5-lite"  # 200M free tokens
EMBEDDING_DIMENSIONS = 1024  # voyage-3.5-lite output dimensions
RAG_TOP_K = 10  # Number of similar headlines to retrieve
RAG_BACKFILL_DAYS = 14  # Number of days to backfill embeddings

COUNTRY_ALIASES = {
    # United States
    'usa': 'USA', 'us': 'USA', 'america': 'USA', 'united states': 'USA', 
    'u.s.': 'USA', 'u.s.a.': 'USA', 'american': 'USA', 'americans': 'USA',
    # United Kingdom  
    'uk': 'GBR', 'britain': 'GBR', 'british': 'GBR', 'england': 'GBR',
    'great britain': 'GBR', 'united kingdom': 'GBR',
    # Russia
    'russia': 'RUS', 'russian': 'RUS', 'russians': 'RUS',
    # South Korea (most common reference)
    'korea': 'KOR', 'korean': 'KOR', 'south korea': 'KOR',
    # North Korea
    'north korea': 'PRK', 'dprk': 'PRK',
    # Iran
    'iran': 'IRN', 'iranian': 'IRN', 'persia': 'IRN',
    # China
    'china': 'CHN', 'chinese': 'CHN', 'prc': 'CHN',
    # Common others
    'uae': 'ARE', 'emirates': 'ARE',
    'saudi': 'SAU', 'saudi arabia': 'SAU',
    'germany': 'DEU', 'german': 'DEU',
    'france': 'FRA', 'french': 'FRA',
    'japan': 'JPN', 'japanese': 'JPN',
    'india': 'IND', 'indian': 'IND',
    'brazil': 'BRA', 'brazilian': 'BRA',
    'canada': 'CAN', 'canadian': 'CAN',
    'australia': 'AUS', 'australian': 'AUS',
    'mexico': 'MEX', 'mexican': 'MEX',
    'israel': 'ISR', 'israeli': 'ISR',
    'palestine': 'PSE', 'palestinian': 'PSE',
    'ukraine': 'UKR', 'ukrainian': 'UKR',
    'taiwan': 'TWN', 'taiwanese': 'TWN',
    # Other regions mapped to prominent countries
    'syria': 'SYR', 'syrian': 'SYR',
    'iraq': 'IRQ', 'iraqi': 'IRQ',
    'lebanon': 'LBN', 'lebanese': 'LBN',
    'jordan': 'JOR', 'jordanian': 'JOR',
    'egypt': 'EGY', 'egyptian': 'EGY',
    'turkey': 'TUR', 'turkish': 'TUR',
    'greece': 'GRC', 'greek': 'GRC',
    'poland': 'POL', 'polish': 'POL',
    'spain': 'ESP', 'spanish': 'ESP',
    'italy': 'ITA', 'italian': 'ITA',
    'netherlands': 'NLD', 'dutch': 'NLD', 'holland': 'NLD',
    'belgium': 'BEL', 'belgian': 'BEL',
    'sweden': 'SWE', 'swedish': 'SWE',
    'norway': 'NOR', 'norwegian': 'NOR',
    'finland': 'FIN', 'finnish': 'FIN',
    'denmark': 'DNK', 'danish': 'DNK',
    'switzerland': 'CHE', 'swiss': 'CHE',
    'austria': 'AUT', 'austrian': 'AUT',
    'portugal': 'PRT', 'portuguese': 'PRT',
    'argentina': 'ARG', 'argentine': 'ARG', 'argentinian': 'ARG',
    'colombia': 'COL', 'colombian': 'COL',
    'chile': 'CHL', 'chilean': 'CHL',
    'peru': 'PER', 'peruvian': 'PER',
    'venezuela': 'VEN', 'venezuelan': 'VEN',
    'cuba': 'CUB', 'cuban': 'CUB',
    'philippines': 'PHL', 'filipino': 'PHL',
    'indonesia': 'IDN', 'indonesian': 'IDN',
    'malaysia': 'MYS', 'malaysian': 'MYS',
    'singapore': 'SGP', 'singaporean': 'SGP',
    'thailand': 'THA', 'thai': 'THA',
    'vietnam': 'VNM', 'vietnamese': 'VNM',
    'pakistan': 'PAK', 'pakistani': 'PAK',
    'bangladesh': 'BGD', 'bangladeshi': 'BGD',
    'nigeria': 'NGA', 'nigerian': 'NGA',
    'kenya': 'KEN', 'kenyan': 'KEN',
    'south africa': 'ZAF',
    'morocco': 'MAR', 'moroccan': 'MAR',
    'algeria': 'DZA', 'algerian': 'DZA',
    'ethiopia': 'ETH', 'ethiopian': 'ETH',
    'ghana': 'GHA', 'ghanaian': 'GHA',
}

REQUIRED_ENVS = ["MOTHERDUCK_TOKEN", "CEREBRAS_API_KEY"]
# Region to country codes mapping (for queries like "Middle East")
REGION_ALIASES = {
    'middle east': ['ISR', 'SAU', 'IRN', 'IRQ', 'SYR', 'JOR', 'LBN', 'ARE', 'KWT', 'QAT', 'BHR', 'OMN', 'YEM', 'PSE'],
    'mideast': ['ISR', 'SAU', 'IRN', 'IRQ', 'SYR', 'JOR', 'LBN', 'ARE', 'KWT', 'QAT', 'BHR', 'OMN', 'YEM', 'PSE'],
    'gulf': ['SAU', 'ARE', 'KWT', 'QAT', 'BHR', 'OMN'],
    'europe': ['DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'NLD', 'BEL', 'POL', 'SWE', 'NOR', 'DNK', 'FIN', 'AUT', 'CHE', 'GRC', 'PRT'],
    'asia': ['CHN', 'JPN', 'KOR', 'IND', 'IDN', 'THA', 'VNM', 'MYS', 'SGP', 'PHL', 'PAK', 'BGD'],
    'africa': ['NGA', 'ZAF', 'EGY', 'KEN', 'ETH', 'GHA', 'MAR', 'DZA'],
}
