-- models/marts/fct_trending_themes.sql  
-- Trending themes aggregated from GKG data

{{
    config(
        materialized='table'
    )
}}

WITH theme_exploded AS (
    SELECT 
        SUBSTR(date_str, 1, 8) as date_key,
        TRIM(theme.value) as theme
    FROM {{ ref('stg_gkg_emotions') }},
    LATERAL UNNEST(STRING_SPLIT(themes, ',')) AS theme(value)
    WHERE themes IS NOT NULL AND themes != ''
),

theme_counts AS (
    SELECT
        theme,
        COUNT(*) as mention_count,
        COUNT(DISTINCT date_key) as days_active
    FROM theme_exploded
    WHERE theme IS NOT NULL AND LENGTH(theme) > 2
    GROUP BY theme
    HAVING COUNT(*) >= 10  -- Filter noise
)

SELECT
    theme,
    mention_count,
    days_active,
    ROUND(mention_count * 1.0 / days_active, 1) as avg_daily_mentions
FROM theme_counts
ORDER BY mention_count DESC
LIMIT 100
