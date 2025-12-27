-- models/marts/dim_countries.sql
-- Dimension table: Country profiles with aggregated metrics

{{ config(
    materialized='table',
    description='Country dimension with aggregated event statistics'
) }}

WITH events AS (
    SELECT * FROM {{ ref('stg_events') }}
    WHERE actor_country_code IS NOT NULL
),

country_stats AS (
    SELECT
        actor_country_code AS country_code,
        
        -- Activity metrics
        COUNT(*) AS total_events,
        COUNT(DISTINCT actor_name) AS unique_actors,
        COUNT(DISTINCT event_date_raw) AS active_days,
        
        -- Time range
        MIN(event_date_raw) AS first_event_date,
        MAX(event_date_raw) AS last_event_date,
        
        -- Article reach
        SUM(article_count) AS total_articles,
        AVG(article_count) AS avg_articles_per_event,
        
        -- Sentiment metrics
        AVG(goldstein_scale) AS avg_goldstein_scale,
        AVG(avg_tone) AS avg_tone,
        
        -- Sentiment distribution
        SUM(CASE WHEN goldstein_scale < 0 THEN 1 ELSE 0 END) AS negative_events,
        SUM(CASE WHEN goldstein_scale >= 0 THEN 1 ELSE 0 END) AS positive_events
        
    FROM events
    GROUP BY actor_country_code
),

ranked AS (
    SELECT
        *,
        ROUND(100.0 * negative_events / NULLIF(total_events, 0), 2) AS negative_event_pct,
        ROW_NUMBER() OVER (ORDER BY total_events DESC) AS activity_rank
    FROM country_stats
)

SELECT * FROM ranked
