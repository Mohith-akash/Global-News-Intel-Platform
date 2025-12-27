-- models/marts/dim_actors.sql
-- Dimension table: Actor profiles with aggregated metrics

{{ config(
    materialized='table',
    description='Actor dimension with aggregated statistics'
) }}

WITH events AS (
    SELECT * FROM {{ ref('stg_events') }}
    WHERE actor_name IS NOT NULL
),

actor_stats AS (
    SELECT
        actor_name,
        actor_country_code,
        
        -- Activity metrics
        COUNT(*) AS total_events,
        COUNT(DISTINCT event_date_raw) AS active_days,
        MIN(event_date_raw) AS first_seen_date,
        MAX(event_date_raw) AS last_seen_date,
        
        -- Article reach
        SUM(article_count) AS total_articles,
        AVG(article_count) AS avg_articles_per_event,
        MAX(article_count) AS max_articles_single_event,
        
        -- Sentiment profile
        AVG(goldstein_scale) AS avg_goldstein_scale,
        AVG(avg_tone) AS avg_tone,
        
        -- Most common sentiment
        MODE() WITHIN GROUP (ORDER BY sentiment_category) AS typical_sentiment
        
    FROM events
    GROUP BY actor_name, actor_country_code
),

ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY total_events DESC) AS event_rank,
        ROW_NUMBER() OVER (ORDER BY total_articles DESC) AS article_rank
    FROM actor_stats
)

SELECT * FROM ranked
