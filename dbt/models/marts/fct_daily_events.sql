-- models/marts/fct_daily_events.sql
-- Fact table: Daily event aggregations
-- This is the main analytics table for dashboards

{{ config(
    materialized='table',
    description='Daily aggregated event metrics for analytics dashboard'
) }}

WITH events AS (
    SELECT * FROM {{ ref('stg_events') }}
),

daily_agg AS (
    SELECT
        -- Date dimension
        event_date_raw,
        event_year,
        event_month,
        event_day,
        
        -- Counts
        COUNT(*) AS event_count,
        COUNT(DISTINCT actor_name) AS unique_actors,
        COUNT(DISTINCT actor_country_code) AS unique_countries,
        
        -- Article metrics
        SUM(article_count) AS total_articles,
        AVG(article_count) AS avg_articles_per_event,
        
        -- Sentiment metrics
        AVG(goldstein_scale) AS avg_goldstein_scale,
        AVG(avg_tone) AS avg_tone,
        
        -- Sentiment distribution
        SUM(CASE WHEN sentiment_category = 'Very Negative' THEN 1 ELSE 0 END) AS very_negative_count,
        SUM(CASE WHEN sentiment_category = 'Negative' THEN 1 ELSE 0 END) AS negative_count,
        SUM(CASE WHEN sentiment_category = 'Neutral' THEN 1 ELSE 0 END) AS neutral_count,
        SUM(CASE WHEN sentiment_category = 'Positive' THEN 1 ELSE 0 END) AS positive_count,
        SUM(CASE WHEN sentiment_category = 'Very Positive' THEN 1 ELSE 0 END) AS very_positive_count,
        
        -- Coverage metrics
        SUM(CASE WHEN has_headline THEN 1 ELSE 0 END) AS events_with_headlines,
        SUM(CASE WHEN has_embedding THEN 1 ELSE 0 END) AS events_with_embeddings
        
    FROM events
    GROUP BY 
        event_date_raw,
        event_year,
        event_month,
        event_day
)

SELECT 
    *,
    -- Derived metrics
    ROUND(100.0 * events_with_headlines / NULLIF(event_count, 0), 2) AS headline_coverage_pct,
    ROUND(100.0 * events_with_embeddings / NULLIF(event_count, 0), 2) AS embedding_coverage_pct,
    ROUND(100.0 * (positive_count + very_positive_count) / NULLIF(event_count, 0), 2) AS positive_event_pct,
    ROUND(100.0 * (negative_count + very_negative_count) / NULLIF(event_count, 0), 2) AS negative_event_pct
FROM daily_agg
ORDER BY event_date_raw DESC
