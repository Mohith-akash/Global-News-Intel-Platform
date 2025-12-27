-- models/staging/stg_events.sql
-- Staging model: Clean raw GDELT events
-- Materialized as a view for real-time access to source data

{{ config(
    materialized='view',
    description='Cleaned and standardized GDELT events from raw ingestion table'
) }}

WITH source AS (
    SELECT * FROM {{ source('gdelt', 'events_dagster') }}
),

cleaned AS (
    SELECT
        -- Primary Key
        EVENT_ID AS event_id,
        
        -- Temporal
        CAST(DATE AS VARCHAR) AS event_date_raw,
        CAST(SUBSTR(CAST(DATE AS VARCHAR), 1, 4) AS INTEGER) AS event_year,
        CAST(SUBSTR(CAST(DATE AS VARCHAR), 5, 2) AS INTEGER) AS event_month,
        CAST(SUBSTR(CAST(DATE AS VARCHAR), 7, 2) AS INTEGER) AS event_day,
        
        -- Actors
        MAIN_ACTOR AS actor_name,
        ACTOR_COUNTRY_CODE AS actor_country_code,
        
        -- Event Details
        EVENT_CATEGORY_CODE AS event_category_code,
        
        -- Metrics
        IMPACT_SCORE AS goldstein_scale,
        ARTICLE_COUNT AS article_count,
        SENTIMENT_SCORE AS avg_tone,
        
        -- Content
        NEWS_LINK AS source_url,
        HEADLINE AS headline,
        
        -- Embeddings (for RAG)
        EMBEDDING AS embedding,
        
        -- Derived: Sentiment Category
        CASE 
            WHEN IMPACT_SCORE < -5 THEN 'Very Negative'
            WHEN IMPACT_SCORE < -2 THEN 'Negative'
            WHEN IMPACT_SCORE < 2 THEN 'Neutral'
            WHEN IMPACT_SCORE < 5 THEN 'Positive'
            ELSE 'Very Positive'
        END AS sentiment_category,
        
        -- Derived: Has Headline flag
        CASE WHEN HEADLINE IS NOT NULL THEN TRUE ELSE FALSE END AS has_headline,
        
        -- Derived: Has Embedding flag
        CASE WHEN EMBEDDING IS NOT NULL THEN TRUE ELSE FALSE END AS has_embedding
        
    FROM source
    WHERE EVENT_ID IS NOT NULL
)

SELECT * FROM cleaned
