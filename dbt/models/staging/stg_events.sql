-- models/staging/stg_events.sql
-- Staging model: one clean row per GDELT event, only the columns the marts use.
--
-- Incremental on purpose. The raw table is 30M+ rows and a full dedup window
-- over it, recomputed by every mart, used up MotherDuck's daily compute on the
-- free plan. The first build dedups everything once; after that each run only
-- re-reads the last 7 days (the ingest re-reads a 24h window and the embedding
-- job backfills recent events) and replaces those event ids.

{{ config(
    materialized='incremental',
    unique_key='event_id',
    incremental_strategy='delete+insert',
    description='Cleaned, deduplicated GDELT events from the raw ingestion table'
) }}

WITH source AS (
    SELECT * FROM {{ source('gdelt', 'events_dagster') }}
    WHERE EVENT_ID IS NOT NULL
    {% if is_incremental() %}
      AND DATE >= strftime(current_date - INTERVAL 7 DAY, '%Y%m%d')
    {% endif %}
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

        -- Derived: Sentiment Category
        CASE
            WHEN IMPACT_SCORE < -5 THEN 'Very Negative'
            WHEN IMPACT_SCORE < -2 THEN 'Negative'
            WHEN IMPACT_SCORE < 2 THEN 'Neutral'
            WHEN IMPACT_SCORE < 5 THEN 'Positive'
            ELSE 'Very Positive'
        END AS sentiment_category,

        -- Coverage flags (headline text and the 1024-dim embedding stay in the
        -- raw table; the marts only need to know whether they exist)
        HEADLINE IS NOT NULL AS has_headline,
        EMBEDDING IS NOT NULL AS has_embedding

    FROM source
    -- the raw table holds ~1.5M repeated EVENT_IDs (Nov/Dec 2025 backfill
    -- overlap, plus overlapping ingest runs in mid 2026); keep one row per
    -- event, preferring the copy with a headline
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY EVENT_ID
        ORDER BY (HEADLINE IS NULL), ARTICLE_COUNT DESC
    ) = 1
)

SELECT * FROM cleaned
