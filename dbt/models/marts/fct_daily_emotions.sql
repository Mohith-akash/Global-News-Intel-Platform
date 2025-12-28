-- models/marts/fct_daily_emotions.sql
-- Daily aggregation of global emotions from news

{{
    config(
        materialized='table'
    )
}}

WITH daily_emotions AS (
    SELECT
        SUBSTR(date_str, 1, 8) as date_key,
        AVG(avg_tone) as avg_mood,
        AVG(positive_score) as avg_positive,
        AVG(negative_score) as avg_negative,
        -- Core emotions
        AVG(fear) as avg_fear,
        AVG(anger) as avg_anger,
        AVG(sadness) as avg_sadness,
        AVG(joy) as avg_joy,
        AVG(trust) as avg_trust,
        AVG(anticipation) as avg_anticipation,
        -- Additional emotions
        AVG(anxiety) as avg_anxiety,
        AVG(hostility) as avg_hostility,
        AVG(depression) as avg_depression,
        COUNT(*) as article_count
    FROM {{ ref('stg_gkg_emotions') }}
    GROUP BY SUBSTR(date_str, 1, 8)
)

SELECT
    date_key,
    -- Convert YYYYMMDD to proper date
    CAST(SUBSTR(date_key, 1, 4) || '-' || SUBSTR(date_key, 5, 2) || '-' || SUBSTR(date_key, 7, 2) AS DATE) as event_date,
    ROUND(avg_mood, 2) as avg_mood,
    ROUND(avg_positive, 2) as avg_positive,
    ROUND(avg_negative, 2) as avg_negative,
    ROUND(avg_fear, 2) as avg_fear,
    ROUND(avg_anger, 2) as avg_anger,
    ROUND(avg_sadness, 2) as avg_sadness,
    ROUND(avg_joy, 2) as avg_joy,
    ROUND(avg_trust, 2) as avg_trust,
    ROUND(avg_anticipation, 2) as avg_anticipation,
    ROUND(avg_anxiety, 2) as avg_anxiety,
    ROUND(avg_hostility, 2) as avg_hostility,
    ROUND(avg_depression, 2) as avg_depression,
    article_count
FROM daily_emotions
ORDER BY date_key DESC
