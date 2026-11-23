-- models/staging/stg_gkg_emotions.sql
-- Staging model for GKG emotions data

{{
    config(
        materialized='view'
    )
}}

SELECT
    GKG_ID as gkg_id,
    CAST(DATE AS VARCHAR) as date_str,
    SOURCE as source_name,
    PERSONS as persons,
    ORGS as organizations,
    TOP_THEMES as themes,
    AVG_TONE as avg_tone,
    POSITIVE_SCORE as positive_score,
    NEGATIVE_SCORE as negative_score,
    EMOTION_FEAR as fear,
    EMOTION_ANGER as anger,
    EMOTION_SADNESS as sadness,
    EMOTION_JOY as joy,
    EMOTION_DISGUST as disgust,
    EMOTION_SURPRISE as surprise,
    EMOTION_CONFIDENCE as confidence,
    EMOTION_ANTICIPATION as anticipation,
    EMOTION_ANXIETY as anxiety,
    EMOTION_HOSTILITY as hostility,
    EMOTION_DEPRESSION as depression,

    -- Derived: bucket tone into human-readable sentiment category
    CASE
        WHEN AVG_TONE < -2 THEN 'negative'
        WHEN AVG_TONE > 2  THEN 'positive'
        ELSE 'neutral'
    END AS sentiment_category,

    -- Derived: strongest of eight WordNet Affect emotions per article, NULL when
    -- the article has none of their words (checked first, or it would read as fear)
    CASE GREATEST(
            EMOTION_FEAR, EMOTION_ANGER, EMOTION_SADNESS, EMOTION_JOY,
            EMOTION_DISGUST, EMOTION_SURPRISE, EMOTION_CONFIDENCE, EMOTION_ANTICIPATION
        )
        WHEN 0 THEN NULL
        WHEN EMOTION_FEAR         THEN 'fear'
        WHEN EMOTION_ANGER        THEN 'anger'
        WHEN EMOTION_SADNESS      THEN 'sadness'
        WHEN EMOTION_JOY          THEN 'joy'
        WHEN EMOTION_DISGUST      THEN 'disgust'
        WHEN EMOTION_SURPRISE     THEN 'surprise'
        WHEN EMOTION_CONFIDENCE   THEN 'confidence'
        WHEN EMOTION_ANTICIPATION THEN 'anticipation'
        ELSE NULL
    END AS dominant_emotion
FROM {{ source('gdelt', 'gkg_emotions') }}
-- broken records split by line breaks in the extras field have no DATE and
-- a fragment for an id; the ingest filters them and the old ones were deleted
-- on 2026-10-17, this stays as a guard
WHERE DATE IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY GKG_ID ORDER BY DATE DESC) = 1
