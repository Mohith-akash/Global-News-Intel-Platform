-- models/staging/stg_gkg_emotions.sql
-- Staging model for GKG emotions data

{{
    config(
        materialized='view'
    )
}}

SELECT
    GKG_ID as gkg_id,
    DATE as date_str,
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
    EMOTION_TRUST as trust,
    EMOTION_ANTICIPATION as anticipation,
    EMOTION_ANXIETY as anxiety,
    EMOTION_HOSTILITY as hostility,
    EMOTION_DEPRESSION as depression
FROM gkg_emotions
