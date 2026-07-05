-- Singular test: daily emotion aggregates must be built from at least one
-- article; zero or negative counts indicate a broken aggregation.

SELECT *
FROM {{ ref('fct_daily_emotions') }}
WHERE article_count <= 0
