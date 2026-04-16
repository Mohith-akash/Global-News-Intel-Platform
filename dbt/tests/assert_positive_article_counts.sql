-- Singular test: assert no daily event aggregates have negative article totals.
-- dbt tests pass when they return zero rows; any offending rows here indicate
-- a data corruption or aggregation bug in fct_daily_events.

SELECT *
FROM {{ ref('fct_daily_events') }}
WHERE total_articles < 0
