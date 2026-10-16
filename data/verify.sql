-- Day-1 verification (run BEFORE extract.sql). Answers the questions in the
-- project brief: which tables exist, how fresh each one is, and which carry a
-- user id. Pick T0 so that [T0 - 56d, T0 + 28d) ends before the oldest
-- "max_ts" among the tables used.
--
-- Cost: scans only the timestamp/id columns (a few GB). Run with
--   bq query --use_legacy_sql=false --maximum_bytes_billed=20000000000 < data/verify.sql

-- 1. Tables in the dataset
SELECT table_name
FROM `bigquery-public-data.stackoverflow.INFORMATION_SCHEMA.TABLES`
ORDER BY table_name;

-- 2. Columns of the tables we rely on (confirm user-id and timestamp columns)
SELECT table_name, column_name, data_type
FROM `bigquery-public-data.stackoverflow.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name IN ('posts_answers', 'posts_questions', 'comments', 'users', 'votes')
ORDER BY table_name, ordinal_position;

-- 3. Freshness of each table
SELECT 'posts_answers' AS tbl, MIN(creation_date) AS min_ts, MAX(creation_date) AS max_ts
FROM `bigquery-public-data.stackoverflow.posts_answers`
UNION ALL
SELECT 'posts_questions', MIN(creation_date), MAX(creation_date)
FROM `bigquery-public-data.stackoverflow.posts_questions`
UNION ALL
SELECT 'comments', MIN(creation_date), MAX(creation_date)
FROM `bigquery-public-data.stackoverflow.comments`
UNION ALL
SELECT 'users', MIN(creation_date), MAX(creation_date)
FROM `bigquery-public-data.stackoverflow.users`
UNION ALL
SELECT 'votes', MIN(creation_date), MAX(creation_date)
FROM `bigquery-public-data.stackoverflow.votes`;

-- 4. Acceptance timestamps: votes with vote_type_id = 1 ("AcceptedByOriginator")
--    have a date but NO user id. Check they exist near the candidate T0 window.
SELECT DATE_TRUNC(DATE(creation_date), MONTH) AS month, COUNT(*) AS accept_votes
FROM `bigquery-public-data.stackoverflow.votes`
WHERE vote_type_id = 1
GROUP BY month
ORDER BY month DESC
LIMIT 24;
