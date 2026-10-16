-- Per-user pre/post metrics from the public Stack Overflow dataset.
-- Parameter: @t0 DATE  (experiment start; choose it with data/verify.sql)
--
--   pre-period  : [t0 - 56 days, t0)      -> covariates
--   post-period : [t0, t0 + 28 days)      -> outcomes
--   population  : accounts created before the pre-period that posted an answer
--                 or comment during the pre-period
--
-- LEAKAGE RULES (see DECISION.md, D-010):
--   * Never read snapshot columns (users.reputation, up_votes, posts.score,
--     accepted_answer_id status "as of today", ...) as covariates.
--   * An answer counts as accepted in a window only if its acceptance vote is
--     dated inside that same window's end boundary, so pre-period covariates
--     cannot see acceptances that happened during the experiment.
--
-- Output columns match expkit/io/schema.py.

WITH
params AS (
  SELECT
    TIMESTAMP(@t0)                                  AS t0,
    TIMESTAMP(DATE_SUB(@t0, INTERVAL 56 DAY))       AS pre_start,
    TIMESTAMP(DATE_ADD(@t0, INTERVAL 28 DAY))       AS post_end
),

accept_votes AS (           -- acceptance date per answer (vote_type_id 1 = accepted)
  SELECT post_id, MIN(creation_date) AS accepted_at
  FROM `bigquery-public-data.stackoverflow.votes`, params
  WHERE vote_type_id = 1
    AND creation_date >= pre_start AND creation_date < post_end
  GROUP BY post_id
),

answers AS (
  SELECT
    a.owner_user_id AS user_id,
    IF(a.creation_date < p.t0, 'pre', 'post') AS period,
    -- accepted within the same window the answer was posted in
    CASE
      WHEN v.accepted_at IS NULL THEN 0
      WHEN a.creation_date < p.t0 AND v.accepted_at < p.t0 THEN 1
      WHEN a.creation_date >= p.t0 AND v.accepted_at < p.post_end THEN 1
      ELSE 0
    END AS accepted
  FROM `bigquery-public-data.stackoverflow.posts_answers` a
  CROSS JOIN params p
  LEFT JOIN accept_votes v ON v.post_id = a.id
  WHERE a.owner_user_id IS NOT NULL
    AND a.creation_date >= p.pre_start AND a.creation_date < p.post_end
),

comments AS (
  SELECT c.user_id, IF(c.creation_date < p.t0, 'pre', 'post') AS period
  FROM `bigquery-public-data.stackoverflow.comments` c
  CROSS JOIN params p
  WHERE c.user_id IS NOT NULL
    AND c.creation_date >= p.pre_start AND c.creation_date < p.post_end
),

answer_agg AS (
  SELECT
    user_id,
    COUNTIF(period = 'pre')                     AS pre_answers,
    COUNTIF(period = 'pre'  AND accepted = 1)   AS pre_accepted,
    COUNTIF(period = 'post')                    AS post_answers,
    COUNTIF(period = 'post' AND accepted = 1)   AS post_accepted
  FROM answers
  GROUP BY user_id
),

comment_agg AS (
  SELECT
    user_id,
    COUNTIF(period = 'pre')  AS pre_comments,
    COUNTIF(period = 'post') AS post_comments
  FROM comments
  GROUP BY user_id
),

joined AS (
  SELECT
    COALESCE(a.user_id, c.user_id)     AS user_id,
    COALESCE(a.pre_answers, 0)         AS pre_answers,
    COALESCE(a.pre_accepted, 0)        AS pre_accepted,
    COALESCE(c.pre_comments, 0)        AS pre_comments,
    COALESCE(a.post_answers, 0)        AS post_answers,
    COALESCE(a.post_accepted, 0)       AS post_accepted,
    COALESCE(c.post_comments, 0)       AS post_comments
  FROM answer_agg a
  FULL OUTER JOIN comment_agg c ON a.user_id = c.user_id
)

SELECT
  j.user_id,
  DATE_DIFF(DATE(p.t0), DATE(u.creation_date), DAY)                AS tenure_days,
  j.pre_answers, j.pre_accepted, j.pre_comments,
  j.post_answers, j.post_accepted, j.post_comments,
  1                                                                AS pre_active,
  IF(j.post_answers + j.post_comments > 0, 1, 0)                   AS post_active
FROM joined j
CROSS JOIN params p
JOIN `bigquery-public-data.stackoverflow.users` u ON u.id = j.user_id
WHERE u.id > 0                                      -- drop the "Community" bot account (id -1)
  AND u.creation_date < p.pre_start                 -- full pre-period exposure
  AND j.pre_answers + j.pre_comments > 0            -- active in the pre-period
