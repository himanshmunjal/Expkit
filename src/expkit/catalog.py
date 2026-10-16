"""Metric definitions for the Stack Overflow extract.

Each metric's covariate is the same behaviour in the pre-period, which is the
classic CUPED choice. tenure is deliberately excluded from defaults: it is
pre-experiment and safe, but adds little on these metrics; use it in the
multi-covariate extension.
"""

from __future__ import annotations

from expkit._types import MetricSpec

ANSWERS = MetricSpec("answers_posted", "mean", "post_answers", covariates=("pre_answers",))
# pre_active is 1 for everyone (population = users active in the pre-period), so
# it carries no information; pre-period volume is used instead.
ACTIVE = MetricSpec("active_28d", "binary", "post_active",
                    covariates=("pre_answers", "pre_comments"))
COMMENTS = MetricSpec("comments_posted", "mean", "post_comments", covariates=("pre_comments",))
ACCEPT_RATE = MetricSpec("accepted_answer_rate", "ratio", "post_accepted",
                         denominator="post_answers")

CATALOG: dict[str, MetricSpec] = {m.name: m for m in (ANSWERS, ACTIVE, COMMENTS, ACCEPT_RATE)}


def get_metric(name: str) -> MetricSpec:
    try:
        return CATALOG[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}; available: {sorted(CATALOG)}") from None
