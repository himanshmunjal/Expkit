"""Profile the per-user metrics table (step 2 of the pipeline).

    python data/prepare.py --data data/users_metrics.parquet --out results/profile.json

Writes the distribution facts that justify later choices: zero share, tail
quantiles, pre/post correlation (= the CUPED ceiling rho^2), and the ratio of
accepted answers. Run it on the real extract before any simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from expkit.io import source_from_uri

PAIRS = [("pre_answers", "post_answers"), ("pre_comments", "post_comments")]


def profile(df) -> dict:
    out: dict = {"n_users": int(len(df)), "columns": {}}
    for col in df.columns.drop("user_id"):
        s = df[col].astype(float)
        out["columns"][col] = {
            "mean": s.mean(), "sd": s.std(), "zero_share": float((s == 0).mean()),
            **{f"q{int(q * 1000) / 10:g}": float(s.quantile(q)) for q in (0.5, 0.9, 0.99, 0.999)},
            "max": s.max(),
        }
    out["pre_post_correlation"] = {
        f"{a}->{b}": float(np.corrcoef(df[a], df[b])[0, 1]) for a, b in PAIRS
    }
    out["cuped_ceiling_rho2"] = {k: v**2 for k, v in out["pre_post_correlation"].items()}
    out["accepted_answer_rate_post"] = float(df.post_accepted.sum() / max(df.post_answers.sum(), 1))
    return json.loads(json.dumps(out, default=float))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/users_metrics.parquet")
    p.add_argument("--out", default="results/profile.json")
    a = p.parse_args()
    result = profile(source_from_uri(a.data).load())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["pre_post_correlation"], indent=2))


if __name__ == "__main__":
    main()
