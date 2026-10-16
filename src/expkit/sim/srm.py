"""SRM detection simulation using the production-style hash assignment.

A bucketing bug is simulated by silently losing a fraction of treatment users
(as if a client crash or a logging filter dropped them) before counting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from expkit.assign import hash_assign
from expkit.sim._parallel import run_many
from expkit.stats.srm import DEFAULT_SRM_THRESHOLD, srm_test


def run_srm_simulation(
    user_ids: np.ndarray,
    n_users: int,
    drop_fraction: float,
    n_runs: int = 500,
    threshold: float = DEFAULT_SRM_THRESHOLD,
    seed: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    if not 0 <= drop_fraction < 1:
        raise ValueError("drop_fraction must be in [0, 1)")
    user_ids = np.asarray(user_ids)

    def one_run(run_id: int, rng: np.random.Generator) -> dict:
        ids = rng.choice(user_ids, size=n_users, replace=False)
        arms = hash_assign(ids, salt=f"srm-sim-{seed}-{run_id}")
        n_c = int(np.sum(arms == 0))
        n_t = int(rng.binomial(int(np.sum(arms == 1)), 1.0 - drop_fraction))
        res = srm_test([n_c, n_t], threshold=threshold)
        return {"run": run_id, "drop_fraction": drop_fraction, "n_control": n_c,
                "n_treatment": n_t, "p_value": res.p_value, "detected": res.mismatch}

    return pd.DataFrame(run_many(one_run, n_runs, seed, n_jobs))
