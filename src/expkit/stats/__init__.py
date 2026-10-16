"""Statistical methods. Importing this package registers the built-in methods."""

from expkit.stats import methods as _methods  # noqa: F401  (registers built-ins)
from expkit.stats.bayes import BayesResult, beta_binomial
from expkit.stats.bootstrap import bootstrap_mean_diff, bootstrap_ratio_diff
from expkit.stats.cuped import add_zero_indicator, cuped_adjust, cuped_theta, cuped_ttest
from expkit.stats.delta import (
    delta_ratio_test,
    jackknife_ratio_and_variance,
    jackknife_ratio_test,
    naive_ratio_ttest,
    ratio_and_variance,
)
from expkit.stats.multiple import adjust, benjamini_hochberg, bonferroni, holm
from expkit.stats.power import (
    binary_sd,
    minimum_detectable_effect,
    power_at,
    required_sample_size,
)
from expkit.stats.srm import DEFAULT_SRM_THRESHOLD, srm_test
from expkit.stats.ttest import mann_whitney, welch_ttest, z_test
from expkit.stats.winsor import cap_value, winsorize_pair

__all__ = [
    "BayesResult", "beta_binomial", "bootstrap_mean_diff", "bootstrap_ratio_diff",
    "add_zero_indicator", "cuped_adjust", "cuped_theta", "cuped_ttest", "delta_ratio_test",
    "jackknife_ratio_and_variance", "jackknife_ratio_test",
    "naive_ratio_ttest", "ratio_and_variance", "adjust", "benjamini_hochberg", "bonferroni",
    "holm", "binary_sd", "minimum_detectable_effect", "power_at", "required_sample_size",
    "DEFAULT_SRM_THRESHOLD", "srm_test", "mann_whitney", "welch_ttest", "z_test", "cap_value",
    "winsorize_pair",
]
