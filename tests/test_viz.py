"""Charts: every figure draws from real results, and synthetic inputs are refused by default."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")
from expkit.viz import FIGURES, build_figures, draw  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_every_figure_is_documented():
    names = [f.name for f in FIGURES]
    assert len(names) == len(set(names))
    for f in FIGURES:
        assert f.title and f.how_to_read and f.takeaway, f.name


@pytest.mark.parametrize("spec", FIGURES, ids=lambda f: f.name)
def test_figure_draws_from_real_results(spec):
    if not all((RESULTS / f).exists() for f in spec.inputs):
        pytest.skip("real results not present (run make experiments)")
    fig = draw(spec, RESULTS)
    assert fig.axes and fig.get_suptitle()


def test_synthetic_inputs_are_refused(tmp_path):
    pd.DataFrame({"correction": ["none"], "family_wise_error": [0.6], "n_tests": [20],
                  "data_source": ["synthetic"]}).to_csv(tmp_path / "multiple_testing_summary.csv", index=False)
    pd.DataFrame({"drop_fraction": [0.0, 0.05], "detection_rate": [0.0, 0.6], "median_p": [0.5, 0.01],
                  "data_source": ["synthetic"] * 2}).to_csv(tmp_path / "srm_summary.csv", index=False)
    with pytest.raises(ValueError, match="not real data"):
        build_figures(tmp_path, tmp_path / "out")
