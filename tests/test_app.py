"""Streamlit app: renders, and its simulation buttons run without exceptions (headless)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP, default_timeout=300).run()


def test_app_renders(app):
    assert not app.exception
    assert [t.label for t in app.tabs][:4] == ["Sample size / MDE", "A/A simulator", "Peeking demo",
                                               "Analyze a CSV"]
    assert int(app.metric[0].value.replace(",", "")) > 0       # users per arm


def test_aa_ratio_simulation_includes_default_method(app):
    app.selectbox[0].set_value("accepted_answer_rate")
    app.slider[0].set_value(100)
    app.button[0].click().run()
    assert not app.exception
    assert "jackknife_ratio" in set(app.dataframe[0].value["method"])


def test_peeking_simulation_runs(app):
    app.slider[1].set_value(200)
    app.button[1].click().run()
    assert not app.exception
    assert set(app.dataframe[-1].value["rule"]) >= {"naive_peeking", "obf", "msprt"}


def test_results_tab_shows_documented_charts(app):
    pytest.importorskip("matplotlib")
    from expkit.viz import FIGURES
    root = Path(APP).parents[1] / "results"
    ready = [f for f in FIGURES if all((root / i).exists() for i in f.inputs)]
    if not ready:
        pytest.skip("no results present")
    assert not app.exception
    assert {f.title for f in ready} <= {h.value for h in app.subheader}
