import json

import pytest

from expkit.cli import main
from expkit.io import SchemaError, SyntheticSource, source_from_uri, validate_user_metrics


def test_synthetic_matches_schema_and_has_pathologies(population):
    validate_user_metrics(population)
    assert (population.pre_active == 1).all()
    assert (population.post_answers == 0).mean() > 0.3          # zero inflation
    q999 = population.post_answers.quantile(0.999)
    assert q999 > 10 * max(population.post_answers.mean(), 1e-9)  # heavy tail
    assert population[["pre_answers", "post_answers"]].corr().iloc[0, 1] > 0.3


def test_schema_rejects_violations(population):
    bad = population.head(10).copy()
    bad.loc[bad.index[0], "post_accepted"] = bad.post_answers.iloc[0] + 1
    with pytest.raises(SchemaError):
        validate_user_metrics(bad)
    with pytest.raises(SchemaError):
        validate_user_metrics(population.drop(columns="tenure_days"))
    dup = population.head(5).copy()
    dup.loc[dup.index[1], "user_id"] = dup.user_id.iloc[0]
    with pytest.raises(SchemaError):
        validate_user_metrics(dup)


def test_source_from_uri(tmp_path):
    assert isinstance(source_from_uri("synthetic:100"), SyntheticSource)
    with pytest.raises(ValueError):
        source_from_uri("data.xlsx")
    path = tmp_path / "u.parquet"
    SyntheticSource(n_users=200).load().to_parquet(path)
    assert len(source_from_uri(str(path)).load()) == 200


def test_cli_power(capsys):
    main(["power", "--baseline-mean", "1.0", "--sd", "1.0", "--mde", "0.1"])
    out = json.loads(capsys.readouterr().out)
    assert out["n_control"] == 1570


def test_cli_methods(capsys):
    main(["methods", "--kind", "ratio"])
    out = capsys.readouterr().out
    assert "delta" in out and "welch" not in out


def test_cli_simulate_and_analyze(tmp_path, population):
    out = tmp_path / "s.csv"
    main(["simulate", "--data", "synthetic:5000", "--n-per-arm", "500", "--runs", "10",
          "--out", str(out)])
    assert out.exists()

    df = population.head(4000).copy()
    df["arm"] = ["control", "treatment"] * 2000
    data = tmp_path / "d.parquet"
    df.to_parquet(data)
    report = tmp_path / "r.json"
    main(["analyze", "--data", str(data), "--config", "configs/example_experiment.toml",
          "--format", "json", "--out", str(report)])
    assert json.loads(report.read_text())["decision"] in {"ship", "do_not_ship", "extend", "invalid"}
