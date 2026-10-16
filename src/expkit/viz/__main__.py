import argparse

from expkit.viz import build_figures

p = argparse.ArgumentParser(prog="python -m expkit.viz", description="Draw the result charts.")
p.add_argument("--results", default="results")
p.add_argument("--out", default="docs/figures")
p.add_argument("--allow-synthetic", action="store_true")
p.add_argument("--formats", nargs="+", default=["png"])
a = p.parse_args()
for path in build_figures(a.results, a.out, tuple(a.formats), require_real=not a.allow_synthetic):
    print(path)
