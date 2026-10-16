import numpy as np
import pytest

from expkit.io import SyntheticSource


@pytest.fixture(scope="session")
def population():
    return SyntheticSource(n_users=20_000, seed=11).load()


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
