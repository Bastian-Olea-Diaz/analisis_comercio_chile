"""Tests del diseño experimental con datos simulados: asignación, errores agrupados, CUPED y correcciones."""
import numpy as np
import pandas as pd
import pytest

from comercio_chile import experiment as E


@pytest.fixture(scope="module")
def units():
    """Población con grupos de tamaño variable y un efecto común por grupo (correlación intraclase)."""
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 30, size=400)
    cluster = np.repeat(np.arange(len(sizes)), sizes).astype(str)
    shock = rng.normal(0, 1, len(sizes))[np.repeat(np.arange(len(sizes)), sizes)]
    x = rng.normal(0, 1, len(cluster))
    p = 1 / (1 + np.exp(-(-2 + 0.8 * x + 1.2 * shock)))
    return pd.DataFrame({"hs6": cluster, "score": p, "x": x, "y": (rng.random(len(p)) < p).astype(float)})


def test_assign_keeps_clusters_together_and_respects_weights(units):
    arm = E.assign(units, ["A", "B", "C"], [2, 1, 1], seed=1)
    assert (units.assign(b=arm).groupby("hs6").b.nunique() == 1).all()
    share = arm.groupby(units.hs6).first().value_counts(normalize=True)
    assert share["A"] == pytest.approx(0.5, abs=0.02)


def test_clustered_aa_controls_false_positives(units):
    """Con correlación dentro de los grupos, el test ingenuo da demasiados falsos positivos; el agrupado no."""
    y, cl = units.y.to_numpy(), units.hs6.to_numpy()
    naive, clustered = [], []
    for s in range(300):
        arm = E.assign(units, ["A", "A2"], [1, 1], seed=s).to_numpy()
        naive.append(E.compare(y, arm, cl, "A", clustered=False).p.iloc[0] < 0.05)
        clustered.append(E.compare(y, arm, cl, "A").p.iloc[0] < 0.05)
    assert np.mean(naive) > 0.10
    assert np.mean(clustered) < 0.09


def test_cuped_keeps_effect_and_reduces_variance(units):
    rng = np.random.default_rng(3)
    arm = E.assign(units, ["A", "B"], [1, 1], seed=3).to_numpy()
    y = units.y.to_numpy() + 0.05 * (arm == "B") + rng.normal(0, 0.01, len(arm))
    X = units[["x"]].to_numpy()
    raw = E.compare(y, arm, units.hs6.to_numpy(), "A").iloc[0]
    adj = E.compare(E.cuped(y, X), arm, units.hs6.to_numpy(), "A").iloc[0]
    assert adj.se < raw.se
    assert abs(adj.diferencia - 0.05) < 3 * adj.se


def test_holm_matches_textbook_example():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    assert E.holm(p).to_dict() == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_obf_bounds_control_alpha():
    bounds = E.obf_bounds(4, sims=100_000, seed=1)
    z = np.cumsum(np.random.default_rng(2).standard_normal((100_000, 4)), axis=1) / np.sqrt(np.arange(1, 5))
    assert (np.abs(z) >= bounds).any(axis=1).mean() == pytest.approx(0.05, abs=0.004)
    assert bounds[0] > bounds[-1] > 1.96
