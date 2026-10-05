"""Tests anti-leakage: las features y el set de entrenamiento en el origen t0 NO deben cambiar si se
alteran los datos de meses posteriores a t0.

Estrategia: copiar la grilla, reemplazar con ruido todas las columnas > t0 y verificar que el resultado
sea idéntico. Si alguna feature leyera el futuro, el test fallaría.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from comercio_chile import forecast as F
from comercio_chile.features import features, load_grid, regular_mask


@pytest.fixture(scope="module")
def grid():
    return load_grid("exportaciones")


def _perturb_future(grid, t0):
    g = copy.deepcopy(grid)
    rng = np.random.default_rng(0)
    for name in ("valor", "cantidad", "n_decl", "n_emp"):
        m = getattr(g, name)
        m[:, t0 + 1:] = rng.uniform(0, 1e7, size=m[:, t0 + 1:].shape)
    return g


@pytest.mark.parametrize("h", [1, 2, 3])
def test_features_ignore_future(grid, h):
    t0 = grid.t_of("2025-09")
    rows = np.where(regular_mask(grid, t0))[0]
    a = features(grid, t0, h, rows)
    b = features(_perturb_future(grid, t0), t0, h, rows)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("h", [1, 3])
def test_training_set_ignores_future(grid, h):
    t0 = grid.t_of("2025-09")
    Xa, ya = F.training_set(grid, t0, h)
    Xb, yb = F.training_set(_perturb_future(grid, t0), t0, h)
    pd.testing.assert_frame_equal(Xa, Xb)
    np.testing.assert_array_equal(ya, yb)


def test_regular_mask_ignores_future(grid):
    t0 = grid.t_of("2025-09")
    np.testing.assert_array_equal(regular_mask(grid, t0), regular_mask(_perturb_future(grid, t0), t0))
