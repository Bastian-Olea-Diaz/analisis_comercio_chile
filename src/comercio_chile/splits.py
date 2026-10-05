"""Divisiones entrenamiento / validación / test por porcentaje (Problema 1, h = 1).

Compara divisiones aleatorias por fila con divisiones cronológicas por mes objetivo sobre el mismo universo
de filas. En series de tiempo la división aleatoria filtra información: la fila (s, t+1) tiene como `lag_0`
el objetivo de la fila (s, t). En ambos casos la validación elige el número de árboles (early stopping) y el
test se evalúa una vez.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import forecast as F
from .features import Grid, features, regular_mask, target


def universe(grid: Grid, h: int = 1) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Todas las filas (serie regular en t, origen t) con objetivo observado. Devuelve X, y, mes objetivo."""
    Xs, ys, ms = [], [], []
    for t in range(F.FIRST_ORIGIN, grid.valor.shape[1] - h):
        rows = np.where(regular_mask(grid, t))[0]
        Xs.append(features(grid, t, h, rows))
        ys.append(target(grid, t, h, rows))
        ms.append(np.full(len(rows), t + h))
    X = pd.concat(Xs, ignore_index=True)
    for c in ("capitulo", "pais"):
        X[c] = X[c].astype("category")
    return X, np.concatenate(ys), np.concatenate(ms)


def random_split(n: int, fracs: tuple, seed: int = 42) -> list[np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    cut = np.cumsum([int(round(f * n)) for f in fracs[:-1]])
    return np.split(idx, cut)


def chrono_split(months: np.ndarray, fracs: tuple) -> list[np.ndarray]:
    """Divide por mes objetivo en orden temporal (proporciones sobre el n.º de meses)."""
    uniq = np.sort(np.unique(months))
    n_tr, n_va = int(round(fracs[0] * len(uniq))), int(round(fracs[1] * len(uniq)))
    groups = [uniq[:n_tr], uniq[n_tr:n_tr + n_va], uniq[n_tr + n_va:]]
    return [np.where(np.isin(months, g))[0] for g in groups]


def fit_evaluate(X, y, parts, params: dict | None = None, max_rounds: int = 2000) -> dict:
    """Entrena en `train`, elige el n.º de árboles con early stopping en `valid` y evalúa en `test`."""
    tr, va, te = parts
    enc = lambda i: np.log1p(y[i]) - X["log_media_3"].to_numpy()[i]
    p = {**(params or F.ModelConfig().params), "objective": "l1", "n_estimators": max_rounds}
    model = lgb.LGBMRegressor(**p).fit(
        X.iloc[tr], enc(tr), eval_set=[(X.iloc[va], enc(va))], eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = np.expm1(model.predict(X.iloc[te]) + X["log_media_3"].to_numpy()[te]).clip(0)
    pred_va = np.expm1(model.predict(X.iloc[va]) + X["log_media_3"].to_numpy()[va]).clip(0)
    return {"arboles_elegidos": model.best_iteration_,
            **{f"val_{k}": v for k, v in F.metrics(y[va], pred_va).items() if k != "n"},
            **{f"test_{k}": v for k, v in F.metrics(y[te], pred).items()},
            "pred_test": pred}
