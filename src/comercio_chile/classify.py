"""Problema 2: ¿tendrá comercio el próximo mes una serie intermitente? (clasificación binaria)

Población en el origen t: series con comercio en 1..9 de los últimos 12 meses (las no regulares).
Objetivo: valor en t + 1 > 0.

Features = las de forecasting (features.py) + patrón de actividad + features del grafo de relatedness.
El grafo se reconstruye en cada origen con los 12 meses previos (inclusive t), de modo que nunca contiene
información posterior al origen.
"""
from functools import cache

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from . import graph
from .config import SEED
from .features import REGULAR_MIN_ACTIVE, Grid, features, window

FIRST_ORIGIN = 11
MIN_PRESENCE = 10_000   # US$ en 12 meses para considerar que Chile "ya comercia" q con c (matriz M)
GRAPH_FEATURES = ["densidad", "ubicuidad_p", "diversidad_c"]


def intermittent_mask(grid: Grid, t: int) -> np.ndarray:
    n = (window(grid.valor, t, 12) > 0).sum(axis=1)
    return (n >= 1) & (n < REGULAR_MIN_ACTIVE)


@cache
def _proximity(flow: str, t_label: str) -> graph.Proximity:
    months = [str(p) for p in pd.period_range(end=pd.Period(t_label, "M"), periods=12, freq="M")]
    return graph.proximity(graph.firm_products(months, flow=flow))


def graph_features(grid: Grid, t: int, rows: np.ndarray) -> pd.DataFrame:
    """Densidad de relatedness ω(p, c), ubicuidad de p y diversidad de c, con la ventana t-11..t."""
    prox = _proximity(grid.flow, str(grid.months[t]))
    win = window(grid.valor, t, 12).sum(axis=1)
    pres = grid.keys.assign(M=(win >= MIN_PRESENCE).astype(np.int8))
    M = pres.pivot_table(index="hs6", columns="pais", values="M", aggfunc="max", fill_value=0)
    ubic, divers = M.sum(axis=1), M.sum(axis=0)
    dens = graph.density(prox, M.reindex(prox.products, fill_value=0)).set_index(["hs6", "pais"]).densidad
    k = grid.keys.iloc[rows]
    mi = pd.MultiIndex.from_frame(k[["hs6", "pais"]])
    return pd.DataFrame({
        "densidad": dens.reindex(mi).to_numpy(),            # NaN si el producto no está en la red
        "ubicuidad_p": k.hs6.map(ubic).fillna(0).to_numpy(),
        "diversidad_c": k.pais.map(divers).fillna(0).to_numpy(),
    }, index=rows)


def dataset(grid: Grid, t: int, use_graph: bool = True):
    rows = np.where(intermittent_mask(grid, t))[0]
    X = features(grid, t, 1, rows)
    A = grid.valor[rows] > 0
    col = lambda k: A[:, k].astype(float) if 0 <= k < A.shape[1] else np.full(len(rows), np.nan)
    X["activo_t"], X["activo_t1"], X["activo_t2"] = col(t), col(t - 1), col(t - 2)
    X["activo_estacional"] = col(t + 1 - 12)
    X["activos_3"] = window(A, t, 3).sum(axis=1)
    X["activos_6"] = window(A, t, 6).sum(axis=1)
    if use_graph:
        X = X.join(graph_features(grid, t, rows))
    y = (grid.valor[rows, t + 1] > 0).astype(int)
    return rows, X, y


BASELINES = {
    "persistencia": lambda X: X["activo_t"].to_numpy(),
    "persistencia_estacional": lambda X: X["activo_estacional"].fillna(0).to_numpy(),
    "frecuencia_12m": lambda X: X["meses_activos_12"].to_numpy() / 12,
}

PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)


def walk_forward(grid: Grid, targets: list[str], use_graph: bool = True, params: dict | None = None) -> pd.DataFrame:
    """Probabilidades out-of-sample. Para el mes objetivo T: origen t0 = T - 1; se entrena con orígenes
    t <= t0 - 1 (cuyo resultado en t + 1 <= t0 ya se observó)."""
    cache, out = {}, []
    get = lambda t: cache.setdefault(t, dataset(grid, t, use_graph))
    for T in targets:
        t0 = grid.t_of(T) - 1
        parts = [get(t) for t in range(FIRST_ORIGIN, t0)]
        X = pd.concat([p[1] for p in parts], ignore_index=True)
        y = np.concatenate([p[2] for p in parts])
        for c in ("capitulo", "pais"):
            X[c] = X[c].astype("category")
        model = lgb.LGBMClassifier(**(params or PARAMS)).fit(X, y)
        rows, Xt, yt = get(t0)
        for c in ("capitulo", "pais"):
            Xt[c] = pd.Categorical(Xt[c].astype(str), categories=X[c].cat.categories)
        df = pd.DataFrame({"periodo": T, "serie": rows, "y": yt, "modelo": model.predict_proba(Xt)[:, 1],
                           "valor_t": grid.valor[rows, t0]})
        for name, fn in BASELINES.items():
            df[name] = fn(Xt)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"AP": average_precision_score(y, p), "AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p),
            "LogLoss": log_loss(y, p), "tasa_base": y.mean(), "n": len(y)}
