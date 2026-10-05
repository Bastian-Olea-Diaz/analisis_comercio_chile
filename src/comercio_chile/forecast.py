"""Problema 1: forecasting del valor mensual por serie (hs6 × país) con validación walk-forward.

Protocolo temporal: para un mes objetivo T y horizonte h, el origen es t0 = T - h. El modelo se entrena
solo con pares (features en t, objetivo en t + h) tales que t + h <= t0, es decir, objetivos ya
observados en t0. Se predice para las series regulares en t0. Se reentrena en cada mes, como en producción.
"""
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import SEED
from .features import Grid, features, regular_mask, target, window

FIRST_ORIGIN = 11   # 2024-12: primer origen con 12 meses de historia (rezago estacional disponible)


# ----------------------------------------------------------------------------------------- baselines
BASELINES = {
    "naive": lambda V, t, h: V[:, t],
    "naive_estacional": lambda V, t, h: V[:, t + h - 12],
    "estacional_tendencia": lambda V, t, h: V[:, t + h - 12] * (
        np.clip((window(V, t, 3).sum(axis=1) + 1) / (window(V, t - 12, 3).sum(axis=1) + 1), 0.2, 5) if t >= 14 else 1.0),
    "media_movil_3": lambda V, t, h: window(V, t, 3).mean(axis=1),
}


# ------------------------------------------------------------------------------------------- métricas
def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    """WAPE (US$), MAE en log1p (por serie) y sesgo del total agregado."""
    yhat = np.clip(yhat, 0, None)
    return {
        "WAPE": np.abs(yhat - y).sum() / y.sum(),
        "MAE_log": np.abs(np.log1p(yhat) - np.log1p(y)).mean(),
        "sesgo_total": (yhat.sum() - y.sum()) / y.sum(),
        "n": len(y),
    }


# --------------------------------------------------------------------------------------------- modelo
@dataclass
class ModelConfig:
    params: dict = field(default_factory=lambda: dict(
        objective="regression", n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1))
    # Objetivo del modelo:
    #   log1p      log(1+y) absoluto
    #   delta      log(1+y) - log(1+nivel), nivel = media de los últimos 3 meses (objetivo sin escala)
    #   raw        y en US$ (para objetivos de media como tweedie)
    #   raw_offset y en US$ con offset log(1+nivel): el modelo aprende solo el multiplicador sobre el nivel
    target: str = "delta"
    weighting: str = "none"         # none | log | sqrt   (peso de cada serie en el entrenamiento)


def _weights(X: pd.DataFrame, kind: str):
    if kind == "none":
        return None
    level = np.expm1(X["log_media_12"].to_numpy())
    return {"log": np.log1p(level), "sqrt": np.sqrt(level)}[kind]


def training_set(grid: Grid, t0: int, h: int):
    """Pares (X, y) con origen t en [FIRST_ORIGIN, t0 - h] (objetivo t + h ya observado en t0)."""
    Xs, ys = [], []
    for t in range(FIRST_ORIGIN, t0 - h + 1):
        rows = np.where(regular_mask(grid, t))[0]
        Xs.append(features(grid, t, h, rows))
        ys.append(target(grid, t, h, rows))
    X = pd.concat(Xs, ignore_index=True)
    for c in ("capitulo", "pais"):  # categorías consistentes entre orígenes
        X[c] = X[c].astype("category")
    return X, np.concatenate(ys)


def _encode_target(y: np.ndarray, X: pd.DataFrame, kind: str):
    """(objetivo transformado, init_score) según el tipo de objetivo."""
    level = X["log_media_3"].to_numpy()
    return {"log1p": (np.log1p(y), None), "delta": (np.log1p(y) - level, None),
            "raw": (y, None), "raw_offset": (y, level)}[kind]


def _decode(p: np.ndarray, X: pd.DataFrame, kind: str) -> np.ndarray:
    level = X["log_media_3"].to_numpy()
    if kind == "log1p":
        return np.expm1(p)
    if kind == "delta":
        return np.expm1(p + level)
    if kind == "raw_offset":   # predict() no suma el init_score: se agrega en la escala del link (log)
        return np.exp(np.log(np.clip(p, 1e-12, None)) + level)
    return p


def fit(grid: Grid, t0: int, h: int, cfg: ModelConfig) -> lgb.LGBMRegressor:
    X, y = training_set(grid, t0, h)
    model = lgb.LGBMRegressor(**cfg.params)
    yt, init = _encode_target(y, X, cfg.target)
    model.fit(X, yt, sample_weight=_weights(X, cfg.weighting), init_score=init)
    model.target_ = cfg.target
    model.categories_ = {c: X[c].cat.categories for c in ("capitulo", "pais")}
    return model


def predict(model, grid: Grid, t0: int, h: int, rows: np.ndarray) -> np.ndarray:
    X = features(grid, t0, h, rows)
    for c, cats in model.categories_.items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    return _decode(model.predict(X), X, model.target_).clip(0)


def walk_forward(grid: Grid, h: int, targets: list[str], cfg: ModelConfig | None = None,
                 retrain: bool = True) -> pd.DataFrame:
    """Predicciones out-of-sample para cada mes objetivo. Con retrain=False el modelo se entrena una sola vez,
    en el origen del primer mes (modelo "congelado", para estudiar degradación y drift)."""
    cfg = cfg or ModelConfig()
    out, model = [], None
    for T in targets:
        t0 = grid.t_of(T) - h
        rows = np.where(regular_mask(grid, t0))[0]
        if retrain or model is None:
            model = fit(grid, t0, h, cfg)
        df = pd.DataFrame({"periodo": T, "h": h, "serie": rows, "y": target(grid, t0, h, rows),
                           "modelo": predict(model, grid, t0, h, rows),
                           "commodity": grid.keys.commodity.to_numpy()[rows]})
        V = grid.valor[rows]
        for name, fn in BASELINES.items():
            df[name] = np.clip(fn(V, t0, h), 0, None)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def score(pred: pd.DataFrame, methods: list[str], by: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for key, g in (pred.groupby(by) if by else [((), pred)]):
        keys = dict(zip(by or [], key if isinstance(key, tuple) else (key,), strict=True))
        rows += [{**keys, "metodo": m, **metrics(g.y.to_numpy(), g[m].to_numpy())} for m in methods]
    return pd.DataFrame(rows)
