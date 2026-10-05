"""Panel denso serie × mes y features para forecasting.

Regla anti-leakage: toda feature para el origen t usa solo columnas <= t de las matrices. La única
columna > t que se toca es la del objetivo (t + h), en `target()`.

Una serie = (hs6, país) dentro de un flujo. Se usan solo bienes y países reales (sin pseudo-países ni
producto desconocido). Los valores negativos (ajustes) se truncan en 0 a nivel serie-mes para poder usar
log1p; afectan < 0,001% del valor.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PROCESSED

COMMODITY_CHAPTERS = {"26", "27", "28", "71", "74"}  # minerales, combustibles, química inorg. (litio), metales preciosos, cobre
REGULAR_MIN_ACTIVE = 10      # meses con comercio, de los últimos 12, para considerar "regular" una serie
TOP_PARTNERS = 40            # países con categoría propia; el resto -> "otro"


@dataclass
class Grid:
    flow: str
    keys: pd.DataFrame        # S filas: hs6, pais, capitulo, commodity
    months: pd.PeriodIndex    # T meses
    valor: np.ndarray         # S×T, US$
    cantidad: np.ndarray
    n_decl: np.ndarray
    n_emp: np.ndarray

    @property
    def log_valor(self) -> np.ndarray:
        return np.log1p(self.valor)

    def t_of(self, period: str) -> int:
        return self.months.get_loc(pd.Period(period, "M"))


def load_grid(flow: str) -> Grid:
    p = pd.read_parquet(PROCESSED / f"panel_{flow}.parquet")
    p = p[(p.segmento == "bienes") & ~p.pseudo_pais & (p.hs6 != "000000")]
    p["periodo"] = p.periodo.astype(str)
    keys = p[["hs6", "pais"]].drop_duplicates().sort_values(["hs6", "pais"]).reset_index(drop=True)
    months = pd.period_range(p.periodo.min(), p.periodo.max(), freq="M")
    si = pd.MultiIndex.from_frame(keys).get_indexer(pd.MultiIndex.from_frame(p[["hs6", "pais"]]))
    ti = months.get_indexer(pd.PeriodIndex(p.periodo, freq="M"))

    def mat(col):
        m = np.zeros((len(keys), len(months)))
        np.add.at(m, (si, ti), p[col].to_numpy(dtype=float))
        return m

    keys["capitulo"] = keys.hs6.str[:2]
    keys["commodity"] = keys.capitulo.isin(COMMODITY_CHAPTERS)
    return Grid(flow, keys, months, np.clip(mat("valor_usd"), 0, None), np.clip(mat("cantidad"), 0, None),
                mat("n_decl"), mat("n_empresas"))


def window(m: np.ndarray, t: int, n: int) -> np.ndarray:
    """Columnas t-n+1..t (las que existan)."""
    return m[:, max(0, t - n + 1): t + 1]


def regular_mask(grid: Grid, t: int) -> np.ndarray:
    return (window(grid.valor, t, 12) > 0).sum(axis=1) >= REGULAR_MIN_ACTIVE


def features(grid: Grid, t: int, h: int, rows: np.ndarray) -> pd.DataFrame:
    """Features en el origen t para predecir t + h, para las series `rows` (índices)."""
    V, L = grid.valor[rows], grid.log_valor[rows]
    nan = np.full(len(rows), np.nan)
    col = lambda m, k: m[:, k] if 0 <= k < m.shape[1] else nan
    f = {}
    for k in range(6):
        f[f"lag_{k}"] = col(L, t - k)
    f["lag_estacional"] = col(L, t + h - 12)           # mismo mes del año anterior que el objetivo
    f["lag_12"] = col(L, t - 12)
    for n in (3, 6, 12):
        f[f"media_log_{n}"] = window(L, t, n).mean(axis=1)
    f["log_media_3"] = np.log1p(window(V, t, 3).mean(axis=1))
    f["log_media_12"] = np.log1p(window(V, t, 12).mean(axis=1))
    # versiones relativas al nivel reciente (sin escala): permiten que el modelo global trate igual a
    # series grandes y pequeñas
    for k in ("lag_0", "lag_1", "lag_2", "lag_estacional", "lag_12", "log_media_12"):
        f[f"rel_{k}"] = f[k] - f["log_media_3"]
    f["std_log_12"] = window(L, t, 12).std(axis=1)
    act = window(V, t, 12) > 0
    f["meses_activos_12"] = act.sum(axis=1)
    # meses transcurridos desde el último mes con comercio (0 = hubo comercio en t; 12 = ninguno en la ventana)
    f["meses_desde_actividad"] = np.where(act.any(axis=1), np.argmax(act[:, ::-1], axis=1), 12)
    f["desvio_estacional"] = f["lag_estacional"] - f["media_log_12"]
    s_now, s_prev = window(V, t, 3).sum(axis=1), (window(V, t - 12, 3).sum(axis=1) if t >= 14 else nan)
    f["crec_interanual_3m"] = np.log1p(s_now) - np.log1p(s_prev)

    # precio: valor unitario actual vs mediana de 12 meses (unidades consistentes dentro de una serie)
    Q = grid.cantidad[rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.where((V > 0) & (Q > 0), V / Q, np.nan)
    uv_w = window(uv, t, 12)
    with np.errstate(all="ignore"):
        f["precio_vs_mediana12"] = np.log(col(uv, t)) - np.log(np.nanmedian(np.where(np.isnan(uv_w).all(axis=1, keepdims=True), 1, uv_w), axis=1))
    f["log_n_decl"] = np.log1p(col(grid.n_decl[rows], t))
    f["log_n_emp_media3"] = np.log1p(window(grid.n_emp[rows], t, 3).mean(axis=1))

    # contexto sectorial: capítulo completo (todas las series, no solo las seleccionadas)
    cap = grid.keys.capitulo.to_numpy()
    cap_codes, cap_idx = np.unique(cap, return_inverse=True)
    C = np.zeros((len(cap_codes), grid.valor.shape[1]))
    np.add.at(C, cap_idx, grid.valor)
    Cr = C[cap_idx[rows]]
    f["cap_crec_interanual"] = np.log1p(Cr[:, t]) - (np.log1p(Cr[:, t - 12]) if t >= 12 else nan)
    f["cap_mom_3m"] = np.log1p(Cr[:, t]) - np.log1p(window(Cr, t - 1, 3).mean(axis=1)) if t >= 1 else nan

    keys = grid.keys.iloc[rows]
    # ranking de socios solo con meses <= t (con todo el período se filtraría información futura)
    top = grid.keys.assign(v=grid.valor[:, : t + 1].sum(axis=1)).groupby("pais").v.sum().nlargest(TOP_PARTNERS).index
    out = pd.DataFrame(f, index=rows)
    out["mes_objetivo"] = (grid.months[t] + h).month
    out["capitulo"] = pd.Categorical(keys.capitulo.to_numpy())
    out["pais"] = pd.Categorical(np.where(keys.pais.isin(top), keys.pais, "otro"))
    out["commodity"] = keys.commodity.to_numpy().astype(int)
    return out


def target(grid: Grid, t: int, h: int, rows: np.ndarray) -> np.ndarray:
    """Valor (US$) en t + h: la única lectura de columnas posteriores al origen."""
    return grid.valor[rows, t + h]


CATEGORICAL = ["capitulo", "pais"]
