"""Monitoreo de drift en 4 capas: calidad de datos, data drift, target drift y desempeño.

Escenario: el modelo del Problema 1 se entrena una vez hasta un mes de corte y se despliega sin reentrenar;
cada mes se compara contra una referencia (lo que el modelo conocía al entrenarse). El diseño y las lecciones
del período de control están en docs/PROYECTO.md §11.

Umbrales:
    PSI de features por serie   > 0,10 advertencia | > 0,25 alerta
    Gráfico de control (z)      |z| > 2 advertencia | |z| > 3, o 2 advertencias seguidas del mismo signo, alerta.
                                μ y σ salen del desempeño del propio modelo en la referencia: un modelo de
                                mediana tiene sesgo normal negativo, no cero.
    Índice de precios           |log| > 0,10 advertencia | > 0,20 alerta (solo commodities: en productos
                                heterogéneos el valor unitario no es un precio)
    Reglas ingenuas             WAPE/ref > 1,2 / 1,5 ; |sesgo| > 10% / 20%  (solo para comparar)
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import forecast as F
from .config import INTERIM, PROCESSED
from .features import Grid, features, regular_mask, target

FULL_ORIGIN = 14   # 2025-03: primer origen con todas las features definidas; antes, el PSI compara NaN con valores
CONTEXT_FEATURES = ["cap_crec_interanual", "cap_mom_3m"]   # tasas por capítulo: cambian por diseño, sin PSI
PSI_THRESHOLDS = (0.10, 0.25)
PRICE_THRESHOLDS = (0.10, 0.20)
NAIVE = {"wape_ratio": (1.2, 1.5), "sesgo_abs": (0.10, 0.20)}
OK, WARN, ALERT = "ok", "advertencia", "alerta"
L1 = F.ModelConfig(params={**F.ModelConfig().params, "objective": "l1"}, target="delta")


def level(v: float, warn: float, alert: float) -> str:
    v = abs(v)
    return ALERT if v > alert else WARN if v > warn else OK


# ------------------------------------------------------------------------------------- estadísticos
def psi(ref, cur, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI con bins = cuantiles de la referencia; los NaN forman su propio bin."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    r, c = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(r) == 0 or len(c) == 0:
        return np.nan
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    rp = np.append(np.histogram(r, edges)[0] / len(ref), np.isnan(ref).mean())
    cp = np.append(np.histogram(c, edges)[0] / len(cur), np.isnan(cur).mean())
    rp, cp = np.clip(rp, eps, None), np.clip(cp, eps, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def psi_categorical(ref: pd.Series, cur: pd.Series, ref_w=None, cur_w=None, eps: float = 1e-4) -> float:
    """PSI sobre categorías, opcionalmente ponderado (p. ej. por valor = drift del 'mix')."""
    rp = pd.Series(1.0 if ref_w is None else ref_w, index=ref.index).groupby(ref.to_numpy()).sum()
    cp = pd.Series(1.0 if cur_w is None else cur_w, index=cur.index).groupby(cur.to_numpy()).sum()
    cats = rp.index.union(cp.index)
    rp = (rp.reindex(cats, fill_value=0) / rp.sum()).clip(eps)
    cp = (cp.reindex(cats, fill_value=0) / cp.sum()).clip(eps)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def ks(ref, cur) -> float:
    """Estadístico D de Kolmogorov–Smirnov (se reporta D, no el p-value: con n grande todo es 'significativo')."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    return float(ks_2samp(ref, cur).statistic) if len(ref) and len(cur) else np.nan


# ------------------------------------------------------------------------------------------ capas
def feature_frame(grid: Grid, origins: list[int]) -> pd.DataFrame:
    out = []
    for t in origins:
        rows = np.where(regular_mask(grid, t))[0]
        X = features(grid, t, 1, rows)
        X["_origen"], X["_serie"], X["_y"], X["_valor_t"] = t, rows, target(grid, t, 1, rows), grid.valor[rows, t]
        out.append(X)
    return pd.concat(out, ignore_index=True)


def series_features(X: pd.DataFrame) -> list[str]:
    skip = {"capitulo", "pais", "mes_objetivo", "commodity", *CONTEXT_FEATURES}
    return [c for c in X.columns if not c.startswith("_") and c not in skip]


def data_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = [{"feature": c, "PSI": psi(ref[c], cur[c]), "KS": ks(ref[c], cur[c]), "tipo": "serie"} for c in series_features(ref)]
    rows.append({"feature": "mix capítulo (ponderado por valor)", "tipo": "composición", "KS": np.nan,
                 "PSI": psi_categorical(ref.capitulo.astype(str), cur.capitulo.astype(str), ref._valor_t.to_numpy(), cur._valor_t.to_numpy())})
    rows.append({"feature": "objetivo log(1+y)", "tipo": "objetivo",
                 "PSI": psi(np.log1p(ref._y), np.log1p(cur._y)), "KS": ks(np.log1p(ref._y), np.log1p(cur._y))})
    for c in CONTEXT_FEATURES:   # contexto: se reporta el nivel ponderado por valor, sin PSI
        level_ref, level_cur = (np.average(X[c].fillna(0), weights=X._valor_t + 1) for X in (ref, cur))
        rows.append({"feature": c, "tipo": "contexto", "PSI": np.nan, "KS": np.nan, "nivel_ref": level_ref, "nivel_actual": level_cur})
    df = pd.DataFrame(rows)
    df["estado"] = [level(v, *PSI_THRESHOLDS) if pd.notna(v) else OK for v in df.PSI]
    return df


def price_index(grid: Grid, ref_origins: list[int], t: int) -> dict:
    """Índice de precios implícito: Σ w·log(valor unitario_t / mediana de referencia) / Σ w, w = valor en t,
    por segmento. Capta shocks concentrados en pocas series de alto valor (lo que un PSI por conteo diluye)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.where((grid.valor > 0) & (grid.cantidad > 0), grid.valor / grid.cantidad, np.nan)
        base = np.nanmedian(uv[:, ref_origins], axis=1)
        lr = np.log(uv[:, t] / base)
    ok = np.isfinite(lr) & regular_mask(grid, t)
    out = {}
    for seg, m in [("commodities", grid.keys.commodity.to_numpy()), ("resto", ~grid.keys.commodity.to_numpy())]:
        sel = ok & m
        out[seg] = float(np.average(lr[sel], weights=grid.valor[sel, t])) if sel.any() else np.nan
    return out


def control_limits(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def control_status(series: pd.Series, mu: float, sigma: float) -> pd.Series:
    z = (series - mu) / sigma
    st = z.abs().map(lambda v: ALERT if v > 3 else WARN if v > 2 else OK)
    consecutive = (st == WARN) & (st.shift(1) == WARN) & (np.sign(z) == np.sign(z.shift(1)))
    return st.where(~consecutive, ALERT)


# --------------------------------------------------------------------------------------- monitoreo
def run(grid: Grid, cut: str, months: list[str], perf_ref_months: list[str]) -> dict:
    """Monitorea `months` (objetivos h=1) con un modelo congelado entrenado con objetivos <= `cut`."""
    t_cut = grid.t_of(cut)
    ref_origins = list(range(FULL_ORIGIN, t_cut))
    ref = feature_frame(grid, ref_origins)

    # referencia de desempeño: walk-forward out-of-sample antes del despliegue (lo que el equipo observó)
    pref = F.walk_forward(grid, 1, perf_ref_months, L1)
    ref_perf = pd.DataFrame([{**F.metrics(g.y.values, g.modelo.values), "periodo": p} for p, g in pref.groupby("periodo")])
    ref_price = pd.DataFrame([price_index(grid, ref_origins, t) for t in ref_origins])

    frozen = F.walk_forward(grid, 1, months, L1, retrain=False)
    retrained = F.walk_forward(grid, 1, months, L1, retrain=True)
    drift, perf = [], []
    for T in months:
        t = grid.t_of(T) - 1
        dd = data_drift(ref, feature_frame(grid, [t]))
        dd["periodo"] = T
        drift.append(dd)
        fz, rt = frozen[frozen.periodo == T], retrained[retrained.periodo == T]
        mf = F.metrics(fz.y.values, fz.modelo.values)
        pi = price_index(grid, ref_origins, t)
        perf.append({"periodo": T, "WAPE": mf["WAPE"], "sesgo": mf["sesgo_total"],
                     "WAPE_reentrenado": F.metrics(rt.y.values, rt.modelo.values)["WAPE"],
                     "precio_commodities": pi["commodities"], "precio_resto": pi["resto"]})
    perf = pd.DataFrame(perf)

    # reglas ingenuas (umbral absoluto) vs gráfico de control (calibrado con la referencia)
    wref = ref_perf.WAPE.mean()
    perf["ingenua_WAPE"] = [level(v / wref, *NAIVE["wape_ratio"]) for v in perf.WAPE]
    s = perf.sesgo.map(lambda v: level(v, *NAIVE["sesgo_abs"]))
    perf["ingenua_sesgo"] = s.where(~((s != OK) & (s.shift(1) != OK)), ALERT)
    for col, refcol in [("WAPE", "WAPE"), ("sesgo", "sesgo_total")]:
        mu, sd = control_limits(ref_perf[refcol])
        perf[f"control_{col}"] = control_status(perf[col], mu, sd)
        perf[f"z_{col}"] = (perf[col] - mu) / sd
    for seg in ("commodities", "resto"):
        mu, sd = control_limits(ref_price[seg])
        perf[f"z_precio_{seg}"] = (perf[f"precio_{seg}"] - mu) / sd          # informativo
        perf[f"estado_precio_{seg}"] = perf[f"precio_{seg}"].map(lambda v: level(v, *PRICE_THRESHOLDS))
    return {"drift": pd.concat(drift, ignore_index=True), "perf": perf, "ref_perf": ref_perf,
            "ref_price": ref_price, "ref_origins": [str(grid.months[t]) for t in ref_origins]}


def count_alerts(perf: pd.DataFrame, drift: pd.DataFrame) -> dict:
    serie = drift[drift.tipo == "serie"]
    return {
        "meses": len(perf),
        "PSI features: meses con alerta": int(serie[serie.estado == ALERT].periodo.nunique()),
        "PSI features: meses con advertencia": int(serie[serie.estado == WARN].periodo.nunique()),
        "Mix por valor: meses ≥ advertencia": int(drift[(drift.tipo == "composición") & (drift.estado != OK)].shape[0]),
        "Reglas ingenuas (WAPE o sesgo): meses con alerta": int(((perf.ingenua_WAPE == ALERT) | (perf.ingenua_sesgo == ALERT)).sum()),
        "Gráfico de control (WAPE o sesgo): meses con alerta": int(((perf.control_WAPE == ALERT) | (perf.control_sesgo == ALERT)).sum()),
        "Precio commodities: meses con advertencia": int((perf.estado_precio_commodities == WARN).sum()),
        "Precio commodities: meses con alerta": int((perf.estado_precio_commodities == ALERT).sum()),
    }


# --------------------------------------------------------------------------- capa 0: calidad de datos
def data_quality(flow: str, months: list[str], ref_months: list[str]) -> pd.DataFrame:
    """Chequeos baratos al recibir el archivo del mes: integridad del parseo (log de ingesta), volumen vs el
    mismo mes del año anterior (|Δ| > 25% advertencia, > 50% alerta) y valor con códigos nunca vistos en la
    referencia (> 1% alerta). Cualquier fila descartada o valor no convertible es alerta: detiene el pipeline."""
    log = pd.read_csv(INTERIM / "_ingest_log.csv")
    log = log[log.kind == flow].assign(periodo=lambda d: d.year.astype(str) + "-" + d.month.astype(str).str.zfill(2)).set_index("periodo")
    panel = pd.read_parquet(PROCESSED / f"panel_{flow}.parquet")
    panel["periodo"] = panel.periodo.astype(str)
    ref = panel[panel.periodo.isin(ref_months)]
    hs_ref, pa_ref = set(ref.hs6), set(ref.pais)
    rows = []
    for m in months:
        cur = panel[panel.periodo == m]
        nuevo = ~cur.hs6.isin(hs_ref) | ~cur.pais.isin(pa_ref)
        r = log.loc[m]
        prev = str(pd.Period(m, "M") - 12)   # volumen vs el mismo mes del año anterior (el volumen es estacional)
        rows.append({"periodo": m, "filas": int(r.rows),
                     "Δ volumen vs año anterior %": 100 * (r.rows / log.loc[prev, "rows"] - 1) if prev in log.index else np.nan,
                     "% filas reparadas": 100 * (r.lines_joined + r.lines_padded) / r.rows,
                     "filas descartadas": int(r.invalid_rows), "valores no convertibles": int(r.numeric_coerced + r.date_coerced),
                     "% valor con códigos nuevos": 100 * cur.valor_usd[nuevo].sum() / cur.valor_usd.sum()})
    df = pd.DataFrame(rows).set_index("periodo")
    dv = df["Δ volumen vs año anterior %"].abs()
    df["estado"] = np.where((df["filas descartadas"] > 0) | (df["valores no convertibles"] > 0) | (df["% valor con códigos nuevos"] > 1)
                            | (dv > 50), ALERT, np.where(dv > 25, WARN, OK))
    return df
