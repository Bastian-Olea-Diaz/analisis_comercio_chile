"""Experimento de activación producto–mercado: ¿predice la densidad de relatedness qué pares
(producto, país) nuevos empezará a exportar Chile?

Protocolo fuera de tiempo:
    base   = ventana de 12 meses donde se construye el grafo y se mide la presencia M
    target = meses siguientes; un par se activa si su valor en target >= min_presence
    candidatos = pares (p, c) con M[p, c] = 0 en la base (Chile no los exportaba)

Features sin grafo (baselines) vs con grafo (densidad):
    ubicuidad_p   = n.º de países a los que va p          (productos "fáciles de colocar")
    diversidad_c  = n.º de productos que compra c a Chile  (mercados "abiertos" a Chile)
    log valor_p, log valor_c
    densidad_pc   = ω[p, c]                               (aporta el grafo)
"""
import numpy as np
import pandas as pd

from . import graph


def months(start: str, n: int) -> list[str]:
    return [str(p) for p in pd.period_range(start, periods=n, freq="M")]


def build_dataset(panel: pd.DataFrame, base: list[str], target: list[str], *, min_value: float = 5_000,
                  min_obs: int = 10, weighting: str = "newman", min_presence: float = 10_000) -> pd.DataFrame:
    prox = graph.proximity(graph.firm_products(base, min_value), min_obs=min_obs, weighting=weighting)
    M = graph.presence(panel, base, prox.products, min_presence)
    countries = M.columns[M.sum(axis=0) > 0]
    M = M[countries]

    df = graph.density(prox, M)
    pres = M.stack().rename("presente").reset_index()
    df = df.merge(pres, on=["hs6", "pais"])

    goods = panel[(panel.segmento == "bienes") & ~panel.pseudo_pais]
    vb = goods[goods.periodo.astype(str).isin(base)]
    vt = goods[goods.periodo.astype(str).isin(target)].groupby(["hs6", "pais"]).valor_usd.sum()

    df["ubicuidad_p"] = df.hs6.map(M.sum(axis=1))
    df["diversidad_c"] = df.pais.map(M.sum(axis=0))
    df["log_valor_p"] = np.log1p(df.hs6.map(vb.groupby("hs6").valor_usd.sum()).fillna(0))
    df["log_valor_c"] = np.log1p(df.pais.map(vb.groupby("pais").valor_usd.sum()).fillna(0))
    df["valor_target"] = pd.MultiIndex.from_frame(df[["hs6", "pais"]]).map(vt).fillna(0).to_numpy()
    df["activa"] = (df.valor_target >= min_presence).astype(int)

    cand = df[df.presente == 0].drop(columns="presente").reset_index(drop=True)
    return cand


BASELINE = ["ubicuidad_p", "diversidad_c", "log_valor_p", "log_valor_c"]
WITH_GRAPH = BASELINE + ["densidad"]
