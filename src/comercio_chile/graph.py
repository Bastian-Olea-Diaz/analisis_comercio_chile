"""Grafos de relatedness entre productos (capacidades compartidas) y densidad producto–mercado.

Idea: dos productos están relacionados si las mismas empresas los exportan (comparten capacidades:
cadena de frío, insumos, logística, know-how). Los IDs de empresa solo son válidos dentro de un mes,
así que la co-ocurrencia se cuenta por empresa-mes y luego se suma: nunca se vincula una empresa
entre meses.

Definiciones (P = productos HS6, C = países, F = empresas-mes):
    X[f, p]  = 1 si la empresa-mes f exportó p con valor >= min_value
    k_f      = n.º de productos de f
    N_p      = Σ_f X[f, p]                                    (empresas-mes que exportan p)
    C[p, q]  = Σ_f w_f X[f, p] X[f, q]                         (co-ocurrencia, p != q)
               w_f = 1                si weighting="none"
               w_f = 1 / (k_f - 1)    si weighting="newman"  (Newman 2001: cada empresa reparte
                                       una unidad de vínculo entre sus productos; evita que unos
                                       pocos distribuidores con decenas de productos dominen)
    φ[p, q]  = C[p, q] / max(N_p, N_q)                         (proximidad; Hidalgo et al. 2007
                                                                usan el mínimo de las condicionales)
    M[q, c]  = 1 si Chile exportó q a c en la ventana (valor >= min_presence)
    ω[p, c]  = Σ_q φ[p, q] M[q, c] / Σ_q φ[p, q]                (densidad de relatedness)

ω[p, c] responde: "de los productos relacionados con p, ¿qué fracción (ponderada) ya se exporta a c?".
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from .data import load
from .schema import FIELDS, GOODS_OPERATIONS


def firm_products(periods: list[str], min_value: float = 5_000, flow: str = "exportaciones") -> pd.DataFrame:
    """(periodo, empresa, hs6, valor) de bienes con valor >= min_value por empresa-mes.

    En importaciones, la red se interpreta como "productos que compran las mismas empresas" (cadenas de
    insumos). El ID de importador también es válido solo dentro del mes.
    """
    firm, hs, val = (FIELDS[flow][k] for k in ("trader", "hs", "item_value"))
    cols = [firm, hs, val] + (["TIPOOPERACION"] if flow == "exportaciones" else [])
    df = load(flow, cols, periods=periods)
    if flow == "exportaciones":
        df = df[df.TIPOOPERACION.isin(GOODS_OPERATIONS)]
    df = df.dropna(subset=[hs])
    df["hs6"] = df[hs].str[:6]
    fp = (df.groupby(["periodo", firm, "hs6"], observed=True)[val].sum()
          .reset_index().rename(columns={firm: "empresa", val: "valor"}))
    return fp[fp.valor >= min_value]


@dataclass
class Proximity:
    products: pd.Index        # hs6, en el orden de las filas/columnas de phi
    n_obs: pd.Series          # N_p: empresas-mes que exportan p
    phi: np.ndarray           # matriz P×P simétrica, diagonal 0

    def edges(self, min_phi: float = 0.0) -> pd.DataFrame:
        i, j = np.triu_indices_from(self.phi, k=1)
        w = self.phi[i, j]
        keep = w > min_phi
        return pd.DataFrame({"p": self.products[i[keep]], "q": self.products[j[keep]], "phi": w[keep]})


def proximity(fp: pd.DataFrame, min_obs: int = 10, weighting: str = "newman") -> Proximity:
    """Matriz de proximidad entre productos a partir de co-exportación por empresa-mes."""
    n_obs = fp.groupby("hs6").size()
    products = n_obs[n_obs >= min_obs].index.sort_values()
    fp = fp[fp.hs6.isin(products)]

    row_id = fp.groupby(["periodo", "empresa"], observed=True).ngroup().to_numpy()
    col_id = products.get_indexer(fp.hs6)
    X = sparse.csr_matrix((np.ones(len(fp)), (row_id, col_id)), shape=(row_id.max() + 1, len(products)))

    k = np.asarray(X.sum(axis=1)).ravel()
    if weighting == "newman":
        w = np.where(k > 1, 1.0 / np.maximum(k - 1, 1), 0.0)
    elif weighting == "none":
        w = np.ones_like(k)
    else:
        raise ValueError(weighting)

    C = (X.T @ sparse.diags(w) @ X).toarray()
    np.fill_diagonal(C, 0)
    N = np.asarray(X.sum(axis=0)).ravel()
    phi = C / np.maximum.outer(N, N)
    return Proximity(products=products, n_obs=pd.Series(N, index=products), phi=phi)


def presence(panel: pd.DataFrame, periods: list[str], products: pd.Index, min_presence: float = 10_000) -> pd.DataFrame:
    """Matriz binaria M[hs6, país]: Chile exportó el producto al país en la ventana."""
    w = panel[panel.periodo.astype(str).isin(periods) & (panel.segmento == "bienes") & ~panel.pseudo_pais]
    v = w.groupby(["hs6", "pais"]).valor_usd.sum().unstack(fill_value=0)
    return (v.reindex(index=products, fill_value=0) >= min_presence).astype(np.int8)


def density(prox: Proximity, M: pd.DataFrame) -> pd.DataFrame:
    """ω[p, c] para todos los productos de la red y países de M (formato largo)."""
    phi = prox.phi
    denom = phi.sum(axis=1, keepdims=True)
    omega = np.divide(phi @ M.values, denom, out=np.zeros((len(phi), M.shape[1])), where=denom > 0)
    return (pd.DataFrame(omega, index=prox.products, columns=M.columns)
            .stack().rename("densidad").reset_index().rename(columns={"level_0": "hs6"}))


def to_graph(prox: Proximity, min_phi: float = 0.0) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(prox.products)
    G.add_weighted_edges_from(prox.edges(min_phi).itertuples(index=False, name=None), weight="phi")
    return G


def communities(G: nx.Graph, resolution: float = 1.0, seed: int = 42) -> pd.Series:
    """Louvain (maximiza modularidad) -> Series hs6 -> id de comunidad (0 = la de mayor tamaño)."""
    parts = nx.community.louvain_communities(G, weight="phi", resolution=resolution, seed=seed)
    parts = sorted(parts, key=len, reverse=True)
    return pd.Series({n: i for i, c in enumerate(parts) for n in c}, name="comunidad")


def backbone(G: nx.Graph, extra_quantile: float = 0.995) -> nx.Graph:
    """Esqueleto para visualizar: árbol de expansión máxima + aristas más fuertes (como el Product Space)."""
    B = nx.maximum_spanning_tree(G, weight="phi")
    w = np.array([d["phi"] for *_, d in G.edges(data=True)])
    thr = np.quantile(w, extra_quantile)
    B.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d["phi"] >= thr)
    return B


def participation(G: nx.Graph, com: pd.Series) -> pd.Series:
    """Coeficiente de participación (Guimerà & Amaral 2005): P_i = 1 - Σ_s (k_is / k_i)^2.

    0 = todas las conexiones del producto quedan dentro de una comunidad; cerca de 1 = conexiones
    repartidas entre muchas comunidades (producto "puente" entre ecosistemas).
    """
    out = {}
    for n in G:
        w = pd.Series({nb: d["phi"] for nb, d in G[n].items()})
        if w.sum() == 0:
            out[n] = 0.0
            continue
        share = w.groupby(com.reindex(w.index).to_numpy()).sum() / w.sum()
        out[n] = 1 - (share ** 2).sum()
    return pd.Series(out, name="participacion")
