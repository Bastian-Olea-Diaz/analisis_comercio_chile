"""Validación del estimador de diferencias en diferencias con datos simulados.

Antes de creerle a un estimador sobre datos reales, conviene comprobar que recupera un efecto conocido cuando
se cumple su supuesto, y que no inventa uno cuando no hay efecto. Los datos simulados imitan la estructura del
panel real: flujos de escala muy distinta, shocks por producto y mes, y muchos ceros.
"""
import numpy as np
import pandas as pd
import pytest

from comercio_chile import causal as C


def _simulate(true_effect: float, n_products: int = 150, n_countries: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.period_range("2024-01", "2025-07", freq="M")
    p, c, t = np.meshgrid(np.arange(n_products), np.arange(n_countries), np.arange(len(months)), indexing="ij")
    p, c, t = p.ravel(), c.ravel(), t.ravel()
    pair_level = rng.normal(0, 1.5, size=(n_products, n_countries))[p, c]   # escalas muy distintas
    product_month = rng.normal(0, 0.4, size=(n_products, len(months)))[p, t]   # precios, cosechas
    us, post = c == 0, months[t] >= C.TREATED
    mu = np.exp(pair_level + product_month + np.log1p(true_effect) * (us & post))
    y = rng.poisson(mu * 2) / 2   # ruido de conteo y ceros
    d = pd.DataFrame({"hs6": p.astype(str), "pais": c.astype(str), "periodo": months[t], "valor_musd": y})
    d["eeuu"], d["post"], d["gravado"] = us, post, True
    d["tratado"] = us & post
    d["par"] = d.hs6 + "_" + d.pais
    d["hs6_mes"] = d.hs6 + "_" + d.periodo.astype(str)
    return d


@pytest.mark.parametrize("true_effect", [-0.30, -0.10, 0.0])
def test_did_recovers_known_effect(true_effect):
    fit = C.did(_simulate(true_effect))
    estimate = C.effect(fit.coef()["tratado"])
    low, high = (C.effect(v) for v in fit.confint().loc["tratado"])
    assert abs(estimate - true_effect) < 0.05
    assert low <= true_effect <= high


def test_sample_ignores_post_period():
    """Los pares producto–destino se eligen solo con datos previos al arancel."""
    df = C.panel(end="2025-07")
    pairs = df[["hs6", "pais"]].drop_duplicates()
    pre_pairs = df[~df.post & (df.valor_usd > 0)][["hs6", "pais"]].drop_duplicates()
    assert len(pairs) == len(pre_pairs)
