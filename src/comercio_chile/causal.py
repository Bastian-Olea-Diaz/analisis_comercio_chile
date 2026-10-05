"""Efecto causal de los aranceles de EE.UU. de 2025 sobre las exportaciones chilenas.

Dos diseños, porque los dos aranceles tienen estructuras distintas:

1. Arancel recíproco del 10% (EO 14257, vigente desde el 5-abr-2025). Afectó a casi todos los productos
   chilenos que entran a EE.UU., salvo los del Annex II (cobre, minerales críticos, energía, madera,
   farmacéuticos...). Se estima con diferencias en diferencias: exportaciones de un producto gravado a EE.UU.
   vs las del mismo producto a otros destinos, antes y después, con PPML (Poisson con efectos fijos), el
   estimador estándar para flujos de comercio con ceros. La triple diferencia con los productos exentos
   queda como contraste, porque no pasa el test de pretendencias.
2. Arancel de la Sección 232 al cobre semielaborado (50%, desde el 1-ago-2025). Afecta a muy pocos flujos
   chilenos, sobre todo alambre de cobre (7408), así que se analiza como estudio de caso.

Ventana principal del arancel recíproco: abr–jul 2025, cuando casi todos los proveedores de EE.UU. pagaban el
mismo 10%. Desde agosto cambian las reglas (tasas mayores para varios competidores el 7-ago, modificación del
Annex II el 8-sep, Sección 232 a la madera el 14-oct, exención agrícola el 13-nov, fallo de la Corte Suprema y
paso a la Sección 122 en feb-2026), y esos meses solo se describen.
"""
import re

import numpy as np
import pandas as pd
import pyfixest as pf
import requests

from .config import PROCESSED, REFERENCES

ANNEX_II_URL = "https://www.whitehouse.gov/wp-content/uploads/2025/04/Annex-II.pdf"
ANNEX_II_PDF = REFERENCES / "raw" / "eo14257_annex_ii.pdf"
ANNEX_II_CSV = REFERENCES / "codigos" / "eeuu_annex_ii_hs6.csv"

US = "225"
REFERENCE_MONTH = pd.Period("2024-12", "M")   # último mes antes del cambio de gobierno en EE.UU.
TREATED = pd.Period("2025-04", "M")           # arancel recíproco vigente
MAIN_END = pd.Period("2025-07", "M")          # fin de la ventana con un 10% igual para casi todos

# Capítulos fuera de la comparación: tuvieron su propio arancel o una anticipación ajena al arancel recíproco.
EXCLUDED_CHAPTERS = {
    "72": "acero (Sección 232)",
    "73": "manufacturas de acero (Sección 232)",
    "76": "aluminio (Sección 232)",
    "74": "cobre (Sección 232 desde agosto y arbitraje de precios del cátodo durante la investigación)",
    "71": "metales preciosos (dudas arancelarias sobre el oro y cambios en el Annex II de septiembre)",
}
EXCLUDED_HS4 = {"8703": "automóviles (Sección 232)", "8708": "partes de automóviles (Sección 232)"}
COPPER_SEMIS = ("7407", "7408", "7409", "7411")   # barras, alambre, planchas y tubos


# ------------------------------------------------------------------------------------- exenciones
def _download(url: str, attempts: int = 5) -> bytes:
    """whitehouse.gov corta a veces la transferencia a medias: se reintenta y se valida el largo."""
    for i in range(attempts):
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            expected = int(resp.headers.get("Content-Length", len(resp.content)))
            if len(resp.content) == expected:
                return resp.content
        except requests.RequestException:
            if i == attempts - 1:
                raise
    raise OSError(f"Descarga incompleta: {url}")


def build_annex_ii() -> pd.DataFrame:
    """Annex II (PDF oficial) -> hs6 exento, marcando si la exención cubre la subpartida completa.

    El anexo lista subpartidas de 8 dígitos de EE.UU. Cuando aparece `hs6 + "00"` la exención cubre la
    subpartida internacional completa; cuando solo aparecen algunas aperturas de 8 dígitos, el hs6 queda
    como exento parcial y se excluye del análisis, porque no se puede saber qué parte del flujo estaba gravada.
    """
    import pypdf

    if not ANNEX_II_PDF.exists():
        ANNEX_II_PDF.parent.mkdir(parents=True, exist_ok=True)
        ANNEX_II_PDF.write_bytes(_download(ANNEX_II_URL))
    text = "\n".join(page.extract_text() for page in pypdf.PdfReader(ANNEX_II_PDF).pages)
    hts8 = pd.Series(sorted(set(re.findall(r"(?m)^\s*(\d{8})\b", text))))
    df = pd.DataFrame({"hts8": hts8, "hs6": hts8.str[:6]})
    out = df.groupby("hs6").hts8.agg(n_hts8="size", completo=lambda s: s.str.endswith("00").any())
    return out.reset_index()


def annex_ii() -> pd.DataFrame:
    return pd.read_csv(ANNEX_II_CSV, dtype={"hs6": str})


def product_groups(hs6: pd.Series) -> pd.Series:
    """hs6 -> 'gravado' | 'exento' | 'exento_parcial' | 'excluido'."""
    ex = annex_ii().set_index("hs6")
    group = pd.Series("gravado", index=hs6.index)
    listed = hs6.map(ex.completo)
    group[listed == True] = "exento"   # noqa: E712 (map deja NaN para los no listados)
    group[listed == False] = "exento_parcial"   # noqa: E712
    group[hs6.str[:2].isin(EXCLUDED_CHAPTERS) | hs6.str[:4].isin(EXCLUDED_HS4)] = "excluido"
    return group


# ------------------------------------------------------------------------------------- panel
def panel(end: str = "2026-08") -> pd.DataFrame:
    """Panel balanceado hs6 × destino × mes de exportaciones de bienes, con ceros explícitos.

    Entran los pares hs6–destino con comercio en el período previo (ene-2024 a mar-2025): definir la
    muestra con meses posteriores al arancel la condicionaría al resultado.
    """
    p = pd.read_parquet(PROCESSED / "panel_exportaciones.parquet",
                        columns=["periodo", "segmento", "hs6", "pais", "valor_usd", "pseudo_pais"])
    p = p[(p.segmento == "bienes") & ~p.pseudo_pais & (p.periodo <= pd.Period(end, "M"))]
    p = p.groupby(["hs6", "pais", "periodo"], as_index=False).valor_usd.sum()

    pre = p[p.periodo < TREATED]
    pairs = pre.loc[pre.valor_usd > 0, ["hs6", "pais"]].drop_duplicates()
    months = pd.period_range("2024-01", end, freq="M")
    grid = pairs.merge(pd.DataFrame({"periodo": months}), how="cross")
    df = grid.merge(p, on=["hs6", "pais", "periodo"], how="left").fillna({"valor_usd": 0.0})
    # 6 celdas con valor neto negativo (correcciones de declaraciones, en total US$ 1.300): PPML exige y >= 0
    df["valor_usd"] = df.valor_usd.clip(lower=0)

    df["grupo"] = product_groups(df.hs6)
    df["eeuu"] = df.pais == US
    df["post"] = df.periodo >= TREATED
    df["mes"] = (df.periodo - REFERENCE_MONTH).apply(lambda d: d.n)   # meses desde la referencia
    df["valor_musd"] = df.valor_usd / 1e6
    return df


def estimation_sample(df: pd.DataFrame, end: pd.Period = MAIN_END, groups=("gravado", "exento")) -> pd.DataFrame:
    """Muestra de estimación: solo productos que Chile exportó a EE.UU. antes del arancel.

    Un producto que nunca fue a EE.UU. no aporta a la comparación (su efecto fijo producto × mes absorbe todo);
    dejarlo solo agrega ruido a los efectos fijos de destino.
    """
    d = df[(df.periodo <= end) & df.grupo.isin(groups)].copy()
    to_us = set(d.loc[d.eeuu, "hs6"])
    d = d[d.hs6.isin(to_us)]
    d["gravado"] = d.grupo == "gravado"
    d["tratado"] = d.eeuu & d.gravado & d.post
    d["par"] = d.hs6 + "_" + d.pais
    d["hs6_mes"] = d.hs6 + "_" + d.periodo.astype(str)
    d["pais_mes"] = d.pais + "_" + d.periodo.astype(str)
    return d


# ------------------------------------------------------------------------------------- modelos
FE_DID = "par + hs6_mes"
FE_DDD = "par + hs6_mes + pais_mes"


def effect(beta: float) -> float:
    """Coeficiente PPML -> cambio porcentual del valor exportado."""
    return float(np.expm1(beta))


def _fepois(fml: str, d: pd.DataFrame, cluster: str):
    return pf.fepois(fml, data=d, vcov={"CRV1": cluster}, fixef_maxiter=100_000)


def did(d: pd.DataFrame, cluster: str = "hs6"):
    """Doble diferencia PPML con productos gravados: EE.UU. vs otros destinos del mismo producto.

    Efectos fijos:
        par (hs6 × destino)  nivel de cada flujo
        hs6 × mes            precios y oferta de cada producto (cosechas, ciclo de precios)
    `tratado` mide cuánto cambió el valor exportado a EE.UU. frente al que se exportó del mismo producto,
    el mismo mes, a otros destinos. Supuesto: sin el arancel, esa brecha habría seguido igual (tendencias
    paralelas), algo que se contrasta con el event study.
    """
    return _fepois(f"valor_musd ~ tratado | {FE_DID}", d[d.gravado], cluster)


def ddd(d: pd.DataFrame, cluster: str = "hs6"):
    """Triple diferencia: agrega los productos exentos (Annex II) y el efecto fijo destino × mes.

    En teoría separa el arancel de un shock de demanda propio de EE.UU.; en la práctica los exentos que
    Chile vende a EE.UU. son pocos y volátiles (madera, yodo, litio), y sus pretendencias no pasan el test.
    Se reporta como contraste.
    """
    return _fepois(f"valor_musd ~ tratado | {FE_DDD}", d, cluster)


def event_study(d: pd.DataFrame, triple: bool = False, cluster: str = "hs6"):
    """Un coeficiente por mes, relativo a dic-2024 (fijado en 0). Devuelve (tabla, test de pretendencias).

    El test es un Wald conjunto sobre los meses de 2024: si se rechaza, la brecha ya se movía antes de
    cualquier anuncio y el diseño no es creíble. Ene–mar 2025 se excluyen del test a propósito: ahí puede
    haber anticipación (adelanto de embarques), que es parte de la historia, no una violación del supuesto.
    """
    d = d.copy() if triple else d[d.gravado].copy()
    treated = d.eeuu & d.gravado
    months = [m for m in sorted(d.mes.unique()) if m != 0]
    names = [f"m_{m}".replace("-", "n") for m in months]
    for m, name in zip(months, names):
        d[name] = (treated & (d.mes == m)).astype(float)
    fit = _fepois(f"valor_musd ~ {' + '.join(names)} | {FE_DDD if triple else FE_DID}", d, cluster)

    pre = [i for i, m in enumerate(months) if m < 0]
    R = np.zeros((len(pre), len(names)))
    R[range(len(pre)), pre] = 1
    wald = fit.wald_test(R=R)

    tidy = fit.tidy().loc[names, ["Estimate", "2.5%", "97.5%"]]
    tidy.columns = ["beta", "ic_inf", "ic_sup"]
    tidy["mes"] = months
    tidy = pd.concat([tidy, pd.DataFrame({"beta": [0.0], "ic_inf": [0.0], "ic_sup": [0.0], "mes": [0]})])
    tidy["periodo"] = [REFERENCE_MONTH + m for m in tidy.mes]
    return tidy.sort_values("mes").reset_index(drop=True), {"estadistico": float(wald.iloc[0]), "p": float(wald.iloc[1])}


def placebo_countries(d: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Inferencia por permutación: repite la doble diferencia asignando el 'arancel' a otros destinos.

    Con un solo país tratado, los errores estándar agrupados pueden ser optimistas. Si el efecto de EE.UU.
    es real, debería quedar en la cola de la distribución de efectos placebo. EE.UU. sale de cada placebo
    para que su propio cambio no contamine la comparación.
    """
    base = d[d.gravado & ~d.eeuu]
    top = base[~base.post].groupby("pais").valor_usd.sum().nlargest(n).index
    rows = []
    for c in top:
        x = base.assign(tratado=(base.pais == c) & base.post)
        rows.append({"pais": c, "beta": did(x).coef()["tratado"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- cobre
def copper_semis() -> pd.DataFrame:
    """Exportaciones mensuales de semielaborados de cobre (valor y toneladas), a EE.UU. y al resto."""
    p = pd.read_parquet(PROCESSED / "panel_exportaciones.parquet",
                        columns=["periodo", "segmento", "hs6", "pais", "valor_usd", "peso_kg", "pseudo_pais"])
    p = p[(p.segmento == "bienes") & ~p.pseudo_pais & p.hs6.str[:4].isin(COPPER_SEMIS)]
    p = p.assign(destino=np.where(p.pais == US, "eeuu", "resto"))
    t = p.groupby(["periodo", "destino"]).agg(musd=("valor_usd", "sum"), toneladas=("peso_kg", "sum")).unstack(fill_value=0)
    t.columns = [f"{v}_{dest}" for v, dest in t.columns]
    t["musd_eeuu"] /= 1e6
    t["musd_resto"] /= 1e6
    t["toneladas_eeuu"] /= 1e3
    t["toneladas_resto"] /= 1e3
    return t.reindex(pd.period_range(t.index.min(), t.index.max(), freq="M"), fill_value=0)


def block_bootstrap_diff(pre: np.ndarray, post: np.ndarray, block: int = 3, n: int = 5000, seed: int = 42):
    """IC 95% de la diferencia de medias post − pre, remuestreando bloques de meses consecutivos.

    Los bloques conservan la autocorrelación de la serie mensual; un bootstrap mes a mes la ignoraría y
    daría intervalos demasiado estrechos.
    """
    rng = np.random.default_rng(seed)

    def resample(x):
        starts = rng.integers(0, len(x) - block + 1, size=int(np.ceil(len(x) / block)))
        return np.concatenate([x[s:s + block] for s in starts])[:len(x)]

    diffs = np.array([resample(post).mean() - resample(pre).mean() for _ in range(n)])
    return post.mean() - pre.mean(), np.percentile(diffs, [2.5, 97.5])


def main() -> None:
    ANNEX_II_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = build_annex_ii()
    out.to_csv(ANNEX_II_CSV, index=False)
    print(f"Annex II: {len(out)} hs6 ({out.completo.sum()} completos, {(~out.completo).sum()} parciales)")


if __name__ == "__main__":
    main()
