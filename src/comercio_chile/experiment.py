"""Diseño y análisis de un experimento A/B/n semi-sintético sobre oportunidades de exportación.

Escenario: una agencia de promoción de exportaciones prueba tipos de apoyo sobre pares producto–país que Chile no
exportaba en 2025. Qué es real y qué es simulado:
    real       la población (candidatos del notebook 02), las covariables previas, el resultado sin tratamiento
               Y(0) (si el par se exportó en ene–ago 2026) y su trayectoria mes a mes
    simulado   la asignación (aleatoria, como en un experimento) y el efecto de cada tratamiento, que es un
               supuesto declarado: se inyecta sobre Y(0) y por eso el efecto verdadero se conoce

Unidad de aleatorización: el producto (hs6). Una campaña sobre vino en Japón llega a los mismos exportadores que
una sobre vino en Corea, así que aleatorizar pares sueltos contaminaría el control (interferencia). Todos los
pares de un producto reciben el mismo brazo y la inferencia usa errores agrupados por producto.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .config import PROCESSED

MIN_PRESENCE = 10_000        # US$ en ene–ago 2026 para considerar "activado" un par (igual que el notebook 02)
OUTCOME_MONTHS = pd.period_range("2026-01", "2026-08", freq="M")
COVARIATES = ["score", "log_valor_2024", "log_valor_2025", "meses_2024", "meses_2025"]


# ------------------------------------------------------------------------------------- población
def opportunities(n: int) -> pd.DataFrame:
    """Los n candidatos con mayor probabilidad de activación según el modelo con grafo (notebook 02).

    Covariables previas al tratamiento (todo hasta dic-2025): el puntaje del modelo, el valor exportado del
    par en 2024 y 2025 (en 2025 siempre bajo US$ 10.000, por definición de candidato) y los meses con comercio.
    Resultado: `activa` y la trayectoria acumulada `activa_<mes>` para analizar revisiones intermedias.
    """
    cand = pd.read_parquet(PROCESSED / "activacion_test_scores.parquet", columns=["hs6", "pais", "score", "activa"])
    cand = cand.sort_values("score", ascending=False).head(n).reset_index(drop=True)

    p = pd.read_parquet(PROCESSED / "panel_exportaciones.parquet",
                        columns=["periodo", "segmento", "hs6", "pais", "valor_usd", "pseudo_pais"])
    p = p[(p.segmento == "bienes") & ~p.pseudo_pais].assign(valor_usd=lambda x: x.valor_usd.clip(lower=0))
    p = p.merge(cand[["hs6", "pais"]], on=["hs6", "pais"])

    year = p.periodo.dt.year
    for y in (2024, 2025):
        g = p[year == y].groupby(["hs6", "pais"]).agg(v=("valor_usd", "sum"), m=("periodo", "nunique"))
        cand = cand.merge(g.rename(columns={"v": f"valor_{y}", "m": f"meses_{y}"}).reset_index(), on=["hs6", "pais"], how="left")
        cand[f"log_valor_{y}"] = np.log1p(cand.pop(f"valor_{y}").fillna(0))
        cand[f"meses_{y}"] = cand[f"meses_{y}"].fillna(0)

    cum = (p[p.periodo.isin(OUTCOME_MONTHS)].groupby(["hs6", "pais", "periodo"]).valor_usd.sum()
           .unstack(fill_value=0).reindex(columns=OUTCOME_MONTHS, fill_value=0).cumsum(axis=1) >= MIN_PRESENCE)
    cum.columns = [f"activa_{m}" for m in cum.columns]
    cand = cand.merge(cum.astype(int).reset_index(), on=["hs6", "pais"], how="left").fillna({c: 0 for c in cum.columns})
    assert (cand[f"activa_{OUTCOME_MONTHS[-1]}"] == cand.activa).all()   # la trayectoria reproduce el resultado
    return cand


def icc(y: np.ndarray, cluster: np.ndarray) -> float:
    """Correlación intraclase (estimador ANOVA): cuánto se parecen los resultados dentro de un mismo grupo."""
    df = pd.DataFrame({"y": y, "c": cluster})
    g = df.groupby("c").y
    m, k, n = g.size(), g.ngroups, len(df)
    msb = (m * (g.mean() - df.y.mean()) ** 2).sum() / (k - 1)
    msw = ((df.y - g.transform("mean")) ** 2).sum() / (n - k)
    m0 = (n - (m ** 2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (m0 - 1) * msw))


# ------------------------------------------------------------------------------------- diseño
def sample_size(p0: float, rel_lift: float, alpha: float = 0.05, power: float = 0.80, comparisons: int = 1,
                deff: float = 1.0, r2: float = 0.0, control_ratio: float = 1.0) -> dict:
    """Tamaño por brazo para detectar p0 -> p0·(1 + rel_lift) en cada comparación contra el control.

    comparisons    Bonferroni para planificar (conservador; el análisis usa Holm o Dunnett, que tienen más potencia)
    deff           efecto de diseño por aleatorizar grupos: 1 + (m − 1)·ICC
    r2             varianza explicada por las covariables (CUPED): la varianza se multiplica por (1 − r2)
    control_ratio  tamaño del control relativo a cada tratamiento (√k es el óptimo con k tratamientos)
    """
    p1 = p0 * (1 + rel_lift)
    z = stats.norm.ppf(1 - alpha / (2 * comparisons)) + stats.norm.ppf(power)
    var = (p0 * (1 - p0) / control_ratio + p1 * (1 - p1)) * deff * (1 - r2)
    n_treat = int(np.ceil(z ** 2 * var / (p1 - p0) ** 2))
    return {"n_tratamiento": n_treat, "n_control": int(np.ceil(n_treat * control_ratio)),
            "total": int(np.ceil(n_treat * (comparisons + control_ratio)))}


def mde(p0: float, n_treat: int, n_control: int, alpha: float = 0.05, power: float = 0.80, comparisons: int = 1,
        deff: float = 1.0, r2: float = 0.0) -> float:
    """Efecto mínimo detectable (relativo) dado el tamaño de cada brazo. Se resuelve numéricamente."""
    z = stats.norm.ppf(1 - alpha / (2 * comparisons)) + stats.norm.ppf(power)
    lifts = np.linspace(0.001, 3, 30_000)
    p1 = p0 * (1 + lifts)
    se = np.sqrt((p0 * (1 - p0) / n_control + p1 * (1 - p1) / n_treat) * deff * (1 - r2))
    return float(lifts[np.argmax(p1 - p0 >= z * se)])


def assign(units: pd.DataFrame, arms: list[str], weights: list[float], cluster: str = "hs6",
           strata_by: str = "score", n_strata: int = 4, seed: int = 0) -> pd.Series:
    """Aleatorización por grupos (clusters), estratificada.

    Los productos se ordenan en estratos según el promedio de `strata_by` de sus pares; dentro de cada estrato
    se barajan y se reparten entre los brazos en la proporción de `weights`. Estratificar asegura que todos los
    brazos reciban oportunidades "buenas" y "malas" en la misma proporción, lo que reduce la varianza.
    """
    rng = np.random.default_rng(seed)
    cl = units.groupby(cluster)[strata_by].mean().rename("x").to_frame()
    cl["estrato"] = pd.qcut(cl.x.rank(method="first"), n_strata, labels=False)
    w = np.asarray(weights, float) / np.sum(weights)
    out = {}
    for _, g in cl.groupby("estrato"):
        ids = rng.permutation(g.index.to_numpy())
        cuts = np.round(np.cumsum(w) * len(ids)).astype(int)
        for arm, chunk in zip(arms, np.split(ids, cuts[:-1])):
            out.update(dict.fromkeys(chunk, arm))
    return units[cluster].map(out)


def srm_test(arm: pd.Series, arms: list[str], weights: list[float]) -> dict:
    """Sample Ratio Mismatch: ¿los tamaños observados de cada brazo calzan con los planificados?

    Un SRM delata un error de implementación (unidades perdidas, asignación sesgada) que invalida el análisis.
    Se usa un umbral estricto (p < 0,001), como en la industria, porque con muestras grandes pequeñas diferencias
    por azar son normales.
    """
    obs = arm.value_counts().reindex(arms).to_numpy()
    exp = np.asarray(weights, float) / np.sum(weights) * obs.sum()
    chi2, p = stats.chisquare(obs, exp)
    return {"observado": dict(zip(arms, obs.tolist())), "esperado": dict(zip(arms, exp.round(1).tolist())),
            "chi2": float(chi2), "p": float(p)}


def smd(units: pd.DataFrame, arm: pd.Series, control: str, columns: list[str]) -> pd.DataFrame:
    """Diferencia de medias estandarizada de cada covariable vs el control (balance; |SMD| < 0,1 es lo usual)."""
    rows = {}
    for a in arm.unique():
        if a == control:
            continue
        x, c = units[arm == a], units[arm == control]
        rows[a] = {col: (x[col].mean() - c[col].mean()) / np.sqrt((x[col].var() + c[col].var()) / 2) for col in columns}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------- efecto simulado
def base_probability(units: pd.DataFrame) -> np.ndarray:
    """Probabilidad de activación sin tratamiento por par: el puntaje recalibrado a la tasa observada.

    El modelo del notebook 02 sobreestima en el tope del ranking (predice 15% donde se observa 12%); se
    reescala para que el promedio coincida con la tasa real.
    """
    return np.clip(units.score.to_numpy() * units.activa.mean() / units.score.mean(), 0, 0.95)


def potential_outcomes(y0: np.ndarray, q: np.ndarray, rel_lift: float, seed: int = 0) -> np.ndarray:
    """Y(1) a partir de Y(0) real: los pares que se activaron igual siguen activos; los demás se activan con
    probabilidad lift·q/(1 − q), de modo que la tasa esperada sube de q a q·(1 + lift).

    El efecto absoluto es mayor donde la probabilidad base es mayor: el apoyo ayuda más donde ya hay capacidades.
    """
    rng = np.random.default_rng(seed)
    flip = rng.random(len(y0)) < np.clip(rel_lift * q / (1 - q), 0, 1)
    return np.maximum(y0, flip.astype(int))


# ------------------------------------------------------------------------------------- análisis
def cuped(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """CUPED con varias covariables previas: Y − (Ŷ − media(Ŷ)), con Ŷ la proyección lineal de Y en X.

    Las covariables son anteriores al tratamiento, así que no pueden estar afectadas por él: el ajuste no
    cambia el efecto esperado, solo quita la varianza que X explica.
    """
    Xc = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
    pred = Xc @ beta
    return y - (pred - pred.mean())


def _cluster_mean(y: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """Media de un brazo y su varianza robusta a la correlación dentro de cada grupo (linealización)."""
    df = pd.DataFrame({"y": y, "c": cluster}).groupby("c").y.agg(["sum", "size"])
    n, k = df["size"].sum(), len(df)
    mean = df["sum"].sum() / n
    var = k / (k - 1) * ((df["sum"] - mean * df["size"]) ** 2).sum() / n ** 2
    return mean, var


def compare(y: np.ndarray, arm: np.ndarray, cluster: np.ndarray, control: str, clustered: bool = True) -> pd.DataFrame:
    """Diferencia de tasas de cada brazo contra el control, con error estándar por grupos (o ingenuo)."""
    stats_by_arm = {}
    for a in pd.unique(arm):
        m = arm == a
        if clustered:
            stats_by_arm[a] = _cluster_mean(y[m], cluster[m])
        else:
            stats_by_arm[a] = (y[m].mean(), y[m].var(ddof=1) / m.sum())
    mc, vc = stats_by_arm[control]
    rows = []
    for a, (ma, va) in stats_by_arm.items():
        if a == control:
            continue
        se = np.sqrt(va + vc)
        z = (ma - mc) / se
        rows.append({"brazo": a, "tasa": ma, "tasa_control": mc, "diferencia": ma - mc, "efecto_relativo": (ma - mc) / mc,
                     "se": se, "z": z, "p": 2 * stats.norm.sf(abs(z)), "var_brazo": va, "var_control": vc})
    return pd.DataFrame(rows).set_index("brazo")


def holm(p: pd.Series) -> pd.Series:
    """p-values ajustados por Holm (controla la tasa de error de la familia, sin supuestos de dependencia)."""
    order = p.sort_values()
    m = len(p)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(order.to_numpy())])
    return pd.Series(np.minimum(adj, 1), index=order.index).reindex(p.index)


def dunnett(res: pd.DataFrame, seed: int = 0) -> pd.Series:
    """p-values de Dunnett (todos contra un control): usan que las comparaciones comparten el mismo control.

    Las z están correlacionadas porque todas restan la misma media del control; Dunnett aprovecha esa
    correlación y por eso tiene más potencia que Holm o Bonferroni. Se calcula con la normal multivariada.
    """
    v = res.var_brazo.to_numpy() + res.var_control.to_numpy()
    corr = np.sqrt(np.outer(res.var_control, res.var_control)) / np.sqrt(np.outer(v, v))
    np.fill_diagonal(corr, 1)
    mvn = stats.multivariate_normal(np.zeros(len(res)), corr, seed=seed)
    out = {}
    for name, z in res.z.items():
        b = np.full(len(res), abs(z))
        out[name] = 1 - mvn.cdf(b, lower_limit=-b)
    return pd.Series(out)


def omnibus(y: np.ndarray, arm: np.ndarray, cluster: np.ndarray) -> dict:
    """Test global de igualdad de tasas entre todos los brazos (Wald con varianzas por grupos)."""
    arms = list(pd.unique(arm))
    est = [_cluster_mean(y[arm == a], cluster[arm == a]) for a in arms]
    means, var = np.array([e[0] for e in est]), np.array([e[1] for e in est])
    C = np.column_stack([-np.ones(len(arms) - 1), np.eye(len(arms) - 1)])   # brazo j − brazo 1
    d = C @ means
    W = float(d @ np.linalg.solve(C @ np.diag(var) @ C.T, d))
    return {"wald": W, "gl": len(arms) - 1, "p": float(stats.chi2.sf(W, len(arms) - 1))}


# ------------------------------------------------------------------------------------- revisiones intermedias
def obf_bounds(looks: int, alpha: float = 0.05, sims: int = 200_000, seed: int = 0) -> np.ndarray:
    """Fronteras tipo O'Brien–Fleming para `looks` revisiones igualmente espaciadas en información.

    z_k = c·√(K/k): muy exigentes al principio y cerca de 1,96 al final. c se calibra por simulación para que
    la probabilidad de cruzar alguna frontera bajo la hipótesis nula sea exactamente alpha.
    """
    rng = np.random.default_rng(seed)
    inc = rng.standard_normal((sims, looks))
    z = np.cumsum(inc, axis=1) / np.sqrt(np.arange(1, looks + 1))   # z acumulado con información k/K
    shape = np.sqrt(looks / np.arange(1, looks + 1))
    lo, hi = 1.0, 6.0
    for _ in range(60):
        c = (lo + hi) / 2
        crossed = (np.abs(z) >= c * shape).any(axis=1).mean()
        lo, hi = (c, hi) if crossed > alpha else (lo, c)
    return c * shape
