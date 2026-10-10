"""Limpieza y construcción del panel mensual producto (HS6) × país.

Uso:
    python -m comercio_chile.clean

Salidas:
    data/processed/panel_{exportaciones,importaciones}.parquet
        1 fila = (periodo, segmento, hs6, pais): valor_usd, cantidad, peso_kg (solo DUS), n_items, n_decl,
        n_empresas, pseudo_pais, capitulo
    references/codigos/hs6_etiquetas_2025.csv   etiqueta legible por producto (texto de las declaraciones)

Reglas (justificadas en docs/PROYECTO.md §5):
  R1  Segmento DUS: bienes (200, 210), servicios (202), rancho (211).
  R2  Un ítem truncado en origen que es el único nulo de su declaración se reconstruye exactamente como
      total de encabezado - suma del resto. Su producto no es recuperable: hs6 = "000000".
  R3  Los valores negativos (ajustes) se conservan.
  R4  Los pseudo-países (códigos >= 900 y 997) se conservan, marcados.
  R5  n_empresas se cuenta dentro del mes: los IDs de empresa se renumeran cada mes.
"""
import pandas as pd

from .config import PROCESSED, REFERENCES
from .data import hs6_labels, load
from .references import load as load_reference
from .schema import FIELDS, GOODS_OPERATIONS

SEGMENTS_DUS = {**{op: "bienes" for op in GOODS_OPERATIONS}, "202": "servicios", "211": "rancho"}


def impute_truncated(df: pd.DataFrame, f: dict) -> dict:
    """R2 (modifica `df`). Devuelve un resumen de lo imputado."""
    g = df.groupby(["periodo", f["decl"]])
    n_null = g[f["item_value"]].transform(lambda x: x.isna().sum())
    known = g[f["item_value"]].transform("sum")
    target = df[f["item_value"]].isna() & (n_null == 1)
    df.loc[target, f["item_value"]] = df.loc[target, f["header_value"]] - known[target]
    return {"items_nulos": int(df[f["item_value"]].isna().sum() + target.sum()),
            "imputados": int(target.sum()),
            "valor_imputado_usd": float(df.loc[target, f["item_value"]].sum())}


def build_panel(flow: str, periods: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Panel mensual hs6 × país. Con `periods` ('AAAA-MM') se construyen solo esos meses: cada mes es
    independiente, así que un pipeline incremental puede agregar meses nuevos sin reprocesar la historia."""
    f = FIELDS[flow]
    extra = ["TIPOOPERACION", "PESOBRUTOITEM"] if flow == "exportaciones" else []
    df = load(flow, [f[k] for k in ("decl", "item_value", "header_value", "hs", "partner", "trader", "qty")] + extra,
              periods=periods)

    stats = impute_truncated(df, f)
    df = df.dropna(subset=[f["item_value"]])  # más de un ítem nulo en la declaración: no imputable

    df["segmento"] = df["TIPOOPERACION"].map(SEGMENTS_DUS).fillna("otros") if flow == "exportaciones" else "bienes"
    df["hs6"] = df[f["hs"]].str[:6].fillna("000000")
    df = df.rename(columns={f["partner"]: "pais"})

    agg = dict(
        valor_usd=(f["item_value"], "sum"),
        cantidad=(f["qty"], "sum"),
        n_items=(f["item_value"], "size"),
        n_decl=(f["decl"], "nunique"),
        n_empresas=(f["trader"], "nunique"),
    )
    if flow == "exportaciones":
        agg["peso_kg"] = ("PESOBRUTOITEM", "sum")
    panel = df.groupby(["periodo", "segmento", "hs6", "pais"], observed=True).agg(**agg).reset_index()

    pseudo = set(load_reference("paises").query("pseudo_pais")["codigo"]) | {"997"}
    panel["pseudo_pais"] = panel["pais"].isin(pseudo)
    panel["capitulo"] = panel["hs6"].str[:2]
    stats.update(filas_items=len(df), filas_panel=len(panel), valor_total_usd=float(panel["valor_usd"].sum()))
    return panel, stats


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for flow in FIELDS:
        panel, stats = build_panel(flow)
        panel.to_parquet(PROCESSED / f"panel_{flow}.parquet", index=False)
        print(flow, {k: (round(v / 1e6, 2) if "usd" in k else v) for k, v in stats.items()})
    labels = hs6_labels([str(p) for p in pd.period_range("2025-01", "2025-12", freq="M")])
    labels.to_csv(REFERENCES / "codigos" / "hs6_etiquetas_2025.csv")


if __name__ == "__main__":
    main()
