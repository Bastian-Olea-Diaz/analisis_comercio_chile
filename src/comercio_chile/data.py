"""Acceso a la capa interim (parquet por mes) y a las tablas de referencia."""
import pandas as pd

from .config import INTERIM, REFERENCES
from .references import load as load_reference


def load(flow: str, columns: list[str] | None = None, periods: list[str] | None = None) -> pd.DataFrame:
    """Concatena los parquet mensuales de `flow` y agrega `periodo` (Period mensual).

    periods: lista de 'AAAA-MM' para leer solo esos meses.
    """
    frames = [pd.read_parquet(f, columns=columns).assign(periodo=pd.Period(f.stem, "M"))
              for f in sorted((INTERIM / flow).glob("*.parquet")) if not periods or f.stem in periods]
    return pd.concat(frames, ignore_index=True)


def country_names() -> pd.Series:
    """código de país -> nombre (Aduana, Anexo 51-9)."""
    return load_reference("paises").set_index("codigo")["glosa"]


def hs_chapters() -> pd.Series:
    """capítulo SA (2 dígitos) -> nombre corto."""
    return pd.read_csv(REFERENCES / "codigos" / "sa_capitulos.csv", dtype=str).set_index("capitulo")["glosa_corta"]


def _description(nombre: pd.Series, n_words: int = 4) -> pd.Series:
    """Descripción del producto a partir del campo NOMBRE de la DUS.

    Conviven dos formatos: "CODIGO-INTERNO ~ DESCRIPCION~ MARCA~..." y "DESCRIPCION~ATRIBUTO~...". Si el
    primer segmento es un solo token con dígitos o guiones (un código interno), se usa el segundo.
    """
    parts = nombre.fillna("").str.upper().str.split("~")
    first = parts.str[0].str.strip()
    is_code = ~first.str.contains(" ") & first.str.contains(r"[\d\-]")
    text = parts.str[1].where(is_code, first).fillna("")
    words = text.str.replace(r"[^A-ZÁÉÍÓÚÑ ]", " ", regex=True).str.split()
    return words.str[:n_words].str.join(" ")


def hs6_labels(periods: list[str] | None = None) -> pd.Series:
    """hs6 -> descripción más frecuente (ponderada por valor) en las exportaciones de bienes."""
    ex = load("exportaciones", ["CODIGOARANCEL", "NOMBRE", "FOBUS", "TIPOOPERACION"], periods=periods)
    ex = ex[ex.TIPOOPERACION == "200"].dropna(subset=["CODIGOARANCEL"])
    ex["hs6"] = ex.CODIGOARANCEL.str[:6]
    ex["desc"] = _description(ex.NOMBRE)
    ex = ex[ex.desc.str.len() > 2]
    best = ex.groupby(["hs6", "desc"]).FOBUS.sum().reset_index().sort_values("FOBUS", ascending=False)
    return best.drop_duplicates("hs6").set_index("hs6")["desc"].rename("etiqueta")
