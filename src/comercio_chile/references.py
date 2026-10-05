"""Tablas de códigos oficiales de Aduana (Compendio de Normas, Anexo 51) -> CSV.

Uso:
    python -m comercio_chile.references

Genera en references/codigos/: aduanas, tipos_operacion, paises, vias_transporte,
regiones y puertos. Los códigos se normalizan sin ceros a la izquierda ("03" -> "3"), como aparecen en
los datos, para que el join sea exacto.

Las tablas se ubican por el título que las precede (ej. "Anexo 51-9"), no por su posición
en la página, para que el script siga funcionando si Aduana agrega tablas nuevas.
"""
import re
from html.parser import HTMLParser

import pandas as pd
import requests

from .config import REFERENCES

ANEXO_51_URL = "https://www.aduana.cl/compendio-de-normas-anexo-51/aduana/2009-11-19/163937.html"
CACHE = REFERENCES / "raw" / "anexo51.html"
OUT = REFERENCES / "codigos"

# Códigos de "país" que no representan un país socio real (Anexo 51-9 y 51-10):
# zonas francas, ventas a naves, mercancía nacional, orígenes varios / no precisados.
PSEUDO_COUNTRY_MIN_CODE = 900


class _TableParser(HTMLParser):
    """Extrae tablas de primer nivel junto con el texto que las precede."""

    def __init__(self):
        super().__init__()
        self.tables, self._text = [], []
        self._rows = self._row = self._cell = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._rows is not None:
            if any(self._row):
                self._rows.append(self._row)
            self._row = None
        elif tag == "table":
            if self._depth == 1:
                self.tables.append({"before": " ".join(self._text[-15:]), "rows": self._rows})
                self._rows = None
            self._depth -= 1

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)
        elif data.strip() and self._depth == 0:
            self._text.append(data.strip())


def normalize_code(s: pd.Series) -> pd.Series:
    """'03' -> '3'; deja intactos los valores no numéricos."""
    digits = s.astype("string").str.extract(r"^\s*(\d+)", expand=False)
    return digits.str.lstrip("0").replace("", "0")


def _find(tables, before: str, header_has: str):
    for t in tables:
        if before.lower() in t["before"].lower() and t["rows"] and header_has.lower() in " ".join(t["rows"][0]).lower():
            return t["rows"]
    raise LookupError(f"Tabla no encontrada: {before!r} / {header_has!r}")


def _simple(rows, name_idx: int, code_idx: int, **extra) -> pd.DataFrame:
    body = [r for r in rows[1:] if len(r) > max(name_idx, code_idx)]
    df = pd.DataFrame({"codigo": [r[code_idx] for r in body], "glosa": [r[name_idx] for r in body]})
    df["codigo"] = normalize_code(df["codigo"])
    return df.dropna(subset=["codigo"]).assign(**extra)


def _ports(rows) -> pd.DataFrame:
    """La tabla de puertos intercala filas-título de continente ('AMÉRICA DEL NORTE') y país ('CANADÁ:')."""
    out, continente, pais = [], None, None
    for r in rows:
        if len(r) == 1 or not re.match(r"^\s*\d", r[-1] or ""):
            label = r[0].strip()
            if label.endswith(":"):
                pais = label.rstrip(":").strip()
            elif label:
                continente, pais = label, None
            continue
        out.append({"codigo": r[-1], "glosa": r[0], "pais": pais, "continente": continente})
    df = pd.DataFrame(out)
    df["codigo"] = normalize_code(df["codigo"])
    return df


def build() -> dict[str, pd.DataFrame]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(ANEXO_51_URL, timeout=60)
        resp.raise_for_status()
        CACHE.write_bytes(resp.content)
    parser = _TableParser()
    parser.feed(CACHE.read_text(encoding="utf-8"))
    t = parser.tables

    paises = _simple(_find(t, "Anexo 51-9", "NOMBRE"), 0, 1)
    paises["pseudo_pais"] = paises["codigo"].astype(int) >= PSEUDO_COUNTRY_MIN_CODE
    tipos = pd.concat([
        _simple(_find(t, "Anexo 51-2", "GLOSA A CONSIGNAR"), 0, 2, flujo="importaciones"),
        _simple(_find(t, "OPERACIONES DE SALIDA", "TIPO DE OPERACI"), 0, 1, flujo="exportaciones"),
    ])
    return {
        "aduanas": _simple(_find(t, "Anexo 51-1", "ADUANAS"), 0, 1),
        "tipos_operacion": tipos,
        "paises": paises,
        "vias_transporte": _simple(_find(t, "Anexo 51-13", "VÍA DE TRANSPORTE"), 0, 1),
        "regiones": _simple(_find(t, "Anexo 51-44", "NOMBRE REGION"), 0, 1),
        "puertos": _ports(_find(t, "Anexo 51-11", "AMÉRICA")),
    }


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"{name}.csv", dtype={"codigo": str})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in build().items():
        # Duplicados conocidos: 902 (rancho, dos redacciones) y dos puertos chilenos (824, 900).
        # Se conserva la primera aparición para que el join no multiplique filas.
        dupes = df["codigo"].duplicated().sum()
        df = df.drop_duplicates("codigo", keep="first")
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
        print(f"{name:16s} {len(df):4d} códigos" + (f" ({dupes} duplicados)" if dupes else ""))


if __name__ == "__main__":
    main()
