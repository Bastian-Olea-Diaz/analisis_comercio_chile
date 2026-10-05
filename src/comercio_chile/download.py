"""Descarga los registros mensuales de Aduana (2024–2026) desde datos.gob.cl (API CKAN).

Uso:
    python -m comercio_chile.download

Solo se descarga el archivo principal de cada mes (DUS / DIN); Bultos y Documentos de Transporte no se
usan. Los meses partidos en volúmenes (.part01.rar, ...) se descargan completos. Es idempotente: un archivo
que ya existe con el tamaño publicado no se vuelve a descargar. data/raw/manifest.csv registra URL, tamaño y
SHA-256 de cada archivo.
"""
import csv
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import CKAN_API, CKAN_DATASETS, MESES, RAW

# ej.: importaciones-enero-2024.part01.rar | exportaciones-diciembre-2025.rar
FILE_RE = re.compile(
    r"^(?P<flow>exportaciones|importaciones)-(?P<mes>[a-z]+)-(?P<year>\d{4})"
    r"(?:\.part(?P<part>\d+))?\.(?:rar|zip)$"
)


def list_resources(flow: str, year: int) -> list[dict]:
    """Recursos mensuales principales (sin bultos/documentos) de un dataset CKAN."""
    resp = requests.get(CKAN_API, params={"id": CKAN_DATASETS[(flow, year)]}, timeout=60)
    resp.raise_for_status()
    out = []
    for res in resp.json()["result"]["resources"]:
        fname = Path(urlparse(res["url"]).path).name.lower()
        m = FILE_RE.match(fname)
        if not m or m["mes"] not in MESES:
            continue  # metadata, bultos, documentos de transporte o .txt sueltos
        out.append({
            "flow": flow, "year": year, "month": MESES[m["mes"]], "part": int(m["part"] or 1),
            "filename": fname, "url": res["url"], "expected_size": res.get("size"),
        })
    return sorted(out, key=lambda r: (r["month"], r["part"]))


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(res: dict) -> Path:
    dest = RAW / res["flow"] / str(res["year"]) / res["filename"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and res["expected_size"] and dest.stat().st_size == int(res["expected_size"]):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(res["url"], stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    tmp.replace(dest)
    return dest


def main() -> None:
    manifest = []
    for flow, year in CKAN_DATASETS:
        resources = list_resources(flow, year)
        months = sorted({r["month"] for r in resources})
        print(f"{flow} {year}: {len(resources)} archivos, meses {months}")
        for res in resources:
            path = download(res)
            size = path.stat().st_size
            if res["expected_size"] and size != int(res["expected_size"]):
                print(f"  ADVERTENCIA tamaño distinto al declarado: {path.name} "
                      f"({size} vs {res['expected_size']})")
            manifest.append({**res, "size": size, "sha256": sha256(path),
                             "path": path.relative_to(RAW).as_posix()})
            print(f"  ok {path.name} ({size / 1e6:.1f} MB)")

    with (RAW / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Manifiesto: {len(manifest)} archivos, "
          f"{sum(m['size'] for m in manifest) / 1e6:.0f} MB -> {RAW / 'manifest.csv'}")


if __name__ == "__main__":
    main()
