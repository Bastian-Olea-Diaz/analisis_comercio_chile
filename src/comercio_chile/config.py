"""Rutas y constantes del proyecto."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# COMERCIO_CHILE_DATA permite usar el paquete desde otro proyecto (p. ej. un pipeline en producción) con su
# propia carpeta de datos; por defecto se usa data/ dentro de este repositorio.
DATA = Path(os.environ.get("COMERCIO_CHILE_DATA", ROOT / "data"))
RAW = DATA / "raw"            # archivos descargados (.rar/.zip) + manifest.csv
INTERIM = DATA / "interim"    # parquet tipado, 1 fila = 1 ítem de declaración
PROCESSED = DATA / "processed"
REFERENCES = ROOT / "references"
FIGURES = ROOT / "reports" / "figures"

DICTIONARY_XLSX = REFERENCES / "diccionario-aduana-dus-din-v2.0.xlsx"

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

# Datasets del Servicio Nacional de Aduanas en datos.gob.cl (API CKAN)
CKAN_API = "https://datos.gob.cl/api/3/action/package_show"
CKAN_DATASETS = {
    ("exportaciones", 2024): "registro-de-exportaciones-2024",
    ("exportaciones", 2025): "registro-de-exportaciones-2025",
    ("exportaciones", 2026): "registro-de-exportacion-2026",
    ("importaciones", 2024): "registro-de-importacion-2024",
    ("importaciones", 2025): "registro-de-importacion-2025",
    ("importaciones", 2026): "registro-de-importacion-2026",
}

SEED = 42


def unrar_executable() -> str:
    """Ruta a UnRAR: variable de entorno UNRAR, luego el PATH, luego la instalación por defecto de WinRAR."""
    candidates = [os.environ.get("UNRAR"), shutil.which("unrar"), shutil.which("UnRAR"),
                  r"C:\Program Files\WinRAR\UnRAR.exe"]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("No se encontró UnRAR. Instálalo o define la variable de entorno UNRAR.")
