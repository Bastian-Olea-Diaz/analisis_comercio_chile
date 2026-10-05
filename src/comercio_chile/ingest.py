"""Ingesta: archivos mensuales de Aduana (.rar/.zip con un .txt `;`-separado) -> Parquet tipado.

Uso:
    python -m comercio_chile.ingest            # procesa los meses que falten
    python -m comercio_chile.ingest --force    # reprocesa todo

Salida:
    data/interim/{flujo}/{AAAA}-{MM}.parquet    1 fila = 1 ítem de declaración
    data/interim/_ingest_log.csv               métricas de calidad por archivo

Todo se lee como texto y se tipa después con errors="coerce", contando cada valor que no se pudo
convertir: nada se descarta en silencio.
"""
import argparse
import csv
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from .config import INTERIM, RAW, unrar_executable
from .schema import DATE_COLS, FIELDS, NUMERIC_COLS, columns

NULL_DATE = "00000000"


@dataclass
class IngestStats:
    kind: str
    year: int
    month: int
    source: str
    rows: int = 0
    lines_joined: int = 0          # registros partidos por un salto de línea dentro de un campo
    lines_padded: int = 0          # registros truncados en origen (faltan campos finales)
    invalid_rows: int = 0          # filas irreparables (se descartan)
    invalid_samples: list = field(default_factory=list)
    numeric_coerced: int = 0
    date_coerced: int = 0
    rows_outside_month: int = 0    # fecha de aceptación fuera del mes del archivo


def repair_lines(raw: bytes, n_fields: int, stats: IngestStats) -> bytes:
    """Repara registros con un número de campos distinto al del esquema.

    - Un `\\r` suelto dentro de un texto (p. ej. "QASIM INTERNATIONAL\\r") se trata como fin de línea:
      se reemplaza por espacio.
    - Registro partido: tiene < n campos y unido a la siguiente línea suma exactamente n (el salto estaba
      dentro de un campo, por eso se pierde un separador).
    - Registro truncado: < n campos sin poder completarse. El corte ocurre al final del registro, así que
      los campos presentes conservan su posición; se rellenan los faltantes con vacío.
    """
    lines = raw.replace(b"\r\n", b"\n").replace(b"\r", b" ").split(b"\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        nf = line.count(b";") + 1
        if line and nf < n_fields and i + 1 < len(lines) and nf + lines[i + 1].count(b";") == n_fields:
            out.append(line + lines[i + 1])
            stats.lines_joined += 1
            i += 2
            continue
        if line and nf < n_fields:
            line += b";" * (n_fields - nf)
            stats.lines_padded += 1
        out.append(line)
        i += 1
    return b"\n".join(out)


def read_raw(path: Path, flow: str, stats: IngestStats) -> pd.DataFrame:
    """Lee un archivo `;`-separado sin encabezado y devuelve un DataFrame tipado."""
    cols = columns(flow)

    def on_invalid(row):
        stats.invalid_rows += 1
        if len(stats.invalid_samples) < 3:
            stats.invalid_samples.append(row.text[:200])
        return "skip"

    def parse(source):
        return pacsv.read_csv(
            source,
            read_options=pacsv.ReadOptions(column_names=cols, encoding="latin1", block_size=1 << 26),
            # las descripciones contienen comillas sueltas que no delimitan campos
            parse_options=pacsv.ParseOptions(delimiter=";", quote_char=False, invalid_row_handler=on_invalid),
            convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in cols}),
        )

    table = parse(path)
    if stats.invalid_rows:  # solo si hace falta: reparar y volver a leer
        stats.invalid_rows, stats.invalid_samples = 0, []
        table = parse(pa.BufferReader(repair_lines(path.read_bytes(), len(cols), stats)))
    df = table.to_pandas()
    stats.rows = len(df)

    for c in df.columns:
        df[c] = df[c].str.strip().replace("", pd.NA)

    for c in DATE_COLS[flow]:
        raw = df[c].where(df[c] != NULL_DATE)
        parsed = pd.to_datetime(raw, format="%d%m%Y", errors="coerce")
        stats.date_coerced += int((raw.notna() & parsed.isna()).sum())
        df[c] = parsed

    for c in NUMERIC_COLS[flow]:
        raw = df[c]
        parsed = pd.to_numeric(raw.str.replace(",", ".", regex=False), errors="coerce")
        stats.numeric_coerced += int((raw.notna() & parsed.isna()).sum())
        df[c] = parsed.astype("float64")

    date = df[FIELDS[flow]["date"]]
    in_month = (date.dt.year == stats.year) & (date.dt.month == stats.month)
    stats.rows_outside_month = int((date.notna() & ~in_month).sum())
    return df


def extract(archive: Path, workdir: Path) -> Path:
    """Descomprime un .zip o .rar (multivolumen: basta el .part01) y devuelve el único .txt."""
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(workdir)
    else:
        subprocess.run([unrar_executable(), "x", "-y", "-idq", str(archive), str(workdir) + "/"], check=True)
    txts = list(workdir.rglob("*.txt"))
    if len(txts) != 1:
        raise RuntimeError(f"{archive.name}: se esperaba 1 .txt y se encontraron {len(txts)}")
    return txts[0]


def sources() -> list[tuple[str, int, int, Path]]:
    """(flujo, año, mes, primer volumen del archivo) según data/raw/manifest.csv."""
    manifest = RAW / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError("Falta data/raw/manifest.csv: ejecuta primero `python -m comercio_chile.download`.")
    with manifest.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["part"]) == 1]
    return [(r["flow"], int(r["year"]), int(r["month"]), RAW / r["path"]) for r in rows]


def process(flow: str, year: int, month: int, archive: Path, force: bool) -> IngestStats | None:
    out = INTERIM / flow / f"{year}-{month:02d}.parquet"
    if out.exists() and not force:
        return None
    stats = IngestStats(flow, year, month, archive.name)
    with tempfile.TemporaryDirectory() as tmp:
        df = read_raw(extract(archive, Path(tmp)), flow, stats)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    print(f"{flow:14s} {year}-{month:02d}: {stats.rows:>8,} filas | unidas {stats.lines_joined} "
          f"| rellenadas {stats.lines_padded} | inválidas {stats.invalid_rows} "
          f"| no convertibles {stats.numeric_coerced + stats.date_coerced} | fuera de mes {stats.rows_outside_month}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log_path = INTERIM / "_ingest_log.csv"
    previous = pd.read_csv(log_path) if log_path.exists() and not args.force else pd.DataFrame()
    new = [s for job in sources() if (s := process(*job, args.force))]
    if new:
        log = pd.concat([previous, pd.DataFrame([asdict(s) for s in new])], ignore_index=True)
        log = log.drop_duplicates(["kind", "year", "month"], keep="last").sort_values(["kind", "year", "month"])
        log.to_csv(log_path, index=False)


if __name__ == "__main__":
    main()
