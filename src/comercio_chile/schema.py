"""Esquemas de los archivos de Aduana: DUS (exportaciones) y DIN (importaciones).

Los nombres y el orden de columnas se leen del diccionario oficial (hoja `titulos` de
references/diccionario-aduana-dus-din-v2.0.xlsx). Aquí solo se declaran las decisiones de tipado:
fechas DDMMYYYY ("00000000" = sin fecha), montos con coma decimal -> float, y todo lo demás como texto
(los códigos son nominales y el código arancelario conserva sus ceros a la izquierda).

Cada fila es un ítem de una declaración. Los montos de encabezado (TOTALVALORFOB, CIF, ...) se repiten en
cada ítem, por lo que los agregados se calculan siempre con el valor a nivel ítem (`item_value`).
"""
from functools import lru_cache

import openpyxl

from .config import DICTIONARY_XLSX

SHEET_BLOCKS = {"exportaciones": "DUS", "importaciones": "DIN"}

# Tipos de operación DUS que corresponden a exportación de bienes (202 = servicios, 211 = rancho de naves)
GOODS_OPERATIONS = {"200", "210"}

# Columnas con rol semántico en el resto del código
FIELDS = {
    "exportaciones": dict(date="FECHAACEPT", decl="NUMEROIDENT", item_value="FOBUS", header_value="TOTALVALORFOB",
                          hs="CODIGOARANCEL", partner="PAISDESTINO", trader="NRO_EXPORTADOR", qty="CANTIDADMERCANCIA"),
    "importaciones": dict(date="FECACEP", decl="NUMENCRIPTADO", item_value="CIF_ITEM", header_value="CIF",
                          hs="ARANC_NAC", partner="PA_ORIG", trader="NUM_UNICO_IMPORTADOR", qty="CANT_MERC"),
}

DATE_COLS = {
    "exportaciones": ["FECHAACEPT", "FECHAINFORMEEXP", "FECHADOCTOCANCELA"],
    "importaciones": [
        "FECVENCI", "FEC_RS", "FEC_ALMAC", "FECRETIRO", "FECTRA", "FECACEP", "FEC_MANIF",
        "FEC_CONOC", "FEC_AUT", "FEC_DI", *[f"FEC_50{i}" for i in range(1, 8)],
    ],
}

NUMERIC_COLS = {
    "exportaciones": [
        "PORCENTAJEEXPPPAL", "PORCENTAJEEXPSECUNDARIO", "VALORCLAUSULAVENTA",
        "COMISIONESEXTERIOR", "OTROSGASTOS", "VALORLIQUIDORETORNO", "TOTALITEM", "TOTALBULTOS",
        "PESOBRUTOTOTAL", "TOTALVALORFOB", "VALORFLETE", "VALORSEGURO", "VALORCIF",
        "PESOBRUTOCANCELA", "TOTALBULTOSCANCELA", "NUMEROITEM", "CANTIDADMERCANCIA",
        "FOBUNITARIO", "FOBUS", "PESOBRUTOITEM",
    ],
    "importaciones": [
        "TOTINSUM", "NUM_SEC", "NUMDIAS", "VALEXFAB", "MONGASFOB", "TOT_ITEMS", "FOB", "TOT_HOJAS",
        "FLETE", "TOT_BULTOS", "SEGURO", "TOT_PESO", "CIF",
        *[f"CANT_BUL{i}" for i in range(1, 9)],
        "MON_OTRO", *[f"MON_OTR{i}" for i in range(1, 8)], "MON_178", "MON_191",
        *[f"VAL_60{i}" for i in range(1, 8)], "TASA", "MON_699", "MON_199",
        "NUMITEM", "AJU_ITEM", "CANT_MERC", "MERMAS", "PRE_UNIT", "CIF_ITEM", "ADVAL_ALA", "VALAD",
        *[f"OTRO{i}" for i in range(1, 5)], *[f"VAL{i}" for i in range(1, 5)],
    ],
}


@lru_cache
def columns(flow: str) -> list[str]:
    """Nombres de columnas, en orden, según el diccionario oficial."""
    ws = openpyxl.load_workbook(DICTIONARY_XLSX, read_only=True, data_only=True)["titulos"]
    rows = [[str(c).strip() for c in row if c is not None and str(c).strip()] for row in ws.iter_rows(values_only=True)]
    rows = [r for r in rows if r]
    for title, names in zip(rows, rows[1:], strict=False):  # cada título va seguido de sus nombres
        if title == [SHEET_BLOCKS[flow]]:
            return [n.replace("-", "_") for n in names]
    raise KeyError(f"Bloque {SHEET_BLOCKS[flow]!r} no encontrado en {DICTIONARY_XLSX.name}")
