"""Estilo gráfico común del proyecto (matplotlib).

Paleta categórica validada para daltonismo (orden fijo, nunca ciclado), marcas finas,
grilla recesiva y etiquetas directas. Todas las figuras del proyecto usan este módulo.
"""
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt

from .config import FIGURES

# Paleta categórica (orden = mecanismo de seguridad CVD; asignar en este orden)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")   # negativo, neutro, positivo
NEUTRAL = "#8a8984"                                # "Otros" / contexto
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

EXPORT = SERIES[0]   # exportaciones: azul en todo el proyecto
IMPORT = SERIES[1]   # importaciones: naranjo en todo el proyecto


def setup() -> None:
    # Tipografía: Franklin Gothic Medium (incluida en Windows). DejaVu Sans cubre los símbolos que le faltan
    # (∝, por ejemplo) y la reemplaza en sistemas sin ella. Como tiene un solo peso, la negrita se pide pero no
    # existe: se silencia el aviso y los títulos se distinguen por tamaño.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "font.family": ["Franklin Gothic Medium", "DejaVu Sans"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 20,
        "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 2, "lines.markersize": 5,
        "legend.frameon": False,
    })


def subtitle(ax, text: str) -> None:
    """Línea explicativa bajo el título (qué leer en el gráfico)."""
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9, color=TEXT_2, va="bottom")


def label_end(ax, x, y, text: str, color: str, dy: float = 0) -> None:
    """Etiqueta directa al final de una línea (reemplaza la leyenda)."""
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=color,
                fontsize=9, fontweight="bold", va="center")


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
