"""
src/graficar_pesos.py

Dibuja la composicion de cada cartera a lo largo de los 18 anos.

Pregunta que contesta: max_sharpe cerro 2022 en +4.29% mientras XLE
rendia +64.32%. Si venia cargada en energia, se ve en la banda rosa.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "results"

ESTRATEGIAS = ["1/N", "min_varianza", "max_sharpe", "momentum_3", "momentum_5", "min_cvar"]

GRUPOS = {
    "Bonos (IEF, TLT)":         ["IEF", "TLT"],
    "Oro (GLD)":                ["GLD"],
    "Energia (XLE)":            ["XLE"],
    "Internacional (EEM, EFA)": ["EEM", "EFA"],
    "Sectores US (resto)":      ["XLB", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
}

# paleta Okabe-Ito: distinguible tambien para daltonismo
COLORES = ["#0072B2", "#E69F00", "#CC79A7", "#009E73", "#999999"]


def agrupar(pesos):
    salida = {}
    for etiqueta, tickers in GRUPOS.items():
        presentes = [t for t in tickers if t in pesos.columns]
        salida[etiqueta] = pesos[presentes].sum(axis=1) if presentes else 0.0
    return pd.DataFrame(salida)


def leer(nombre):
    return pd.read_parquet(RESULTADOS / f"pesos_{nombre.replace('/', '-')}.parquet")


fig, ejes = plt.subplots(len(ESTRATEGIAS), 1,
                         figsize=(11, 3.3 * len(ESTRATEGIAS)), sharex=True)

for ax, nombre in zip(ejes, ESTRATEGIAS):
    g = agrupar(leer(nombre))
    ax.stackplot(g.index, [g[c].values for c in g.columns],
                 labels=list(g.columns), colors=COLORES, linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("peso")
    ax.set_title(nombre, loc="left", fontsize=11)
    ax.margins(x=0)

manejadores, etiquetas = ejes[0].get_legend_handles_labels()
fig.legend(manejadores, etiquetas, loc="upper center", ncol=5,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Composicion de cada cartera en el tiempo", y=1.0, fontsize=13)
fig.subplots_adjust(top=0.88)
fig.tight_layout(rect=[0, 0, 1, 0.92])

salida = RESULTADOS / "pesos_en_el_tiempo.png"
fig.savefig(salida, dpi=150)
plt.close(fig)

# ---- el test concreto ----
print("Peso en XLE al cierre de cada ano (%):")
tabla = {}
for nombre in ESTRATEGIAS:
    pesos = leer(nombre)
    if "XLE" in pesos.columns:
        xle = pesos["XLE"] * 100
        tabla[nombre] = xle.groupby(xle.index.year).last()
print(pd.DataFrame(tabla).round(1).to_string())

print(f"\nGuardado en {salida}")