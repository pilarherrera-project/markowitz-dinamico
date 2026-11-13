"""
src/graficar_barrido.py

Lee results/barrido_deriva.csv y dibuja el Sharpe neto en funcion del
umbral de deriva, una linea por nivel de costo, un panel por estrategia.

Como leerlo: si la linea SUBE al moverse hacia la derecha, rebalancear
menos MEJORA el resultado. El punto mas alto de cada linea es el umbral
optimo para ese costo. El extremo izquierdo (umbral 0) es el rebalanceo
mensual de siempre.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "results"

tabla = pd.read_csv(RESULTADOS / "barrido_deriva.csv")
COSTOS = [0, 5, 10, 20, 50]
COLORES = ["#94a3b8", "#0072B2", "#009E73", "#E69F00", "#D55E00"]

estrategias = list(tabla["estrategia"].unique())
n = len(estrategias)
fig, ejes = plt.subplots(1, n, figsize=(4.3 * n, 5.0))
if n == 1:
    ejes = [ejes]

for ax, nombre in zip(ejes, estrategias):
    sub = tabla[tabla["estrategia"] == nombre].sort_values("umbral")
    for c, color in zip(COSTOS, COLORES):
        ax.plot(sub["umbral"], sub[f"sharpe_{c}pb"], marker="o", ms=4.5,
                color=color, lw=1.7, label=f"{c} pb")
    # marcar el mejor punto de la curva de 5 pb
    s5 = sub.loc[sub["sharpe_5pb"].idxmax()]
    ax.scatter([s5["umbral"]], [s5["sharpe_5pb"]], s=170, facecolor="none",
               edgecolor="#111827", lw=1.8, zorder=5)
    ax.set_title(nombre, fontsize=11.5)
    ax.set_xlabel("umbral de deriva  (suma |w_opt - w_actual|)")
    ax.grid(True, alpha=0.25)

ejes[0].set_ylabel("Sharpe neto de costos")
ejes[-1].legend(frameon=False, fontsize=9, title="costo", title_fontsize=9,
                loc="lower left")

fig.suptitle("Cuanto se puede dejar de rebalancear antes de perder Sharpe",
             fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
salida = RESULTADOS / "barrido_deriva.png"
fig.savefig(salida, dpi=150)
plt.close(fig)
print(f"Guardado en {salida}")