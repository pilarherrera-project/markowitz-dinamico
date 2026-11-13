"""
src/graficar_validacion.py

Dos paneles:
  izquierda  el umbral que la politica adaptativa eligio cada anio.
             Si salta erraticamente, la seleccion esta siguiendo ruido.
  derecha    Sharpe fuera de muestra de las cuatro politicas.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "results"

tabla = pd.read_csv(RESULTADOS / "validacion_rodante.csv")
umbrales = pd.read_csv(RESULTADOS / "umbrales_elegidos.csv", index_col=0)

C_EST = {"1/N": "#6b7280", "min_varianza": "#2563eb",
         "max_sharpe": "#d97706", "momentum_3": "#059669"}
C_POL = {"mensual": "#94a3b8", "adaptativa": "#0072B2",
         "fijo_insample": "#D55E00", "buy_and_hold": "#111827"}
ORDEN = ["mensual", "adaptativa", "fijo_insample", "buy_and_hold"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.8),
                             gridspec_kw={"width_ratios": [1, 1.15]})
fig.patch.set_facecolor("white")

# ---- panel 1: umbral elegido cada anio ----
for col in umbrales.columns:
    a1.step(umbrales.index, umbrales[col], where="mid", lw=2,
            marker="o", ms=4.5, color=C_EST.get(col), label=col)
a1.set_ylim(-0.1, 2.1)
a1.set_xlabel("anio evaluado")
a1.set_ylabel("umbral elegido (mirando los 5 anios previos)")
a1.set_title("1  |  Que eligio la politica adaptativa, anio a anio",
             loc="left", fontsize=11.5, pad=10)
a1.legend(frameon=False, fontsize=9)
a1.grid(True, alpha=0.25)

# ---- panel 2: Sharpe fuera de muestra ----
estrategias = list(umbrales.columns)
x = np.arange(len(estrategias))
ancho = 0.2
for i, pol in enumerate(ORDEN):
    sub = tabla[tabla["politica"] == pol].set_index("estrategia")
    vals = [sub.loc[e, "sharpe_5pb"] for e in estrategias]
    barras = a2.bar(x + (i - 1.5) * ancho, vals, ancho,
                    color=C_POL[pol], label=pol)
    for b, v in zip(barras, vals):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=8)

a2.set_xticks(x)
a2.set_xticklabels(estrategias)
a2.set_ylabel("Sharpe fuera de muestra (5 pb)")
a2.set_title("2  |  Resultado honesto: 2013-2026, mismo arranque",
             loc="left", fontsize=11.5, pad=10)
a2.legend(frameon=False, fontsize=9, ncol=2)
a2.grid(True, axis="y", alpha=0.25)

fig.tight_layout(w_pad=2.5)
salida = RESULTADOS / "validacion_rodante.png"
fig.savefig(salida, dpi=150, facecolor="white")
plt.close(fig)
print(f"Guardado en {salida}")