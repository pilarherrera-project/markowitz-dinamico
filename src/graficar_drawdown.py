"""
src/graficar_drawdown.py

Panel 1: como cambia el PUESTO de cada estrategia segun la vara de riesgo.
         Si las lineas se cruzan, elegir la metrica cambia la respuesta.
Panel 2: tiempo bajo el agua, la dimension que ninguna metrica clasica
         captura y la que en la practica hace que la gente abandone.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "results"

COL = {"1/N": "#6b7280", "min_varianza": "#2563eb", "max_sharpe": "#d97706",
       "momentum_3": "#059669", "momentum_5": "#7c3aed", "min_cvar": "#be123c",}

tabla = pd.read_csv(RESULTADOS / "metricas_riesgo.csv", index_col=0)
medidas = ["sharpe", "sortino", "calmar", "martin"]
rank = tabla[medidas].rank(ascending=False)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 6.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor("white")

# ---- panel 1: slopegraph de puestos ----
x = list(range(len(medidas)))
for e in rank.index:
    c = COL.get(e, "#111827")
    a1.plot(x, rank.loc[e].values, marker="o", ms=10, lw=2.4, color=c, zorder=3)
    a1.text(-0.14, rank.loc[e, medidas[0]], e, ha="right", va="center",
            fontsize=10, color=c, fontweight="bold")
    a1.text(x[-1] + 0.14, rank.loc[e, medidas[-1]], e, ha="left", va="center",
            fontsize=10, color=c, fontweight="bold")

a1.invert_yaxis()
a1.set_xlim(-1.6, x[-1] + 1.6)
a1.set_xticks(x)
a1.set_xticklabels(["SHARPE\n(volatilidad)", "SORTINO\n(semidesvio)",
                    "CALMAR\n(max drawdown)", "MARTIN\n(ulcer index)"], fontsize=9.5)
a1.set_yticks(range(1, len(rank) + 1))
a1.set_ylabel("puesto (1 = mejor)")
a1.set_title("1  |  Si las lineas se cruzan, la vara cambia la respuesta",
             loc="left", fontsize=11.5, pad=12)
a1.grid(True, axis="y", alpha=0.25)

# ---- panel 2: tiempo bajo el agua ----
orden = tabla.sort_values("bajo_agua_%").index
y = range(len(orden))
a2.barh(list(y), tabla.loc[orden, "bajo_agua_%"], height=0.58,
        color=[COL.get(e, "#111827") for e in orden], alpha=0.9)
for i, e in enumerate(orden):
    a2.text(tabla.loc[e, "bajo_agua_%"] + 0.8, i,
            f"{tabla.loc[e, 'bajo_agua_%']:.1f} %   "
            f"(racha mas larga: {tabla.loc[e, 'racha_anios']:.1f} anios)",
            va="center", fontsize=9.5)
a2.set_yticks(list(y))
a2.set_yticklabels(orden, fontsize=10)
a2.set_xlabel("% del tiempo por debajo del maximo historico")
a2.set_xlim(0, 118)
a2.set_title("2  |  Cuanto tiempo estuviste abajo\n     (lo que hace que un inversor real abandone)",
             loc="left", fontsize=11.5, pad=12)
a2.grid(True, axis="x", alpha=0.25)

fig.tight_layout(w_pad=2.5)
salida = RESULTADOS / "metricas_riesgo.png"
fig.savefig(salida, dpi=150, facecolor="white")
plt.close(fig)
print(f"Guardado en {salida}")