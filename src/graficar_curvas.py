"""
src/graficar_curvas.py

Lee results/retornos_estrategias.parquet y dibuja dos paneles:
  arriba  -> riqueza acumulada en escala logaritmica
  abajo   -> drawdown en el tiempo ("underwater plot")

No calcula estrategias: solo lee y dibuja. Toda la cuenta pesada
ya quedo hecha en correr_backtest.py. Por eso este script no crashea.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # dibuja sin abrir ventana: evita el crash del kernel
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RESULTS = RAIZ / "results"

COLORES = {
    "1/N": "#6b7280",
    "min_varianza": "#2563eb",
    "max_sharpe": "#d97706",
}

retornos = pd.read_parquet(RESULTS / "retornos_estrategias.parquet")
riqueza = (1 + retornos).cumprod()
drawdown = riqueza / riqueza.cummax() - 1

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 7.5), sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

for col in riqueza.columns:
    ax1.plot(riqueza.index, riqueza[col], label=col,
             color=COLORES.get(col), linewidth=1.5)

ax1.set_yscale("log")
ax1.set_ylabel("Riqueza acumulada (base 1, escala log)")
ax1.set_title("Markowitz dinamico | curvas de capital netas de costos (5 pb, rebalanceo mensual)")
ax1.legend(frameon=False, loc="upper left")
ax1.grid(True, which="both", alpha=0.25)

for col in drawdown.columns:
    ax2.fill_between(drawdown.index, drawdown[col] * 100, 0,
                     color=COLORES.get(col), alpha=0.20, linewidth=0)
    ax2.plot(drawdown.index, drawdown[col] * 100,
             color=COLORES.get(col), linewidth=1.0)

ax2.set_ylabel("Drawdown (%)")
ax2.grid(True, alpha=0.25)
ax2.axhline(0, color="black", linewidth=0.8)

fig.tight_layout()
salida = RESULTS / "curvas_capital.png"
fig.savefig(salida, dpi=150)
plt.close(fig)

# --- anatomia de la peor caida de cada estrategia ---
print()
print(f"{'estrategia':<16}{'pico':<13}{'valle':<13}{'profundidad':>13}   recuperacion")
for col in riqueza.columns:
    curva = riqueza[col]
    dd = curva / curva.cummax() - 1
    valle = dd.idxmin()
    pico = curva.loc[:valle].idxmax()
    nivel = curva.loc[pico]
    post = curva.loc[valle:]
    recuperado = post[post >= nivel]
    if len(recuperado):
        dias = (recuperado.index[0] - pico).days
        txt = f"{dias} dias ({dias/365.25:.1f} anios)"
    else:
        txt = "nunca recupero"
    print(f"{col:<16}{str(pico.date()):<13}{str(valle.date()):<13}{dd.min()*100:>12.2f}%   {txt}")

print()
print("Riqueza final:")
print(riqueza.iloc[-1].round(3).to_string())
print(f"\nGuardado en {salida}")