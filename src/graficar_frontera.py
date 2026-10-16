"""Dibuja la frontera eficiente a partir del .npz y guarda el PNG."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")          # dibuja a archivo, sin abrir ventana
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

datos = np.load(RAIZ / "results" / "frontera.npz", allow_pickle=True)
vols, rets = datos["vols"], datos["rets"]
mu, Sigma = datos["mu"], datos["Sigma"]
w_minvar, w_tan = datos["w_minvar"], datos["w_tan"]
activos = datos["activos"]

v_min = np.sqrt(w_minvar @ Sigma @ w_minvar)
r_min = w_minvar @ mu
v_tan = np.sqrt(w_tan @ Sigma @ w_tan)
r_tan = w_tan @ mu

vol_activos = np.sqrt(np.diag(Sigma))

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(vols * 100, rets * 100, "-", lw=2, color="#0E6B5E",
        label="Frontera eficiente", zorder=2)

ax.scatter(vol_activos * 100, mu * 100, s=45, color="#8A99A0",
           edgecolor="white", linewidth=1, zorder=3)
for nombre, v, m in zip(activos, vol_activos, mu):
    ax.annotate(str(nombre), (v * 100, m * 100), fontsize=8,
                xytext=(5, -3), textcoords="offset points", color="#4A5956")

ax.scatter(v_min * 100, r_min * 100, s=140, marker="D", color="#2E5C86",
           edgecolor="white", linewidth=1.5, zorder=4, label="Minima varianza")
ax.scatter(v_tan * 100, r_tan * 100, s=170, marker="*", color="#9C5E22",
           edgecolor="white", linewidth=1.2, zorder=4, label="Maximo Sharpe")

ax.set_xlabel("Volatilidad anual (%)")
ax.set_ylabel("Retorno esperado anual (%)")
ax.set_title("Frontera eficiente - 14 ETFs, 2007-2026\n"
             "Estimada sobre toda la muestra (in-sample)", fontsize=12)
ax.grid(alpha=0.25, linewidth=0.6)
ax.legend(loc="lower right", frameon=False)

salida = RAIZ / "results" / "frontera_eficiente.png"
plt.savefig(salida, dpi=150, bbox_inches="tight")
print(f"Guardado en {salida}")