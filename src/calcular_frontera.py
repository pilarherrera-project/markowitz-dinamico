"""
Calcula la frontera eficiente in-sample y guarda los resultados en disco.
Se corre una vez; el notebook despues solo lee el .npz y grafica.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path

from optimizer import min_varianza, max_sharpe, frontera

RAIZ = Path(__file__).resolve().parent.parent

precios = pd.read_parquet(RAIZ / "data" / "prices.parquet")
retornos = precios.pct_change().dropna()
activos = list(retornos.columns)

mu = retornos.mean().values * 252
Sigma = retornos.cov().values * 252

print("Calculando la frontera...")
vols, rets, pesos = frontera(mu, Sigma, n_puntos=40, w_max=0.35)

print("Calculando carteras de referencia...")
w_minvar = min_varianza(Sigma, w_max=0.35)
w_tan = max_sharpe(mu, Sigma, rf=0.02, w_max=0.35)

RESULTADOS = RAIZ / "results"
RESULTADOS.mkdir(exist_ok=True)

np.savez(
    RESULTADOS / "frontera.npz",
    vols=vols, rets=rets, pesos=pesos,
    mu=mu, Sigma=Sigma,
    w_minvar=w_minvar, w_tan=w_tan,
    activos=np.array(activos),
)

v_min = np.sqrt(w_minvar @ Sigma @ w_minvar)
v_tan = np.sqrt(w_tan @ Sigma @ w_tan)

print()
print(f"Puntos de la frontera: {len(vols)}")
print(f"Minima varianza:  vol {v_min*100:5.2f}%   retorno {w_minvar @ mu*100:5.2f}%")
print(f"Maximo Sharpe:    vol {v_tan*100:5.2f}%   retorno {w_tan @ mu*100:5.2f}%")
print()
print("Guardado en results/frontera.npz")