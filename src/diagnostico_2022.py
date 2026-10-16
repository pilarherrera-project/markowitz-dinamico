"""
src/diagnostico_2022.py

Pregunta: si el optimizador podia ver en diciembre de 2021 que la
correlacion XLK-TLT ya no era negativa, por que min_varianza igual
se comio -18.57% en 2022?

Hipotesis: lo que fallo no fue la correlacion estimada sino la
VOLATILIDAD estimada de los bonos.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
VENTANA = 252
CORTE = "2021-12-31"

precios = pd.read_parquet(RAIZ / "data" / "prices.parquet")
ret = precios.pct_change().dropna()

# ---------- 1) que tenia en pantalla el optimizador ----------
vista = ret.loc[:CORTE].iloc[-VENTANA:]
vol_estimada = vista.std() * np.sqrt(252) * 100

print(f"Ventana que vio el optimizador el {CORTE}:")
print(f"  {vista.index[0].date()} a {vista.index[-1].date()}  ({len(vista)} dias)\n")
print("Volatilidad anualizada estimada, de menor a mayor (%):")
print(vol_estimada.sort_values().round(2).to_string())

# ---------- 2) que paso realmente en 2022 ----------
ret22 = ret.loc["2022-01-01":"2022-12-31"]
vol_realizada = ret22.std() * np.sqrt(252) * 100
retorno22 = ((1 + ret22).prod() - 1) * 100

comparacion = pd.DataFrame({
    "vol_estimada_dic21": vol_estimada,
    "vol_realizada_2022": vol_realizada,
    "ratio": vol_realizada / vol_estimada,
    "retorno_2022_%": retorno22,
}).sort_values("ratio", ascending=False)

print("\n\nEstimado en dic-2021 vs realizado en 2022:")
print(comparacion.round(2).to_string())
print("\n(ratio > 1 = el activo fue MAS volatil de lo que el optimizador creia)")

# ---------- 3) drawdown dentro del ano calendario 2022 ----------
est = pd.read_parquet(RAIZ / "results" / "retornos_estrategias.parquet")
e22 = est.loc["2022-01-01":"2022-12-31"]
curva = (1 + e22).cumprod()
dd22 = (curva / curva.cummax() - 1).min() * 100

print("\n\nDrawdown DENTRO de 2022 (el pico se resetea el 1-ene-2022):")
print(dd22.round(2).sort_values().to_string())
print("\nRetorno total 2022 por estrategia (%):")
print((((1 + e22).prod() - 1) * 100).round(2).to_string())