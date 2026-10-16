"""Corre el backtest walk-forward para varias estrategias y las compara."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path

from backtest import backtest
from estimators import muestral
from optimizer import min_varianza, max_sharpe
from metrics import resumen

RAIZ = Path(__file__).resolve().parent.parent

# ---------- Datos ----------
precios = pd.read_parquet(RAIZ / "data" / "prices.parquet")
retornos = precios.pct_change().dropna()
idx = retornos.index

# ---------- Fechas de decision: ultimo dia habil de cada mes ----------
serie_fechas = pd.Series(idx, index=idx)
fechas_decision = pd.DatetimeIndex(
    serie_fechas.groupby([idx.year, idx.month]).last().values
)

print(f"Datos: {retornos.shape[0]} dias, {retornos.shape[1]} activos")
print(f"Fechas de decision: {len(fechas_decision)} (mensuales)")
print("Ventana de estimacion: 252 dias | Costo: 5 pb | Rebalanceo: mensual")
print()


# ---------- Estrategias ----------
def peso_igual(mu, Sigma):
    return np.repeat(1 / len(mu), len(mu))


ESTRATEGIAS = {
    "1/N": peso_igual,
    "min_varianza": lambda mu, S: min_varianza(S, w_max=0.35),
    "max_sharpe": lambda mu, S: max_sharpe(mu, S, rf=0.02, w_max=0.35),
}

# Rebalanceo por calendario: siempre que toca fecha, opera
def siempre(w_actual, w_opt, ctx):
    return True


# ---------- Correr ----------
filas, series = {}, {}

for nombre, optimizar in ESTRATEGIAS.items():
    print(f"Corriendo {nombre}...")
    ret, pesos, costos = backtest(
        retornos, fechas_decision,
        estimar=muestral,
        optimizar=optimizar,
        decidir=siempre,
        ventana=252,
        costo_bps=5.0,
    )
    filas[nombre] = resumen(ret, costos, rf=0.02)
    series[nombre] = ret

tabla = pd.DataFrame(filas).T

print()
print("=" * 80)
print("RESULTADOS FUERA DE MUESTRA")
print("=" * 80)
print(tabla.round(3).to_string())
print()
print(f"Periodo evaluado: {series['1/N'].index[0].date()} a {series['1/N'].index[-1].date()}")
print()

# ---------- Guardar ----------
RESULTADOS = RAIZ / "results"
RESULTADOS.mkdir(exist_ok=True)
pd.DataFrame(series).to_parquet(RESULTADOS / "retornos_estrategias.parquet")
tabla.to_csv(RESULTADOS / "tabla_estrategias.csv")
print(f"Guardado en {RESULTADOS}")