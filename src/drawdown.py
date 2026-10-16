"""Maximo drawdown de cada activo del universo."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

precios = pd.read_parquet(RAIZ / "data" / "prices.parquet")
retornos = precios.pct_change().dropna()


def max_drawdown(ret):
    """Peor caida pico-a-valle de la serie."""
    curva = (1 + ret).cumprod()
    return (curva / curva.cummax() - 1).min()


def recuperacion_necesaria(dd):
    """Cuanto hay que subir para volver al pico, dado un drawdown dd."""
    return 1 / (1 + dd) - 1


tabla = pd.DataFrame({"max_drawdown_%": retornos.apply(max_drawdown) * 100})
tabla["recuperacion_%"] = tabla["max_drawdown_%"].apply(
    lambda d: recuperacion_necesaria(d / 100) * 100
)
tabla = tabla.sort_values("max_drawdown_%")

print()
print(tabla.round(1).to_string())
print()