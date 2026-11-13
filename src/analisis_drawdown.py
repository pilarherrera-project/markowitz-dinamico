"""
src/analisis_drawdown.py

BLOQUE F2 -- la pregunta del profesor, medida.

"Markowitz toma los retornos basados en volatilidad, pero hay drawdowns
mas grandes en unos activos que en otros."

Pregunta concreta: si cambio la vara con la que mido el riesgo, CAMBIA
el ranking de las estrategias? Si no cambia, la volatilidad ya capturaba
lo que importaba. Si cambia, hay informacion que la varianza no ve.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import pandas as pd

from metricas_riesgo import (max_drawdown, recuperacion_necesaria, ulcer_index,
                             tiempo_bajo_agua, episodios_drawdown,
                             sortino, calmar, martin, cvar)
from metrics import sharpe, retorno_anual, vol_anual

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "results"
RF = 0.02

ret = pd.read_parquet(RESULTADOS / "retornos_estrategias.parquet")
print(f"{ret.shape[0]} dias, {ret.shape[1]} estrategias")
print(f"{ret.index[0].date()} a {ret.index[-1].date()}\n")

filas = {}
for col in ret.columns:
    s = ret[col]
    frac, racha = tiempo_bajo_agua(s)
    d = max_drawdown(s)
    filas[col] = {
        "retorno_%": retorno_anual(s) * 100,
        "vol_%": vol_anual(s) * 100,
        "max_dd_%": d * 100,
        "recuperar_%": recuperacion_necesaria(d) * 100,
        "ulcer": ulcer_index(s),
        "bajo_agua_%": frac * 100,
        "racha_anios": racha / 252,
        "cvar5_%": cvar(s, 0.05),
        "sharpe": sharpe(s, rf=RF),
        "sortino": sortino(s, rf=RF),
        "calmar": calmar(s),
        "martin": martin(s, rf=RF),
    }

tabla = pd.DataFrame(filas).T

print("=" * 96)
print("EL MISMO RIESGO, MEDIDO DE CINCO FORMAS")
print("=" * 96)
print(tabla[["retorno_%", "vol_%", "max_dd_%", "recuperar_%", "ulcer",
             "bajo_agua_%", "racha_anios", "cvar5_%"]].round(2).to_string())

print()
print("=" * 96)
print("LA PREGUNTA DEL PROFESOR: cambia el ranking segun la vara?")
print("=" * 96)
medidas = ["sharpe", "sortino", "calmar", "martin"]
print("\nValores (mas alto es mejor en las cuatro):")
print(tabla[medidas].round(3).to_string())

rank = tabla[medidas].rank(ascending=False).astype(int)
print("\nPuesto (1 = mejor):")
print(rank.to_string())

cambia = rank.nunique(axis=1) > 1
print(f"\nCambian de puesto segun la metrica: {int(cambia.sum())} de {len(rank)}")
for e in rank.index[cambia]:
    detalle = "  ".join(f"{m}:{rank.loc[e, m]}" for m in medidas)
    print(f"   {e:<15}{detalle}")
if not cambia.any():
    print("   NINGUNA. La volatilidad ya capturaba el orden correcto.")

print()
print("=" * 96)
print("LOS TRES PEORES EPISODIOS DE CADA ESTRATEGIA")
print("=" * 96)
cols_ep = ["pico", "valle", "fin", "profundidad_%", "recuperar_%",
           "dias_caida", "dias_recuperacion", "dias_bajo_agua", "recuperado"]
for col in ret.columns:
        ep = episodios_drawdown(ret[col]).head(3).copy()
    numericas = ep.select_dtypes("number").columns
    ep[numericas] = ep[numericas].round(1)
    print(f"\n--- {col} ---")
    print(ep[cols_ep].to_string(index=False))

tabla.to_csv(RESULTADOS / "metricas_riesgo.csv")
print(f"\nGuardado en {RESULTADOS / 'metricas_riesgo.csv'}")