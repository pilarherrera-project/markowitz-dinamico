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
from optimizer_cvar import min_cvar, escenarios

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

# Cada estrategia es un par (estimar, optimizar).
# Casi todas usan 'muestral', que devuelve (mu, Sigma).
# min_cvar usa 'escenarios', que devuelve (mu, matriz de retornos crudos):
# el CVaR es una propiedad de la distribucion entera, no de sus dos
# primeros momentos. El motor no mira lo que hay adentro, solo lo pasa.
ESTRATEGIAS = {
    "1/N":          (muestral,   peso_igual),
    "min_varianza": (muestral,   lambda mu, S: min_varianza(S, w_max=0.35)),
    "max_sharpe":   (muestral,   lambda mu, S: max_sharpe(mu, S, rf=0.02, w_max=0.35)),
    "momentum_3":   (muestral,   momentum(3)),
    "momentum_5":   (muestral,   momentum(5)),
    "min_cvar":     (escenarios, lambda mu, R: min_cvar(R, beta=0.95, w_max=0.35)),
}

# Rebalanceo por calendario: siempre que toca fecha, opera
def siempre(w_actual, w_opt, ctx):
    return True


# ---------- Correr ----------
filas, series = {}, {}

for nombre, (estimar,optimizar) in ESTRATEGIAS.items():
    print(f"Corriendo {nombre}...")
    ret, pesos, costos = backtest(
        retornos, fechas_decision,
        estimar=estimar,
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
"""Corre el backtest walk-forward para varias estrategias y las compara.

Ademas de retornos y metricas, ahora guarda la SERIE DE PESOS de cada
estrategia. Hace falta para dos cosas:
  (a) auditar la composicion de las carteras (que tenia cada una, y cuando),
  (b) el bloque F: el disparador por deriva compara w_actual contra w_optimo,
      y sin la serie de pesos no hay con que medir deriva.
"""
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
RESULTADOS = RAIZ / "results"
RESULTADOS.mkdir(exist_ok=True)

# ---------- Datos ----------
precios = pd.read_parquet(RAIZ / "data" / "prices.parquet")
retornos = precios.pct_change().dropna()
idx = retornos.index
TICKERS = list(retornos.columns)

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
def momentum(k):
    """Top-k activos por mu estimada, equiponderados. Ignora Sigma por completo.

    Como mu = promedio de los ultimos 252 dias habiles x 252, ordenar por mu
    es exactamente una senal de momentum a 12 meses (Jegadeesh-Titman 1993).
    Si esto empata con max_sharpe, toda la maquinaria de Markowitz no esta
    agregando nada en este universo.
    """
    def elegir(mu, Sigma):
        w = np.zeros(len(mu))
        mejores = np.argsort(mu)[-k:]
        w[mejores] = 1 / k
        return w
    return elegir

ESTRATEGIAS = {
    "1/N": peso_igual,
    "min_varianza": lambda mu, S: min_varianza(S, w_max=0.35),
    "max_sharpe": lambda mu, S: max_sharpe(mu, S, rf=0.02, w_max=0.35),
    "momentum_3": momentum(3),
    "momentum_5": momentum(5),
}


# Rebalanceo por calendario: siempre que toca fecha, opera
def siempre(w_actual, w_opt, ctx):
    return True


def archivo_de(nombre):
    """'1/N' no puede ser nombre de archivo en Windows."""
    return nombre.replace("/", "-")


# ---------- Correr ----------
filas, series, pesos_de, costos_de = {}, {}, {}, {}

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

    pesos = pd.DataFrame(pesos)
    if pesos.shape[1] != len(TICKERS):
        raise SystemExit(
            f"'pesos' tiene {pesos.shape[1]} columnas y hay {len(TICKERS)} "
            f"activos. Revisa como arma el DataFrame de pesos backtest.py."
        )
    if not isinstance(pesos.index, pd.DatetimeIndex):
        raise SystemExit("El indice de 'pesos' no son fechas. Revisa backtest.py.")
    pesos.columns = TICKERS

    filas[nombre] = resumen(ret, costos, rf=0.02)
    series[nombre] = ret
    pesos_de[nombre] = pesos
    costos_de[nombre] = pd.DataFrame(costos)

tabla = pd.DataFrame(filas).T

print()
print("=" * 80)
print("RESULTADOS FUERA DE MUESTRA")
print("=" * 80)
print(tabla.round(3).to_string())
print()
print(f"Periodo evaluado: {series['1/N'].index[0].date()} a {series['1/N'].index[-1].date()}")

# ---------- Auditoria de composicion ----------
print()
print("=" * 80)
print("COMPOSICION PROMEDIO DE CADA CARTERA (todo el periodo, %)")
print("=" * 80)
promedios = pd.DataFrame({n: p.mean() * 100 for n, p in pesos_de.items()})
print(promedios.round(2).to_string())

print()
print("=" * 80)
print("COMPOSICION AL CIERRE DE 2021 (lo que cada cartera llevaba a 2022, %)")
print("=" * 80)
foto = pd.DataFrame(
    {n: p.loc[:"2021-12-31"].iloc[-1] * 100 for n, p in pesos_de.items()}
)
print(foto.round(2).to_string())
print()
print("Peso del activo mas grande (%):")
print(foto.max().round(2).to_string())
print("Activo mas grande:")
print(foto.idxmax().to_string())

# ---------- Guardar ----------
pd.DataFrame(series).to_parquet(RESULTADOS / "retornos_estrategias.parquet")
tabla.to_csv(RESULTADOS / "tabla_estrategias.csv")

for nombre in ESTRATEGIAS:
    pesos_de[nombre].to_parquet(RESULTADOS / f"pesos_{archivo_de(nombre)}.parquet")
    costos_de[nombre].to_parquet(RESULTADOS / f"costos_{archivo_de(nombre)}.parquet")

print()
print(f"Guardado en {RESULTADOS}")
for nombre in ESTRATEGIAS:
    print(f"  pesos_{archivo_de(nombre)}.parquet | costos_{archivo_de(nombre)}.parquet")