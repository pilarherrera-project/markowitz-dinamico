"""
src/barrido_deriva.py

EL EXPERIMENTO CENTRAL DEL PROYECTO.

Pregunta: si los insumos se ponen rancios, cuando conviene reoptimizar?
Existe un umbral de informacion nueva por debajo del cual rebalancear
destruye mas valor del que crea?

El disparador opera solo si  suma|w_opt - w_actual| > umbral.
Esa suma ES el turnover que costaria operar, asi que la regla dice
"opera solo si el movimiento vale la pena". Es una politica (s, S) de
inventario: hay una region de no-accion en el medio.

Dos trucos de eficiencia, los dos importantes:

1. CACHE DEL OPTIMIZADOR. w_opt depende de la ventana, no del umbral.
   Todos los umbrales recorren las mismas 237 fechas, asi que el
   optimizador corre una sola vez por fecha y se reusa. ~10x mas rapido.

2. RE-TARIFAR EN VEZ DE RE-CORRER. El costo no afecta las decisiones
   (el disparador mira distancia, no costo), asi que los pesos y el
   turnover son identicos para cualquier costo. Corremos con costo cero
   y despues restamos c x turnover el dia que corresponde.
   Cinco niveles de costo al precio de uno.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
from pathlib import Path

from backtest import backtest
from estimators import muestral
from optimizer import min_varianza, max_sharpe
from metrics import sharpe, max_drawdown, retorno_anual, vol_anual

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "results"
RESULTADOS.mkdir(exist_ok=True)

VENTANA = 252
RF = 0.02
UMBRALES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
COSTOS_BPS = [0, 5, 10, 20, 50]

# ---------- datos ----------
precios = pd.read_parquet(RAIZ / "data" / "prices.parquet")
retornos = precios.pct_change().dropna()
idx = retornos.index
serie_fechas = pd.Series(idx, index=idx)
fechas_decision = pd.DatetimeIndex(
    serie_fechas.groupby([idx.year, idx.month]).last().values
)
print(f"{retornos.shape[0]} dias, {retornos.shape[1]} activos, "
      f"{len(fechas_decision)} fechas de decision")
print(f"{len(UMBRALES)} umbrales x {len(COSTOS_BPS)} costos\n")


# ---------- estrategias ----------
def peso_igual(mu, Sigma):
    return np.repeat(1 / len(mu), len(mu))


def momentum(k):
    def elegir(mu, Sigma):
        w = np.zeros(len(mu))
        w[np.argsort(mu)[-k:]] = 1 / k
        return w
    return elegir


ESTRATEGIAS = {
    "1/N": peso_igual,
    "min_varianza": lambda mu, S: min_varianza(S, w_max=0.35),
    "max_sharpe": lambda mu, S: max_sharpe(mu, S, rf=RF, w_max=0.35),
    "momentum_3": momentum(3),
}


# ---------- el disparador ----------
def deriva(umbral):
    """Opera solo si la distancia L1 entre actual y optima supera el umbral."""
    def decidir(w_actual, w_opt, ctx):
        return np.abs(w_opt - w_actual).sum() > umbral
    return decidir


def con_cache(optimizar):
    """Memoiza w_opt por ventana. La clave son los bytes de mu, que
    identifican la ventana de forma unica."""
    memoria = {}

    def envuelto(mu, Sigma):
        clave = mu.tobytes()
        if clave not in memoria:
            memoria[clave] = optimizar(mu, Sigma)
        return memoria[clave]

    return envuelto


# ---------- barrido ----------
filas = []
arranque = time.time()

for nombre, optimizar in ESTRATEGIAS.items():
    opt = con_cache(optimizar)          # un cache nuevo por estrategia
    print(f"--- {nombre} ---")
    for u in UMBRALES:
        t0 = time.time()
        bruto, pesos, costos = backtest(
            retornos, fechas_decision,
            estimar=muestral, optimizar=opt, decidir=deriva(u),
            ventana=VENTANA, costo_bps=0.0,
        )

        # turnover de cada operacion, puesto en el dia en que se cobraria
        cargo = pd.Series(0.0, index=bruto.index)
        for fecha_dec, turn in zip(costos["fecha"], costos["turnover"]):
            post = bruto.index[bruto.index > fecha_dec]
            if len(post):
                cargo.loc[post[0]] += turn

        anios = len(bruto) / 252
        fila = {
            "estrategia": nombre,
            "umbral": u,
            "veces_opero": int((costos["turnover"] > 0).sum()),
            "turnover_anual": costos["turnover"].sum() / anios,
            "retorno_bruto_%": retorno_anual(bruto) * 100,
            "vol_anual_%": vol_anual(bruto) * 100,
            "max_drawdown_%": max_drawdown(bruto) * 100,
        }
        for c in COSTOS_BPS:
            neto = bruto - (c / 10_000) * cargo
            fila[f"sharpe_{c}pb"] = sharpe(neto, rf=RF)
            fila[f"retorno_{c}pb_%"] = retorno_anual(neto) * 100

        filas.append(fila)
        print(f"  umbral {u:.2f} | opero {fila['veces_opero']:>3}/236 | "
              f"turnover {fila['turnover_anual']:>6.3f} | "
              f"sharpe@5pb {fila['sharpe_5pb']:.3f} | {time.time()-t0:.1f}s")

tabla = pd.DataFrame(filas)
tabla.to_csv(RESULTADOS / "barrido_deriva.csv", index=False)
print(f"\nTiempo total: {time.time()-arranque:.0f} s")

# ---------- lectura 1: umbral optimo por nivel de costo ----------
print()
print("=" * 86)
print("UMBRAL QUE MAXIMIZA EL SHARPE, SEGUN EL COSTO DE TRANSACCION")
print("=" * 86)
for c in COSTOS_BPS:
    col = f"sharpe_{c}pb"
    mejor = tabla.loc[tabla.groupby("estrategia")[col].idxmax(),
                      ["estrategia", "umbral", "turnover_anual", col]]
    print(f"\nCosto = {c} pb")
    print(mejor.set_index("estrategia").round(3).to_string())

# ---------- lectura 2: mensual vs mejor umbral, a 5 pb ----------
print()
print("=" * 86)
print("REBALANCEO MENSUAL (umbral 0) VS MEJOR UMBRAL, a 5 pb")
print("=" * 86)
base = tabla[tabla["umbral"] == 0.0].set_index("estrategia")
mejor = tabla.loc[tabla.groupby("estrategia")["sharpe_5pb"].idxmax()].set_index("estrategia")
comp = pd.DataFrame({
    "umbral_mejor": mejor["umbral"],
    "sharpe_mensual": base["sharpe_5pb"],
    "sharpe_mejor": mejor["sharpe_5pb"],
    "turnover_mensual": base["turnover_anual"],
    "turnover_mejor": mejor["turnover_anual"],
})
comp["recorte_turnover_%"] = (comp["turnover_mejor"] / comp["turnover_mensual"] - 1) * 100
comp["cambio_sharpe_%"] = (comp["sharpe_mejor"] / comp["sharpe_mensual"] - 1) * 100
print(comp.round(3).to_string())

print(f"\nGuardado en {RESULTADOS / 'barrido_deriva.csv'}")