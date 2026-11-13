"""
src/validacion_rodante.py

LA PRUEBA DE HONESTIDAD DEL PROYECTO.

En el bloque F elegimos el umbral mirando los mismos 18 anios que despues
usamos para evaluarlo. Eso es overfitting: con 15 umbrales probados, el
mejor de 15 se ve bien aunque ninguno sirva.

Aca lo hacemos como se hace de verdad: walk-forward.

  para cada anio Y desde 2013:
      mirar SOLO los 5 anios anteriores (Y-5 a Y-1)
      elegir el umbral con mejor Sharpe neto en esa ventana
      aplicarlo durante todo el anio Y, a ciegas

Se comparan cuatro politicas sobre el MISMO periodo (2013-2026) y desde
la MISMA cartera inicial (las cuatro rebalancean mensual hasta 2012):

  adaptativa     el umbral elegido anio a anio, sin mirar el futuro
  mensual        umbral 0: rebalancear siempre que toca fecha
  fijo_insample  el umbral que gano mirando TODA la muestra (el tramposo)
  buy_and_hold   umbral 2.0: no operar nunca mas

Si la adaptativa le gana a mensual, el resultado vale.
Si fijo_insample le gana a adaptativa por mucho, eso mide cuanto de la
mejora del bloque F era overfitting.
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
UMBRALES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60,
            0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00]
COSTOS_BPS = [0, 5, 10, 20, 50]
COSTO_SELECCION = 5        # a que costo se elige el umbral
ENTRENAMIENTO = 5          # anios de ventana de entrenamiento
ANIO_EVAL = 2013           # primer anio evaluado fuera de muestra

# ---------- datos ----------
precios = pd.read_parquet(RAIZ / "data" / "prices.parquet")
retornos = precios.pct_change().dropna()
idx = retornos.index
serie_fechas = pd.Series(idx, index=idx)
fechas_decision = pd.DatetimeIndex(
    serie_fechas.groupby([idx.year, idx.month]).last().values
)


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


# ---------- disparador con calendario ----------
def deriva_calendario(calendario, por_defecto=0.0):
    """Umbral que puede cambiar de un anio a otro.

    calendario: dict {anio: umbral}. Los anios que no estan usan
    'por_defecto', que es 0.0 = rebalanceo mensual (el periodo de arranque).
    """
    def decidir(w_actual, w_opt, ctx):
        u = calendario.get(ctx["fecha"].year, por_defecto)
        return np.abs(w_opt - w_actual).sum() > u
    return decidir


def con_cache(optimizar):
    """Memoiza w_opt por ventana: la clave son los bytes de mu."""
    memoria = {}

    def envuelto(mu, Sigma):
        clave = mu.tobytes()
        if clave not in memoria:
            memoria[clave] = optimizar(mu, Sigma)
        return memoria[clave]

    return envuelto


def correr(optimizar, decidir):
    """Corre un backtest con costo 0 y devuelve (bruto, cargo, costos).

    'cargo' es el turnover puesto en el dia en que se cobraria. Con eso,
    el neto a cualquier costo c es:  bruto - (c/10000) * cargo
    """
    bruto, pesos, costos = backtest(
        retornos, fechas_decision,
        estimar=muestral, optimizar=optimizar, decidir=decidir,
        ventana=VENTANA, costo_bps=0.0,
    )
    cargo = pd.Series(0.0, index=bruto.index)
    fechas_op = pd.to_datetime(costos["fecha"])
    for f, turn in zip(fechas_op, costos["turnover"]):
        post = bruto.index[bruto.index > f]
        if len(post):
            cargo.loc[post[0]] += turn
    costos = costos.copy()
    costos["fecha"] = fechas_op
    return bruto, cargo, costos


def neto(bruto, cargo, costo_bps):
    return bruto - (costo_bps / 10_000) * cargo


arranque = time.time()
resumen_final = []
calendarios = {}

for nombre, optimizar in ESTRATEGIAS.items():
    print(f"\n{'='*86}\n{nombre}\n{'='*86}")
    opt = con_cache(optimizar)

    # ---------- PASO 1: una corrida por umbral, umbral FIJO ----------
    print("Paso 1: corriendo los 15 umbrales fijos...")
    series_fijas = {}
    for u in UMBRALES:
        b, c, _ = correr(opt, deriva_calendario({}, por_defecto=u))
        series_fijas[u] = neto(b, c, COSTO_SELECCION)

    # el umbral tramposo: el mejor mirando TODA la muestra
    u_insample = max(UMBRALES, key=lambda u: sharpe(series_fijas[u], rf=RF))
    print(f"  umbral ganador mirando toda la muestra (tramposo): {u_insample:.2f}")

    # ---------- PASO 2: elegir el umbral anio a anio, sin mirar adelante ----------
    ultimo_anio = int(retornos.index[-1].year)
    calendario = {}
    for anio in range(ANIO_EVAL, ultimo_anio + 1):
        desde = f"{anio - ENTRENAMIENTO}-01-01"
        hasta = f"{anio - 1}-12-31"
        mejor_u, mejor_s = UMBRALES[0], -np.inf
        for u in UMBRALES:
            tramo = series_fijas[u].loc[desde:hasta]
            if len(tramo) < 200:
                continue
            s = sharpe(tramo, rf=RF)
            if s > mejor_s:
                mejor_s, mejor_u = s, u
        calendario[anio] = mejor_u
    calendarios[nombre] = calendario

    print("\nPaso 2: umbral elegido cada anio, mirando solo los 5 anteriores")
    print("  anio: " + "  ".join(str(a) for a in calendario))
    print("  umbr: " + "  ".join(f"{calendario[a]:.2f}" for a in calendario))

    # ---------- PASO 3: las cuatro politicas, mismo arranque ----------
    anios_eval = range(ANIO_EVAL, ultimo_anio + 1)
    POLITICAS = {
        "adaptativa":    calendario,
        "mensual":       {},
        "fijo_insample": {a: u_insample for a in anios_eval},
        "buy_and_hold":  {a: 2.0 for a in anios_eval},
    }

    print("\nPaso 3: evaluando las cuatro politicas sobre 2013-{}...".format(ultimo_anio))
    corte = pd.Timestamp(f"{ANIO_EVAL}-01-01")
    for etiqueta, cal in POLITICAS.items():
        b, c, costos = correr(opt, deriva_calendario(cal, por_defecto=0.0))
        b_ev = b.loc[corte:]
        c_ev = c.loc[corte:]
        anios = len(b_ev) / 252
        ops = costos[costos["fecha"] >= corte]

        fila = {
            "estrategia": nombre,
            "politica": etiqueta,
            "veces_opero": int((ops["turnover"] > 0).sum()),
            "turnover_anual": ops["turnover"].sum() / anios,
        }
        for cb in COSTOS_BPS:
            n = b_ev - (cb / 10_000) * c_ev
            fila[f"sharpe_{cb}pb"] = sharpe(n, rf=RF)
        n5 = b_ev - (COSTO_SELECCION / 10_000) * c_ev
        fila["retorno_5pb_%"] = retorno_anual(n5) * 100
        fila["vol_%"] = vol_anual(n5) * 100
        fila["max_drawdown_%"] = max_drawdown(n5) * 100
        resumen_final.append(fila)

tabla = pd.DataFrame(resumen_final)
tabla.to_csv(RESULTADOS / "validacion_rodante.csv", index=False)

pd.DataFrame(calendarios).to_csv(RESULTADOS / "umbrales_elegidos.csv")

# ---------- lectura ----------
print(f"\n\n{'='*100}")
print(f"RESULTADO FUERA DE MUESTRA  ({ANIO_EVAL}-{ultimo_anio}, mismo arranque para las cuatro)")
print(f"{'='*100}")
cols = ["politica", "sharpe_5pb", "retorno_5pb_%", "vol_%",
        "max_drawdown_%", "turnover_anual", "veces_opero"]
for nombre in ESTRATEGIAS:
    sub = tabla[tabla["estrategia"] == nombre][cols]
    print(f"\n--- {nombre} ---")
    print(sub.set_index("politica").round(3).to_string())

print(f"\n\n{'='*100}")
print("LA PREGUNTA: la adaptativa le gana al rebalanceo mensual?")
print(f"{'='*100}")
piv = tabla.pivot(index="estrategia", columns="politica", values="sharpe_5pb")
piv["adaptativa_vs_mensual_%"] = (piv["adaptativa"] / piv["mensual"] - 1) * 100
piv["cuanto_era_overfitting_%"] = (piv["fijo_insample"] / piv["adaptativa"] - 1) * 100
print(piv.round(3).to_string())

print(f"\nTiempo total: {time.time()-arranque:.0f} s")
print(f"Guardado en {RESULTADOS}")