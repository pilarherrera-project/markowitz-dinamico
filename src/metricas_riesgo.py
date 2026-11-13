"""
src/metricas_riesgo.py

Metricas de riesgo que NO son la volatilidad.

La varianza es un promedio de desvios al cuadrado: es simetrica (castiga
igual subir que bajar) y no tiene memoria (no sabe en que ORDEN llegaron
los retornos). El drawdown es lo contrario: solo mira las caidas y
depende enteramente del camino.

Todas las funciones reciben una serie de retornos simples diarios.
"""

import numpy as np
import pandas as pd


def serie_drawdown(ret):
    """Distancia al maximo historico, dia por dia. Siempre <= 0.

    El .clip(lower=1.0) hace que el capital inicial (1.0) cuente como el
    primer maximo. Sin eso, una caida el primer dia no se registraria
    como drawdown, porque el maximo acumulado arrancaria ya caido.
    """
    curva = (1 + ret).cumprod()
    pico = curva.cummax().clip(lower=1.0)
    return curva / pico - 1


def max_drawdown(ret):
    """La peor caida desde un maximo, en tanto por uno (negativo)."""
    return float(serie_drawdown(ret).min())


def recuperacion_necesaria(d):
    """Cuanto hay que subir para volver de una caida d (d negativo).

        1 / (1 + d) - 1

    Caer 50 % exige subir 100 %. La asimetria es el punto.
    """
    return 1.0 / (1.0 + d) - 1.0


def ulcer_index(ret):
    """Indice de ulcera (Peter Martin, 1987), en %.

        UI = raiz( promedio de dd_t al cuadrado ),  con dd en %

    A diferencia del max drawdown, que es UN solo evento de toda la
    serie, el UI promedia sobre todos los dias: castiga la DURACION.
    Estar tres anios 10 % abajo pesa mas que tocar -25 % un dia y
    recuperarse en dos semanas.
    """
    dd = serie_drawdown(ret) * 100
    return float(np.sqrt((dd ** 2).mean()))


def tiempo_bajo_agua(ret):
    """Devuelve (fraccion del tiempo bajo el agua, racha mas larga en dias)."""
    dd = serie_drawdown(ret)
    bajo = dd < -1e-10
    if not bajo.any():
        return 0.0, 0
    grupos = (bajo != bajo.shift()).cumsum()
    racha = int(bajo.groupby(grupos).sum().max())
    return float(bajo.mean()), racha


def episodios_drawdown(ret):
    """Todos los episodios de drawdown, del peor al menos malo.

    Un episodio arranca el dia en que la curva deja su maximo historico
    y termina el dia en que lo recupera. El ultimo puede quedar abierto.
    """
    dd = serie_drawdown(ret)
    bajo = dd < -1e-10
    if not bajo.any():
        return pd.DataFrame()

    grupos = (bajo != bajo.shift()).cumsum()
    ultimo = dd.index[-1]
    filas = []
    for _, tramo in dd[bajo].groupby(grupos[bajo]):
        i0 = dd.index.get_loc(tramo.index[0])
        i1 = dd.index.get_loc(tramo.index[-1])
        valle = tramo.idxmin()
        iv = dd.index.get_loc(valle)
        recuperado = tramo.index[-1] != ultimo
        filas.append({
            "pico": dd.index[i0 - 1] if i0 > 0 else tramo.index[0],
            "valle": valle,
            "fin": tramo.index[-1] if recuperado else pd.NaT,
            "profundidad_%": float(tramo.min() * 100),
            "recuperar_%": float(recuperacion_necesaria(tramo.min()) * 100),
            "dias_caida": iv - i0 + 1,
            "dias_recuperacion": (i1 - iv) if recuperado else np.nan,
            "dias_bajo_agua": i1 - i0 + 1,
            "recuperado": recuperado,
        })
    return pd.DataFrame(filas).sort_values("profundidad_%").reset_index(drop=True)


def sortino(ret, rf=0.0, periodos=252):
    """Como el Sharpe, pero el denominador solo cuenta desvios HACIA ABAJO.

        semidesvio = raiz( promedio de min(exceso, 0) al cuadrado )

    Es la semivarianza que el propio Markowitz propuso en 1959 como
    alternativa a la varianza, y que descarto porque las maquinas de la
    epoca no podian calcularla.
    """
    exceso = ret - rf / periodos
    abajo = np.minimum(exceso, 0.0)
    semidesvio = float(np.sqrt((abajo ** 2).mean()) * np.sqrt(periodos))
    if semidesvio == 0:
        return np.nan
    return float(exceso.mean() * periodos / semidesvio)


def calmar(ret, periodos=252):
    """Retorno anualizado dividido por el peor drawdown. Mas alto es mejor."""
    anios = len(ret) / periodos
    r = (1 + ret).prod() ** (1 / anios) - 1
    d = abs(max_drawdown(ret))
    return float(r / d) if d > 0 else np.nan


def martin(ret, rf=0.0, periodos=252):
    """Retorno en exceso dividido por el indice de ulcera.

    Es el analogo exacto del Sharpe usando UI en lugar de volatilidad:
    "cuanto retorno extra obtuve por unidad de dolor SOSTENIDO".
    """
    anios = len(ret) / periodos
    r = (1 + ret).prod() ** (1 / anios) - 1
    ui = ulcer_index(ret)
    return float((r - rf) * 100 / ui) if ui > 0 else np.nan


def cvar(ret, nivel=0.05):
    """Perdida promedio en el peor 'nivel' de los dias, en % diario.

    El VaR dice "cuanto es lo peor que NO voy a superar el 95 % de los
    dias". El CVaR dice "y cuando lo supere, cuanto pierdo en promedio".
    El VaR mira donde empieza la cola; el CVaR mira la cola entera.
    """
    q = ret.quantile(nivel)
    cola = ret[ret <= q]
    return float(cola.mean() * 100) if len(cola) else np.nan