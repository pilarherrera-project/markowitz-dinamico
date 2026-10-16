"""
Optimizacion de carteras segun Markowitz (1952).
Todas las funciones reciben arrays de numpy, no DataFrames.
"""
import numpy as np
from scipy.optimize import minimize


def varianza_portafolio(w, Sigma):
    """La funcion objetivo: w' Sigma w"""
    return w @ Sigma @ w


def min_varianza(Sigma, w_max=0.35):
    """
    Cartera de minima varianza global.
    Restricciones: pesos suman 1, long-only, tope w_max por activo.
    """
    n = Sigma.shape[0]
    w0 = np.repeat(1 / n, n)                                # arranca en 1/N
    restricciones = [{"type": "eq",
                      "fun": lambda w: np.sum(w) - 1}]      # suma = 1
    limites = [(0.0, w_max)] * n                            # 0 <= w_i <= w_max

    res = minimize(
        varianza_portafolio, w0, args=(Sigma,),
        method="SLSQP", bounds=limites, constraints=restricciones,
    )
    if not res.success:
        raise RuntimeError(f"No convergio: {res.message}")
    return res.x


def max_sharpe(mu, Sigma, rf=0.0, w_max=0.35):
    """
    Cartera tangente: maximiza (w'mu - rf) / sqrt(w' Sigma w).
    Como scipy solo minimiza, minimizamos el Sharpe negativo.
    """
    n = len(mu)

    def sharpe_negativo(w):
        exceso = w @ mu - rf
        vol = np.sqrt(w @ Sigma @ w)
        return -exceso / vol

    w0 = np.repeat(1 / n, n)
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    limites = [(0.0, w_max)] * n

    res = minimize(sharpe_negativo, w0, method="SLSQP",
                   bounds=limites, constraints=restricciones)
    if not res.success:
        raise RuntimeError(f"No convergio: {res.message}")
    return res.x


def frontera(mu, Sigma, n_puntos=40, w_max=0.35):
    """
    Recorre la frontera eficiente resolviendo el problema para
    distintos retornos objetivo.
    Devuelve (volatilidades, retornos, pesos).
    """
    n = len(mu)
    objetivos = np.linspace(mu.min(), mu.max(), n_puntos)
    vols, rets, pesos = [], [], []

    for objetivo in objetivos:
        restricciones = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, o=objetivo: w @ mu - o},
        ]
        res = minimize(
            varianza_portafolio, np.repeat(1 / n, n), args=(Sigma,),
            method="SLSQP", bounds=[(0.0, w_max)] * n,
            constraints=restricciones,
        )
        if res.success:
            vols.append(np.sqrt(res.fun))
            rets.append(objetivo)
            pesos.append(res.x)

    return np.array(vols), np.array(rets), np.array(pesos)