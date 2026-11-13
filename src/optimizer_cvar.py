"""
src/optimizer_cvar.py

Optimizacion de cartera minimizando CVaR en vez de varianza.

Teorema (Rockafellar y Uryasev, 2000):

    CVaR_beta(w) = min sobre alpha de
                   alpha + (1/((1-beta) T)) * suma_t max(0, perdida_t - alpha)

Esa expresion es convexa en (w, alpha) juntos, y el max(0, .) se elimina
con una variable auxiliar z_t por dia. El resultado es un LP exacto.

A diferencia de Markowitz, el optimizador NO recibe mu y Sigma: recibe
la matriz de escenarios R (T x N) con los retornos crudos. El CVaR es
una propiedad de la distribucion entera, no de sus dos primeros momentos.
"""

import numpy as np
from scipy.optimize import linprog


def escenarios(ventana, periodos=252):
    """Estimador para min_cvar: devuelve (mu anualizada, matriz de escenarios).

    El motor de backtest solo pasa lo que devuelve 'estimar' a 'optimizar',
    sin mirarlo. Por eso podemos poner la matriz de retornos crudos en el
    segundo lugar, donde las otras estrategias llevan Sigma.
    """
    mu = ventana.mean().values * periodos
    return mu, ventana.values


def min_cvar(R, beta=0.95, w_max=0.35, retorno_minimo=None, mu=None):
    """Cartera que minimiza el CVaR al nivel beta.

    R     matriz T x N de retornos simples: una fila por dia, una columna
          por activo. Son los ESCENARIOS. El metodo no supone ninguna
          distribucion: usa la muestra tal cual, colas incluidas.
    beta  nivel de confianza. 0.95 = "el promedio del peor 5 % de los dias".
    w_max tope por activo, igual que en min_varianza.
    retorno_minimo  si se pasa, agrega  w' mu >= retorno_minimo  y el
                    problema pasa a ser media-CVaR (hace falta mu).
    """
    R = np.asarray(R, dtype=float)
    T, n = R.shape
    peso_cola = 1.0 / ((1.0 - beta) * T)

    # ---- x = [ w (n) , alpha (1) , z (T) ] ----
    c = np.concatenate([np.zeros(n), [1.0], np.full(T, peso_cola)])

    # ---- desigualdades: -R w - alpha - z <= 0 ----
    A_ub = np.zeros((T, n + 1 + T))
    A_ub[:, :n] = -R
    A_ub[:, n] = -1.0
    A_ub[:, n + 1:] = -np.eye(T)
    b_ub = np.zeros(T)

    # ---- opcional: piso de retorno esperado ----
    if retorno_minimo is not None:
        if mu is None:
            raise ValueError("retorno_minimo necesita que le pases mu")
        fila = np.zeros((1, n + 1 + T))
        fila[0, :n] = -np.asarray(mu, dtype=float)   # -w'mu <= -objetivo
        A_ub = np.vstack([A_ub, fila])
        b_ub = np.concatenate([b_ub, [-retorno_minimo]])

    # ---- igualdad: los pesos suman 1 ----
    A_eq = np.zeros((1, n + 1 + T))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # ---- cotas: w acotado, alpha libre, z no negativo ----
    cotas = [(0.0, w_max)] * n + [(None, None)] + [(0.0, None)] * T

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=cotas, method="highs")
    if not res.success:
        raise RuntimeError(f"El LP de CVaR no convergio: {res.message}")

    return res.x[:n]


def cvar_de_cartera(R, w, beta=0.95):
    """CVaR empirico de una cartera dada. Positivo = perdida.

    Sirve para verificar por otro camino que min_cvar hizo su trabajo.
    Puede diferir en el tercer decimal del valor que reporta el LP, por
    como cada uno trata el dia justo en el borde de la cola.
    """
    R = np.asarray(R, dtype=float)
    perdidas = -(R @ np.asarray(w, dtype=float))
    k = max(1, int(np.ceil((1.0 - beta) * len(perdidas))))
    return float(np.sort(perdidas)[-k:].mean())


def var_de_cartera(R, w, beta=0.95):
    """VaR empirico: donde ARRANCA la cola. Positivo = perdida."""
    perdidas = -(np.asarray(R, dtype=float) @ np.asarray(w, dtype=float))
    return float(np.quantile(perdidas, beta))