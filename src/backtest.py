"""Motor de backtest walk-forward.

Regla de oro: en la fecha t solo se usa informacion hasta t.
El retorno de (t, t+1] se aplica a pesos decididos antes de conocerlo.
"""
import numpy as np
import pandas as pd


def aplicar_drift(w, r_dia):
    """Como quedan los pesos despues de un dia sin operar.
    El activo que sube pasa a pesar mas: nadie opero y la cartera cambio."""
    crecido = w * (1 + r_dia)
    return crecido / crecido.sum()


def backtest(retornos, fechas_decision, estimar, optimizar, decidir,
             ventana=252, costo_bps=5.0):
    """
    retornos        : DataFrame diario (fechas x activos)
    fechas_decision : fechas donde se EVALUA si operar
    estimar(df)     -> (mu, Sigma)
    optimizar(mu,S) -> w*
    decidir(w_actual, w_opt, ctx) -> True/False
    ventana         : cuantos dias mira hacia atras
    costo_bps       : costo por unidad de turnover, en puntos basicos
    """
    c = costo_bps / 10_000
    n = retornos.shape[1]
    w_actual = np.repeat(1 / n, n)

    reg_ret, reg_w, reg_op = [], [], []

    for i in range(len(fechas_decision) - 1):
        t = fechas_decision[i]
        t_sig = fechas_decision[i + 1]

        # ---------- SOLO informacion hasta t ----------
        historia = retornos.loc[:t].iloc[-ventana:]
        if len(historia) < ventana:
            continue

        mu, Sigma = estimar(historia)
        w_opt = optimizar(mu, Sigma)

        # ---------- decision ----------
        ctx = {"fecha": t, "mu": mu, "Sigma": Sigma}
        if decidir(w_actual, w_opt, ctx):
            turnover = np.abs(w_opt - w_actual).sum()
            costo = c * turnover
            w_actual = w_opt.copy()
            opero = True
        else:
            turnover, costo, opero = 0.0, 0.0, False

        reg_op.append((t, opero, turnover, costo))

        # ---------- recien ahora el futuro: (t, t_sig] ----------
        tramo = retornos.loc[t:t_sig].iloc[1:]
        for j, (fecha, r_dia) in enumerate(tramo.iterrows()):
            r_valores = r_dia.values
            r_port = w_actual @ r_valores
            r_neto = r_port - (costo if j == 0 else 0.0)

            reg_ret.append((fecha, r_neto))
            reg_w.append((fecha, w_actual.copy()))

            w_actual = aplicar_drift(w_actual, r_valores)

    serie_ret = pd.Series(dict(reg_ret)).sort_index()
    pesos = pd.DataFrame(dict(reg_w)).T
    pesos.columns = retornos.columns
    costos = pd.DataFrame(reg_op, columns=["fecha", "opero", "turnover", "costo"])

    return serie_ret, pesos, costos

def fechas_mensuales(indice):
    """Ultimo dia habil de cada mes presente en el indice."""
    s = pd.Series(indice, index=indice)
    return pd.DatetimeIndex(s.groupby([indice.year, indice.month]).last().values)