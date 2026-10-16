"""Metricas de evaluacion de estrategias."""
import numpy as np


def retorno_anual(ret, periodos=252):
    """Retorno geometrico anualizado: el que determina la riqueza final."""
    anios = len(ret) / periodos
    return (1 + ret).prod() ** (1 / anios) - 1


def vol_anual(ret, periodos=252):
    return ret.std() * np.sqrt(periodos)


def sharpe(ret, rf=0.0, periodos=252):
    """Exceso de retorno por unidad de riesgo, anualizado."""
    exceso = ret - rf / periodos
    return np.sqrt(periodos) * exceso.mean() / exceso.std()


def max_drawdown(ret):
    curva = (1 + ret).cumprod()
    return (curva / curva.cummax() - 1).min()


def riqueza_final(ret):
    """Cuanto vale al final cada peso invertido al principio."""
    return (1 + ret).prod()


def resumen(ret, costos=None, rf=0.0, periodos=252):
    """Diccionario con todas las metricas de una estrategia."""
    anios = len(ret) / periodos
    fila = {
        "retorno_anual_%": retorno_anual(ret, periodos) * 100,
        "vol_anual_%": vol_anual(ret, periodos) * 100,
        "sharpe": sharpe(ret, rf, periodos),
        "max_drawdown_%": max_drawdown(ret) * 100,
        "riqueza_final": riqueza_final(ret),
    }
    if costos is not None:
        fila["turnover_anual"] = costos["turnover"].sum() / anios
        fila["costo_total_%"] = costos["costo"].sum() * 100
        fila["veces_opero"] = int(costos["opero"].sum())
    return fila