"""
Tests de integridad del motor de backtest.

NO verifican que la estrategia sea buena.
Verifican que el motor no miente:
  1. que su contabilidad reproduce un calculo hecho por otro camino,
  2. que no usa informacion del futuro,
  3. que el costo se cobra una sola vez, el dia correcto, por el monto correcto.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest, fechas_mensuales
from src.estimators import muestral
from src.optimizer import min_varianza

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def retornos():
    """Se lee una sola vez para los tres tests (scope='module')."""
    precios = pd.read_parquet(RAIZ / "data" / "prices.parquet")
    return precios.pct_change().dropna()


# --- piezas falsas, para aislar el motor del optimizador ---

def estimar_nada(ventana):
    return None, None


def optimizar_nada(mu, Sigma):
    return None


def optimizar_minvar(mu, Sigma):
    return min_varianza(Sigma, w_max=0.35)


def nunca(w_actual, w_opt, ctx):
    return False


def siempre(w_actual, w_opt, ctx):
    return True


# ---------------------------------------------------------------
# TEST 1 - reproduccion
# ---------------------------------------------------------------
def test_reproduce_buy_and_hold(retornos):
    """
    Si nunca rebalanceamos y no hay costos, el motor tiene que dar
    exactamente lo mismo que comprar 1/N el primer dia y no tocar nada.

    El camino independiente es cerrado: la riqueza de un buy-and-hold
    equiponderado es el promedio de las riquezas individuales.
    Si esto pasa, 'aplicar_drift' y los bordes del loop estan bien.
    """
    fechas = fechas_mensuales(retornos.index)

    serie, pesos, costos = backtest(
        retornos, fechas,
        estimar=estimar_nada, optimizar=optimizar_nada, decidir=nunca,
        ventana=252, costo_bps=0.0,
    )

    inicio, fin = serie.index[0], serie.index[-1]
    tramo = retornos.loc[inicio:fin]

    riqueza = (1 + tramo).cumprod().mean(axis=1)
    bh = riqueza.pct_change()
    bh.iloc[0] = riqueza.iloc[0] - 1

    assert list(serie.index) == list(bh.index), "el motor registra dias de mas o de menos"
    np.testing.assert_allclose(serie.values, bh.values, rtol=1e-10, atol=1e-13)
    assert float(np.abs(costos["turnover"]).sum()) == 0.0


# ---------------------------------------------------------------
# TEST 2 - no mira el futuro
# ---------------------------------------------------------------
def test_no_mira_el_futuro(retornos):
    """
    Corremos el backtest dos veces: con todos los datos, y con los
    datos cortados en 2021-12-31. Los retornos anteriores al corte
    tienen que ser IDENTICOS.

    Si difieren aunque sea en el sexto decimal, algo del futuro se
    filtro hacia el pasado. La tolerancia es solo por seguridad
    numerica: en teoria el resultado deberia ser exacto.
    """
    r = retornos.loc["2014-01-01":]
    corte = pd.Timestamp("2021-12-31")

    fechas_todas = fechas_mensuales(r.index)[::3]   # trimestral: mismo test, 3x mas rapido
    serie_larga, _, _ = backtest(
        r, fechas_todas,
        estimar=muestral, optimizar=optimizar_minvar, decidir=siempre,
        ventana=252, costo_bps=5.0,
    )

    r_corto = r.loc[:corte]
    fechas_cortas = fechas_mensuales(r_corto.index)[::3]
    serie_corta, _, _ = backtest(
        r_corto, fechas_cortas,
        estimar=muestral, optimizar=optimizar_minvar, decidir=siempre,
        ventana=252, costo_bps=5.0,
    )

    assert len(serie_corta) > 500, "el test se quedo sin datos, revisa las fechas"
    comun = serie_corta.index
    assert comun.isin(serie_larga.index).all()

    np.testing.assert_allclose(
        serie_larga.loc[comun].values, serie_corta.values,
        rtol=1e-9, atol=1e-12,
    )


# ---------------------------------------------------------------
# TEST 3 - contabilidad del costo
# ---------------------------------------------------------------
def test_el_costo_se_cobra_una_sola_vez(retornos):
    """
    Dos corridas identicas salvo el costo. Como el costo no afecta
    los pesos (solo el retorno registrado), las decisiones y el
    turnover tienen que ser iguales, y la diferencia entre las dos
    series tiene que ser exactamente el costo, cobrado el primer dia
    habil posterior a cada fecha de decision y en ningun otro dia.
    """
    r = retornos.loc["2014-01-01":]
    fechas = fechas_mensuales(r.index)[::3]

    s_sin, _, c_sin = backtest(
        r, fechas, estimar=muestral, optimizar=optimizar_minvar,
        decidir=siempre, ventana=252, costo_bps=0.0,
    )
    s_con, _, c_con = backtest(
        r, fechas, estimar=muestral, optimizar=optimizar_minvar,
        decidir=siempre, ventana=252, costo_bps=50.0,
    )

    np.testing.assert_allclose(
        c_sin["turnover"].values, c_con["turnover"].values, rtol=1e-12,
        err_msg="el costo esta cambiando las decisiones, no deberia",
    )

    dif = s_sin - s_con
    c = 50 / 10_000

    # 1) el total cobrado es exactamente c * suma de turnover
    assert np.isclose(dif.sum(), c * c_con["turnover"].sum(), rtol=1e-9)

    # 2) solo se cobra el primer dia habil posterior a cada fecha de decision
    dias_cobrados = dif.index[dif.abs() > 1e-14]
    primeros = set()
    for f in fechas:
        posteriores = dif.index[dif.index > f]
        if len(posteriores):
            primeros.add(posteriores[0])

    assert len(dias_cobrados) > 20, "no se cobro casi nada, el test no esta probando nada"
    assert set(dias_cobrados).issubset(primeros), "hay costo cobrado en dias que no corresponden"