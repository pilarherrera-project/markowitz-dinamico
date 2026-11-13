"""Tests de las metricas de riesgo, sobre series chicas calculables a mano."""

import numpy as np
import pandas as pd
import pytest

from src.metricas_riesgo import (serie_drawdown, max_drawdown,
                                 recuperacion_necesaria, ulcer_index,
                                 tiempo_bajo_agua, episodios_drawdown, sortino)


@pytest.fixture
def serie():
    """curva: 1.10, 0.88, 1.10, 1.10  ->  drawdowns: 0, -20 %, 0, 0"""
    f = pd.bdate_range("2020-01-01", periods=4)
    return pd.Series([0.10, -0.20, 0.25, 0.00], index=f)


def test_serie_drawdown(serie):
    np.testing.assert_allclose(serie_drawdown(serie).values,
                               [0.0, -0.20, 0.0, 0.0], atol=1e-12)


def test_max_drawdown(serie):
    assert np.isclose(max_drawdown(serie), -0.20)


def test_recuperacion_necesaria():
    assert np.isclose(recuperacion_necesaria(-0.50), 1.00)   # -50 % -> +100 %
    assert np.isclose(recuperacion_necesaria(-0.20), 0.25)


def test_ulcer_index(serie):
    # drawdowns en %: 0, -20, 0, 0  ->  raiz(400/4) = 10
    assert np.isclose(ulcer_index(serie), 10.0)


def test_ulcer_castiga_la_duracion():
    """Mismo max drawdown, distinta duracion: el UI tiene que distinguirlas.

    Este test es el corazon del bloque F2. Si falla, la metrica no esta
    midiendo lo que decimos que mide.
    """
    f = pd.bdate_range("2020-01-01", periods=6)
    rapida = pd.Series([-0.20, 0.25, 0.0, 0.0, 0.0, 0.0], index=f)
    lenta = pd.Series([-0.20, 0.0, 0.0, 0.0, 0.0, 0.25], index=f)
    assert np.isclose(max_drawdown(rapida), max_drawdown(lenta))
    assert ulcer_index(lenta) > 2 * ulcer_index(rapida)


def test_tiempo_bajo_agua(serie):
    frac, racha = tiempo_bajo_agua(serie)
    assert np.isclose(frac, 0.25)
    assert racha == 1


def test_episodios(serie):
    ep = episodios_drawdown(serie)
    assert len(ep) == 1
    assert np.isclose(ep.loc[0, "profundidad_%"], -20.0)
    assert bool(ep.loc[0, "recuperado"])


def test_sortino_ignora_las_subidas():
    """Agregar un dia muy bueno no puede empeorar el Sortino."""
    f5 = pd.bdate_range("2020-01-01", periods=5)
    f6 = pd.bdate_range("2020-01-01", periods=6)
    base = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01], index=f5)
    con_salto = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01, 0.20], index=f6)
    assert sortino(con_salto) > sortino(base)