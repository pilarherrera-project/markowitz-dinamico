"""Tests del optimizador de CVaR."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimizer_cvar import min_cvar, cvar_de_cartera, escenarios
from src.optimizer import min_varianza

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ventana():
    """Los ultimos 252 dias habiles de datos reales."""
    precios = pd.read_parquet(RAIZ / "data" / "prices.parquet")
    return precios.pct_change().dropna().iloc[-252:]


def test_elige_el_activo_sin_riesgo():
    """Dos activos: uno constante y positivo, otro que oscila fuerte.

    La cartera de minimo CVaR tiene que ir entera al constante.
    """
    T = 100
    seguro = np.full(T, 0.001)
    volatil = np.where(np.arange(T) % 2 == 0, 0.05, -0.05)
    R = np.column_stack([seguro, volatil])

    w = min_cvar(R, beta=0.95, w_max=1.0)
    assert w[0] > 0.98
    assert w[1] < 0.02


def test_respeta_las_restricciones(ventana):
    w = min_cvar(ventana.values, beta=0.95, w_max=0.20)
    assert np.isclose(w.sum(), 1.0, atol=1e-8)
    assert w.min() >= -1e-8
    assert w.max() <= 0.20 + 1e-8


def test_le_gana_a_1_sobre_N_en_su_propia_metrica(ventana):
    R = ventana.values
    n = R.shape[1]
    w_cvar = min_cvar(R, beta=0.95, w_max=0.35)
    w_igual = np.repeat(1 / n, n)
    assert cvar_de_cartera(R, w_cvar) < cvar_de_cartera(R, w_igual)


def test_le_gana_a_min_varianza_en_CVaR(ventana):
    """En su propia vara, el optimizador de CVaR tiene que ganar.

    Si no gana, o el LP esta mal planteado o no esta optimizando nada.
    """
    R = ventana.values
    w_cvar = min_cvar(R, beta=0.95, w_max=0.35)
    w_var = min_varianza(ventana.cov().values * 252, w_max=0.35)
    assert cvar_de_cartera(R, w_cvar) <= cvar_de_cartera(R, w_var) + 1e-12


def test_el_piso_de_retorno_se_cumple(ventana):
    mu, R = escenarios(ventana)
    objetivo = float(np.median(mu))
    w = min_cvar(R, beta=0.95, w_max=0.35, retorno_minimo=objetivo, mu=mu)
    assert w @ mu >= objetivo - 1e-8