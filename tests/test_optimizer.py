import numpy as np
from src.optimizer import min_varianza


def test_ejemplo_del_apunte():
    """
    Caso con solucion analitica conocida:
    sigma_A^2 = 2.5, sigma_B^2 = 1.5, sigma_AB = -1.5
    La formula cerrada da w_A = 3/7 = 0.4286
    """
    Sigma = np.array([
        [2.5, -1.5],
        [-1.5, 1.5],
    ])
    w = min_varianza(Sigma, w_max=1.0)   # sin tope, para comparar con la formula

    esperado = (1.5 - (-1.5)) / (2.5 + 1.5 - 2 * (-1.5))   # = 3/7

    assert np.isclose(w[0], esperado, atol=1e-4)
    assert np.isclose(w.sum(), 1.0)


def test_dos_activos_sin_correlacion():
    """Segundo caso: rho = 0, sigma_A = 20%, sigma_B = 12%."""
    sA, sB, rho = 0.20, 0.12, 0.0
    Sigma = np.array([
        [sA**2, rho * sA * sB],
        [rho * sA * sB, sB**2],
    ])
    w = min_varianza(Sigma, w_max=1.0)

    esperado = (sB**2 - rho * sA * sB) / (sA**2 + sB**2 - 2 * rho * sA * sB)

    assert np.isclose(w[0], esperado, atol=1e-4)


def test_respeta_el_tope():
    """Con tope de 0.35 y 4 activos, ningun peso puede pasarse."""
    Sigma = np.diag([0.01, 0.04, 0.09, 0.16])   # varianzas muy distintas
    w = min_varianza(Sigma, w_max=0.35)

    assert w.max() <= 0.35 + 1e-6
    assert w.min() >= -1e-6
    assert np.isclose(w.sum(), 1.0)