"""Estimadores de mu y Sigma. Todos reciben una ventana de retornos
y devuelven (mu, Sigma) anualizados, como arrays de numpy."""
import numpy as np


def muestral(ventana, periodos=252):
    """Media y covarianza muestrales. El estimador clasico."""
    mu = ventana.mean().values * periodos
    Sigma = ventana.cov().values * periodos
    return mu, Sigma