"""
Descarga y prepara los precios del universo del proyecto.
"""
import yfinance as yf
import pandas as pd
from pathlib import Path

# --- Configuracion del universo ------------------------------------
TICKERS = [
    # 9 sectores del S&P 500 (SPDR Select Sector)
    "XLK",  # Tecnologia
    "XLF",  # Financiero
    "XLE",  # Energia
    "XLV",  # Salud
    "XLI",  # Industrial
    "XLY",  # Consumo discrecional
    "XLP",  # Consumo basico
    "XLU",  # Servicios publicos
    "XLB",  # Materiales
    # Otras clases de activo
    "TLT",  # Bonos del Tesoro largos (20+ anios)
    "IEF",  # Bonos del Tesoro medios (7-10 anios)
    "GLD",  # Oro
    "EFA",  # Acciones desarrolladas ex-USA
    "EEM",  # Acciones emergentes
]

INICIO = "2007-01-01"

# Rutas calculadas desde la ubicacion de este archivo
RAIZ = Path(__file__).resolve().parent.parent
RUTA_PRECIOS = RAIZ / "data" / "prices.parquet"


def descargar_precios(tickers=TICKERS, inicio=INICIO):
    """Baja precios ajustados y devuelve una tabla fechas x tickers."""
    datos = yf.download(
        tickers,
        start=inicio,
        auto_adjust=True,
        progress=False,
    )
    precios = datos["Close"]
    precios = precios.dropna()
    return precios


def calcular_retornos(precios):
    """Retornos simples diarios: (P_t - P_{t-1}) / P_{t-1}"""
    return precios.pct_change().dropna()


if __name__ == "__main__":
    precios = descargar_precios()

    RUTA_PRECIOS.parent.mkdir(exist_ok=True)
    precios.to_parquet(RUTA_PRECIOS)

    print("Filas y columnas:", precios.shape)
    print("Desde", precios.index[0].date(), "hasta", precios.index[-1].date())
    print()
    print(precios.tail(3))