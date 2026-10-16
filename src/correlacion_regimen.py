"""
src/correlacion_regimen.py

Mide la correlacion rodante a 252 dias entre un ETF de acciones y uno
de bonos, y la grafica.

Para que sirve: min_varianza baja la varianza apoyandose en que los
bonos se mueven al reves que las acciones. Ese supuesto no es una ley
fisica, es un parametro estimado del pasado. Este script muestra si
ese parametro se mantuvo estable o si cambio de signo.

No calcula estrategias. Solo lee data/prices.parquet.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RESULTS = RAIZ / "results"
VENTANA = 252

# Si sabes cuales son tus tickers, escribilos aca entre comillas.
# Si los dejas en None, el script los busca solo entre los candidatos.
ACCIONES = None
BONOS = None

CANDIDATOS_ACCIONES = ["SPY", "VTI", "IVV", "VOO", "QQQ", "XLK"]
CANDIDATOS_BONOS = ["TLT", "IEF", "AGG", "BND", "LQD", "SHY"]


def elegir(preferido, candidatos, columnas, etiqueta):
    """Devuelve el ticker a usar, o corta con un mensaje util."""
    if preferido is not None:
        if preferido not in columnas:
            raise SystemExit(f"'{preferido}' no esta en los datos.")
        return preferido
    for c in candidatos:
        if c in columnas:
            return c
    raise SystemExit(
        f"No encontre ningun ticker de {etiqueta} entre {candidatos}.\n"
        f"Edita ACCIONES y BONOS arriba del archivo con dos de las columnas de arriba."
    )


precios = pd.read_parquet(RAIZ / "data" / "prices.parquet")
retornos = precios.pct_change().dropna()

print(f"Columnas disponibles: {list(retornos.columns)}\n")

acc = elegir(ACCIONES, CANDIDATOS_ACCIONES, retornos.columns, "acciones")
bon = elegir(BONOS, CANDIDATOS_BONOS, retornos.columns, "bonos")
print(f"Usando acciones = {acc} | bonos = {bon} | ventana = {VENTANA} dias hábiles\n")

# correlacion de Pearson entre los retornos diarios, en una ventana movil
corr = retornos[acc].rolling(VENTANA).corr(retornos[bon]).dropna()

# ---------- tabla: ultimo valor de cada ano ----------
anual = corr.groupby(corr.index.year).last()
print("Correlacion rodante al cierre de cada ano:")
print(anual.round(2).to_string())

# ---------- promedios por sub-periodo ----------
periodos = [
    ("2008-2011  crisis financiera", "2008-01-01", "2011-12-31"),
    ("2012-2020  regimen de correlacion negativa", "2012-01-01", "2020-12-31"),
    ("2021       ano previo al quiebre", "2021-01-01", "2021-12-31"),
    ("2022       el quiebre", "2022-01-01", "2022-12-31"),
    ("2023-hoy   despues", "2023-01-01", None),
]

print("\nPromedio por sub-periodo:")
for etiqueta, desde, hasta in periodos:
    tramo = corr.loc[desde:hasta]
    if len(tramo):
        print(f"  {etiqueta:<46}{tramo.mean():>7.2f}   (min {tramo.min():>5.2f} / max {tramo.max():>5.2f})")

# ---------- grafico ----------
fig, ax = plt.subplots(figsize=(11, 5))

valores = corr.values
ax.fill_between(corr.index, valores, 0, where=(valores >= 0),
                color="#d97706", alpha=0.20, linewidth=0,
                label="correlacion positiva: no diversifica")
ax.fill_between(corr.index, valores, 0, where=(valores < 0),
                color="#2563eb", alpha=0.20, linewidth=0,
                label="correlacion negativa: si diversifica")
ax.plot(corr.index, valores, color="#1f2937", linewidth=1.2)
ax.axhline(0, color="black", linewidth=0.9)

ax.set_ylim(-1, 1)
ax.set_ylabel(f"Correlacion rodante a {VENTANA} dias")
ax.set_title(f"{acc} vs {bon} | el supuesto de diversificacion que usa min_varianza")
ax.legend(frameon=False, loc="lower left", fontsize=9)
ax.grid(True, alpha=0.25)

fig.tight_layout()
salida = RESULTS / "correlacion_regimen.png"
fig.savefig(salida)