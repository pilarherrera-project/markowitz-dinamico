"""
src/graficar_en_ingles.py

Regenera, en ingles, las cinco figuras que usa el README, directamente
en figuras/. No recalcula nada: lee lo que ya esta en results/.

Un solo script para no tener que traducir cinco archivos distintos.
Los nombres de las estrategias quedan como estan (min_varianza, etc.)
porque son identificadores de codigo y el README los cita asi.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "results"
FIG = RAIZ / "figuras"
FIG.mkdir(exist_ok=True)

COL = {"1/N": "#6b7280", "min_varianza": "#2563eb", "max_sharpe": "#d97706",
       "momentum_3": "#059669", "momentum_5": "#7c3aed", "min_cvar": "#be123c"}
ORDEN = ["1/N", "min_varianza", "max_sharpe", "momentum_3", "momentum_5", "min_cvar"]


# --------------------------------------------------------------- 1
def equity_curves():
    ret = pd.read_parquet(RES / "retornos_estrategias.parquet")
    riqueza = (1 + ret).cumprod()
    dd = riqueza / riqueza.cummax() - 1

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    for c in riqueza.columns:
        a1.plot(riqueza.index, riqueza[c], label=c, color=COL.get(c), lw=1.5)
    a1.set_yscale("log")
    a1.set_ylabel("Cumulative wealth (base 1, log scale)")
    a1.set_title("Dynamic Markowitz | equity curves net of costs "
                 "(5 bps, monthly rebalancing)")
    a1.legend(frameon=False, loc="upper left")
    a1.grid(True, which="both", alpha=0.25)

    for c in dd.columns:
        a2.fill_between(dd.index, dd[c] * 100, 0, color=COL.get(c),
                        alpha=0.20, linewidth=0)
        a2.plot(dd.index, dd[c] * 100, color=COL.get(c), lw=1.0)
    a2.set_ylabel("Drawdown (%)")
    a2.axhline(0, color="black", lw=0.8)
    a2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG / "curvas_capital.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- 2
GRUPOS = {
    "Bonds (IEF, TLT)":         ["IEF", "TLT"],
    "Gold (GLD)":               ["GLD"],
    "Energy (XLE)":             ["XLE"],
    "International (EEM, EFA)": ["EEM", "EFA"],
    "US sectors (rest)":        ["XLB", "XLF", "XLI", "XLK",
                                 "XLP", "XLU", "XLV", "XLY"],
}
COLORES_G = ["#0072B2", "#E69F00", "#CC79A7", "#009E73", "#999999"]


def weights_over_time():
    presentes = [e for e in ORDEN
                 if (RES / f"pesos_{e.replace('/', '-')}.parquet").exists()]
    fig, ejes = plt.subplots(len(presentes), 1,
                             figsize=(11, 3.3 * len(presentes)), sharex=True)
    if len(presentes) == 1:
        ejes = [ejes]

    for ax, nombre in zip(ejes, presentes):
        p = pd.read_parquet(RES / f"pesos_{nombre.replace('/', '-')}.parquet")
        g = pd.DataFrame({
            etq: p[[t for t in tk if t in p.columns]].sum(axis=1)
            for etq, tk in GRUPOS.items()
        })
        ax.stackplot(g.index, [g[c].values for c in g.columns],
                     labels=list(g.columns), colors=COLORES_G, linewidth=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("weight")
        ax.set_title(nombre, loc="left", fontsize=11)
        ax.margins(x=0)

    man, etq = ejes[0].get_legend_handles_labels()
    fig.legend(man, etq, loc="upper center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Portfolio composition over time", y=1.0, fontsize=13)
    fig.subplots_adjust(top=0.88)
    fig.savefig(FIG / "pesos_en_el_tiempo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- 3
def threshold_sweep():
    t = pd.read_csv(RES / "barrido_deriva.csv")
    costos = [0, 5, 10, 20, 50]
    colores = ["#94a3b8", "#0072B2", "#009E73", "#E69F00", "#D55E00"]
    ests = list(t["estrategia"].unique())

    fig, ejes = plt.subplots(1, len(ests), figsize=(4.3 * len(ests), 5.0))
    if len(ests) == 1:
        ejes = [ejes]
    for ax, nombre in zip(ejes, ests):
        sub = t[t["estrategia"] == nombre].sort_values("umbral")
        for c, color in zip(costos, colores):
            col = f"sharpe_{c}pb"
            if col in sub.columns:
                ax.plot(sub["umbral"], sub[col], marker="o", ms=4.5,
                        color=color, lw=1.7, label=f"{c} bps")
        mejor = sub.loc[sub["sharpe_5pb"].idxmax()]
        ax.scatter([mejor["umbral"]], [mejor["sharpe_5pb"]], s=170,
                   facecolor="none", edgecolor="#111827", lw=1.8, zorder=5)
        ax.set_title(nombre, fontsize=11.5)
        ax.set_xlabel("drift threshold  (sum |w_opt - w_current|)")
        ax.grid(True, alpha=0.25)

    ejes[0].set_ylabel("Sharpe net of costs")
    ejes[-1].legend(frameon=False, fontsize=9, title="cost",
                    title_fontsize=9, loc="lower left")
    fig.suptitle("How much rebalancing can you skip before losing Sharpe",
                 fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "barrido_deriva.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- 4
def walk_forward():
    t = pd.read_csv(RES / "validacion_rodante.csv")
    u = pd.read_csv(RES / "umbrales_elegidos.csv", index_col=0)

    C_POL = {"mensual": "#94a3b8", "adaptativa": "#0072B2",
             "fijo_insample": "#D55E00", "buy_and_hold": "#111827"}
    NOMBRE = {"mensual": "monthly", "adaptativa": "adaptive",
              "fijo_insample": "in-sample fixed", "buy_and_hold": "buy & hold"}
    ORD_POL = ["mensual", "adaptativa", "fijo_insample", "buy_and_hold"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.8),
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    for c in u.columns:
        a1.step(u.index, u[c], where="mid", lw=2, marker="o", ms=4.5,
                color=COL.get(c), label=c)
    a1.set_ylim(-0.1, 2.1)
    a1.set_xlabel("evaluation year")
    a1.set_ylabel("threshold chosen (looking back 5 years only)")
    a1.set_title("1  |  What the adaptive policy chose, year by year",
                 loc="left", fontsize=11.5, pad=10)
    a1.legend(frameon=False, fontsize=9)
    a1.grid(True, alpha=0.25)

    ests = list(u.columns)
    x = np.arange(len(ests))
    ancho = 0.2
    for i, pol in enumerate(ORD_POL):
        sub = t[t["politica"] == pol].set_index("estrategia")
        vals = [sub.loc[e, "sharpe_5pb"] if e in sub.index else np.nan
                for e in ests]
        barras = a2.bar(x + (i - 1.5) * ancho, vals, ancho,
                        color=C_POL[pol], label=NOMBRE[pol])
        for b, v in zip(barras, vals):
            if not np.isnan(v):
                a2.text(b.get_x() + b.get_width() / 2, v + 0.012,
                        f"{v:.3f}", ha="center", fontsize=8)
    a2.set_xticks(x)
    a2.set_xticklabels(ests)
    a2.set_ylabel("Out-of-sample Sharpe (5 bps)")
    a2.set_title("2  |  Honest result: 2013-2026, identical starting portfolio",
                 loc="left", fontsize=11.5, pad=10)
    a2.legend(frameon=False, fontsize=9, ncol=2)
    a2.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIG / "validacion_rodante.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- 5
def risk_measures():
    t = pd.read_csv(RES / "metricas_riesgo.csv", index_col=0)
    medidas = ["sharpe", "sortino", "calmar", "martin"]
    rank = t[medidas].rank(ascending=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 6.2),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    x = list(range(len(medidas)))
    for e in rank.index:
        c = COL.get(e, "#111827")
        a1.plot(x, rank.loc[e].values, marker="o", ms=10, lw=2.4, color=c, zorder=3)
        a1.text(-0.14, rank.loc[e, medidas[0]], e, ha="right", va="center",
                fontsize=10, color=c, fontweight="bold")
        a1.text(x[-1] + 0.14, rank.loc[e, medidas[-1]], e, ha="left", va="center",
                fontsize=10, color=c, fontweight="bold")
    a1.invert_yaxis()
    a1.set_xlim(-1.6, x[-1] + 1.6)
    a1.set_xticks(x)
    a1.set_xticklabels(["SHARPE\n(volatility)", "SORTINO\n(semideviation)",
                        "CALMAR\n(max drawdown)", "MARTIN\n(ulcer index)"],
                       fontsize=9.5)
    a1.set_yticks(range(1, len(rank) + 1))
    a1.set_ylabel("rank (1 = best)")
    a1.set_title("1  |  If the lines cross, the yardstick changes the answer",
                 loc="left", fontsize=11.5, pad=12)
    a1.grid(True, axis="y", alpha=0.25)

    orden = t.sort_values("bajo_agua_%").index
    y = range(len(orden))
    a2.barh(list(y), t.loc[orden, "bajo_agua_%"], height=0.58,
            color=[COL.get(e, "#111827") for e in orden], alpha=0.9)
    for i, e in enumerate(orden):
        a2.text(t.loc[e, "bajo_agua_%"] + 0.8, i,
                f"{t.loc[e, 'bajo_agua_%']:.1f} %   "
                f"(longest streak: {t.loc[e, 'racha_anios']:.1f} years)",
                va="center", fontsize=9.5)
    a2.set_yticks(list(y))
    a2.set_yticklabels(orden, fontsize=10)
    a2.set_xlabel("% of time below the all-time high")
    a2.set_xlim(0, 125)
    a2.set_title("2  |  How long you spent underwater\n"
                 "     (what makes a real investor give up)",
                 loc="left", fontsize=11.5, pad=12)
    a2.grid(True, axis="x", alpha=0.25)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIG / "metricas_riesgo.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
for nombre, funcion in [
    ("curvas_capital.png", equity_curves),
    ("pesos_en_el_tiempo.png", weights_over_time),
    ("barrido_deriva.png", threshold_sweep),
    ("validacion_rodante.png", walk_forward),
    ("metricas_riesgo.png", risk_measures),
]:
    try:
        funcion()
        print(f"  OK   figuras/{nombre}")
    except Exception as e:
        print(f"  FALLO figuras/{nombre}  ->  {type(e).__name__}: {e}")

print(f"\nListo. Figuras en {FIG}")