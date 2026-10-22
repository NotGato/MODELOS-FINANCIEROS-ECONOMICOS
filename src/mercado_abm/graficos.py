"""Gráficos de una simulación (PNG con matplotlib) y exportación de las series a CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .agentes import TIPOS
from .config import DIAS_HABILES
from .simulacion import Resultados

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CRITICO = "#d03b3b"

ETIQUETAS_TIPOS = ("Fundamentalistas", "Chartistas", "Ruido")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.edgecolor": EJE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": TINTA_SECUNDARIA,
            "axes.titlecolor": TINTA,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": REJILLA,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": TENUE,
            "ytick.color": TENUE,
            "xtick.labelcolor": TINTA_SECUNDARIA,
            "ytick.labelcolor": TINTA_SECUNDARIA,
            "legend.frameon": False,
            "legend.labelcolor": TINTA_SECUNDARIA,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )
    return plt


def _eje_log(ax) -> None:
    """Escala logarítmica con números legibles (20, 50, 100...) en lugar de potencias de 10."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _anos(n: int, inicio: int = 0) -> np.ndarray:
    return (np.arange(n) + inicio) / DIAS_HABILES


def _etiqueta_final(ax, x, y, texto):
    """Etiqueta directa al final de la línea, en tinta (el color lo pone la marca)."""
    ax.annotate(texto, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=TINTA_SECUNDARIA)


def grafico_mercado(res: Resultados, ruta: Path) -> Path:
    plt = _plt()
    hay_margen = res.llamadas_margen.sum() + res.quiebras.sum() > 0
    filas = 5 if hay_margen else 4
    alturas = [3, 1.6, 1.8, 1.4] + ([1.2] if hay_margen else [])
    fig, axes = plt.subplots(filas, 1, figsize=(10, 2.1 * sum(alturas) / 1.6), sharex=True, gridspec_kw={"height_ratios": alturas})

    t_precio = _anos(res.dias + 1)
    t = _anos(res.dias)
    indice = 100 * np.exp(res.log_indice())
    valor = 100 * np.exp(res.log_valor_indice())

    ax = axes[0]
    ax.plot(t_precio, indice, color=SERIES[0], label="Índice (precio)")
    ax.plot(t, valor, color=SERIES[1], label="Valor fundamental agregado")
    _eje_log(ax)
    ax.set_title("Índice de mercado frente a su valor fundamental (base 100, escala log)")
    ax.legend(loc="upper left", ncols=2)
    # Etiquetas finales solo si no se pisan; si no, la leyenda basta
    if abs(np.log(indice[-1] / valor[-1])) > 0.08:
        _etiqueta_final(ax, t_precio, indice, f"{indice[-1]:.0f}")
        _etiqueta_final(ax, t, valor, f"{valor[-1]:.0f}")

    ax = axes[1]
    ax.plot(t, 100 * res.retornos_indice(), color=SERIES[0], linewidth=0.6)
    ax.set_title("Retornos diarios del índice (%): la volatilidad llega en racimos")

    ax = axes[2]
    for k in range(3):
        ax.plot(t, 100 * res.fracciones[:, k], color=SERIES[k], label=ETIQUETAS_TIPOS[k])
    ax.set_ylim(0, 100)
    ax.set_title("Estrategias en uso (% de agentes)")
    ax.legend(loc="upper left", ncols=3)

    ax = axes[3]
    ax.plot(t, res.sentimiento, color=SERIES[0])
    ax.axhline(0, color=EJE, linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_title("Sentimiento del mercado (−1 pánico, +1 euforia)")

    if hay_margen:
        ax = axes[4]
        ax.bar(t, res.llamadas_margen + res.quiebras, width=4 / DIAS_HABILES, color=CRITICO)
        ax.set_title("Llamadas de margen y quiebras por día (agentes)")

    axes[-1].set_xlabel("Años")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def grafico_empresas(res: Resultados, ruta: Path, n: int = 6) -> Path:
    plt = _plt()
    # Una empresa de cada sector y, si sobra hueco, las siguientes
    orden = [int(np.flatnonzero(res.sector == s)[0]) for s in range(len(res.sectores))]
    orden += [j for j in range(len(res.nombres)) if j not in orden]
    elegidas = orden[:n]
    columnas = 3
    filas = int(np.ceil(len(elegidas) / columnas))
    fig, axes = plt.subplots(filas, columnas, figsize=(11, 2.9 * filas), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    t_precio = _anos(res.dias + 1)
    t = _anos(res.dias)
    for ax, j in zip(axes, elegidas):
        ax.plot(t_precio, np.exp(res.log_precio[:, j]), color=SERIES[0], label="Precio")
        ax.plot(t, np.exp(res.log_valor[:, j]), color=SERIES[1], label="Valor verdadero")
        ax.plot(t, np.exp(res.log_consenso[:, j]), color=SERIES[2], linewidth=1.0, label="Valoración media de los agentes")
        _eje_log(ax)
        ax.set_title(f"{res.nombres[j]} · {res.sectores[res.sector[j]]}", fontsize=9.5)
    for ax in axes[len(elegidas) :]:
        ax.set_visible(False)
    manejadores, etiquetas = axes[0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Precio, valor verdadero y valoración media por empresa (escala log)", x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold", color=TINTA)
    for ax in axes[-columnas:]:
        ax.set_xlabel("Años")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


def grafico_hechos(res: Resultados, ruta: Path) -> Path:
    plt = _plt()
    b = res.calentamiento
    r = res.retornos[b:]
    z = ((r - r.mean(axis=0)) / r.std(axis=0)).ravel()
    ri = res.retornos_indice()[b:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    bordes = np.linspace(-10, 10, 81)
    densidad, _ = np.histogram(z, bins=bordes, density=True)
    centros = 0.5 * (bordes[1:] + bordes[:-1])
    visibles = densidad > 0
    ax1.plot(centros[visibles], densidad[visibles], color=SERIES[0], marker="o", markersize=3.5, label="Simulación (todas las acciones)")
    ax1.plot(centros, np.exp(-0.5 * centros**2) / np.sqrt(2 * np.pi), color=TENUE, label="Normal")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-6, 1)
    ax1.set_xlabel("Retorno diario estandarizado (desviaciones típicas)")
    ax1.set_title("Colas pesadas: los días extremos son mucho más frecuentes")
    ax1.legend(loc="lower center")

    retardos = np.arange(1, 101)
    acf_r = [np.corrcoef(ri[:-k], ri[k:])[0, 1] for k in retardos]
    acf_abs = [np.corrcoef(np.abs(ri[:-k]), np.abs(ri[k:]))[0, 1] for k in retardos]
    ax2.plot(retardos, acf_r, color=SERIES[0], label="Retornos r")
    ax2.plot(retardos, acf_abs, color=SERIES[1], label="Retornos absolutos |r|")
    ax2.axhline(0, color=EJE, linewidth=0.8)
    ax2.set_xlabel("Retardo (días)")
    ax2.set_title("Autocorrelación del índice: r no se predice, |r| sí")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def grafico_ratios(res: Resultados, ruta: Path) -> Path:
    plt = _plt()
    t = _anos(res.dias)
    per = np.exp(res.log_precio[1:]) / res.bpa_ttm
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for s, nombre in enumerate(res.sectores):
        mediana = np.median(per[:, res.sector == s], axis=1)
        ax.plot(t, mediana, color=SERIES[s % len(SERIES)], label=nombre)
    _eje_log(ax)
    ax.set_xlabel("Años")
    ax.set_title("PER mediano por sector (precio / beneficio de los últimos 12 meses)")
    ax.legend(loc="upper left", ncols=len(res.sectores))
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def generar_graficos(res: Resultados, carpeta: str | Path) -> list[Path]:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    return [
        grafico_mercado(res, carpeta / "mercado.png"),
        grafico_empresas(res, carpeta / "empresas.png"),
        grafico_hechos(res, carpeta / "hechos_estilizados.png"),
        grafico_ratios(res, carpeta / "ratios.png"),
    ]


def exportar_csv(res: Resultados, carpeta: str | Path) -> list[Path]:
    """Series diarias en CSV: la vista en tabla de los gráficos."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    mercado = carpeta / "mercado.csv"
    log_indice = res.log_indice()
    log_valor = res.log_valor_indice()
    with open(mercado, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dia", "indice", "valor_fundamental", "tipo_interes", "sentimiento",
                    *[f"frac_{t}" for t in TIPOS], "llamadas_margen", "quiebras", "confianza_media"])
        for t in range(res.dias):
            w.writerow([t + 1, f"{100 * np.exp(log_indice[t + 1]):.4f}", f"{100 * np.exp(log_valor[t]):.4f}",
                        f"{res.tasa[t]:.5f}", f"{res.sentimiento[t]:.4f}", *[f"{x:.4f}" for x in res.fracciones[t]],
                        int(res.llamadas_margen[t]), int(res.quiebras[t]), f"{res.confianza[t]:.4f}"])

    rutas = [mercado]
    for nombre, datos in (
        ("precios.csv", np.exp(res.log_precio[1:])),
        ("valor_verdadero.csv", np.exp(res.log_valor)),
        ("per.csv", np.exp(res.log_precio[1:]) / res.bpa_ttm),
    ):
        ruta = carpeta / nombre
        with open(ruta, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dia", *res.nombres])
            for t in range(res.dias):
                w.writerow([t + 1, *[f"{x:.4f}" for x in datos[t]]])
        rutas.append(ruta)
    return rutas
