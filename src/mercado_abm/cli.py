"""Línea de comandos: ``mercado-abm simular | comparar | graficos``."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import SESGOS_FASE_2, Config, actualizar, cargar, fase1
from .metricas import hechos_estilizados, informe, resumen_mercado
from .simulacion import Resultados, simular


def _config(args) -> Config:
    config = cargar(args.config)
    if getattr(args, "fase", None) == 1:
        fase1(config)
    if args.semilla is not None:
        config.simulacion.semilla = args.semilla
    if args.dias is not None:
        config.simulacion.dias = args.dias
    if args.agentes is not None:
        config.agentes.n_agentes = args.agentes
    return config


def _progreso(dia: int, total: int) -> None:
    print(f"\r  año {dia // 252:>3} de {total // 252}", end="", file=sys.stderr, flush=True)


def cmd_simular(args) -> None:
    config = _config(args)
    print(f"Simulando {config.simulacion.dias} días con {config.agentes.n_agentes} agentes "
          f"y {config.economia.n_sectores * config.economia.empresas_por_sector} empresas...", file=sys.stderr)
    inicio = time.time()
    res = simular(config, _progreso)
    print(f"\r  hecho en {time.time() - inicio:.0f} s" + " " * 20, file=sys.stderr)

    print(informe(hechos_estilizados(res)))
    print()
    for clave, valor in resumen_mercado(res).items():
        print(f"{clave:28s} {valor:.4g}" if isinstance(valor, float) else f"{clave:28s} {valor}")

    carpeta = Path(args.salida)
    res.guardar(carpeta)
    from .graficos import exportar_csv

    exportar_csv(res, carpeta)
    if not args.sin_graficos:
        from .graficos import generar_graficos

        generar_graficos(res, carpeta)
    print(f"\nResultados en {carpeta}/")


def cmd_graficos(args) -> None:
    from .graficos import generar_graficos

    res = Resultados.cargar(args.carpeta)
    for ruta in generar_graficos(res, args.carpeta):
        print(ruta)


# --------------------------------------------------------------------------- comparar
def _variantes() -> dict[str, dict]:
    """Fase 1, fase 2 completa y fase 2 quitando cada sesgo de uno en uno."""
    sin_fase2 = {s: False for s in SESGOS_FASE_2}
    variantes = {"fase 1": {"comportamiento": sin_fase2}, "fase 2": {}}
    for sesgo in SESGOS_FASE_2:
        variantes[f"fase 2 sin {sesgo.replace('_', ' ')}"] = {"comportamiento": {sesgo: False}}
    return variantes


def _metricas_variante(tarea) -> dict[str, float]:
    ruta, cambios, semilla, dias, agentes = tarea
    config = cargar(ruta)
    actualizar(config, cambios)
    config.simulacion.semilla = semilla
    if dias:
        config.simulacion.dias = dias
    if agentes:
        config.agentes.n_agentes = agentes
    res = simular(config)
    hechos = {h.nombre: h.valor for h in hechos_estilizados(res)}
    resumen = resumen_mercado(res)
    return {
        "vol. índice": hechos["Volatilidad anual (índice)"],
        "curtosis índice": hechos["Curtosis diaria (índice)"],
        "agrup. vol.": hechos["Agrupación de volatilidad: autocorr. |r| retardo 1 (índice)"],
        "asimetría": hechos["Asimetría (índice)"],
        "efecto apal.": hechos["Efecto apalancamiento (índice)"],
        "desalin.": hechos["Desalineamiento mediano |ln(P/V)|"],
        "máx. caída": hechos["Máxima caída del índice"],
        "peor día": resumen["peor_dia_indice"],
        "días <-5%": resumen["dias_caida_mayor_5pct"],
        "% chartistas": 100 * resumen["fraccion_chartistas"],
        "quiebras": resumen["quiebras"],
    }


def cmd_comparar(args) -> None:
    semillas = [int(s) for s in args.semillas.split(",")]
    variantes = _variantes()
    tareas = [(args.config, cambios, s, args.dias, args.agentes) for cambios in variantes.values() for s in semillas]
    procesos = args.procesos or os.cpu_count() or 1
    print(f"Comparando {len(variantes)} variantes × {len(semillas)} semillas en {procesos} procesos...", file=sys.stderr)
    with ProcessPoolExecutor(procesos) as ex:
        resultados = list(ex.map(_metricas_variante, tareas))

    columnas = list(resultados[0])
    ancho = max(len(v) for v in variantes)
    print(f"{'variante':<{ancho}}  " + "  ".join(f"{c:>15}" for c in columnas))
    for i, nombre in enumerate(variantes):
        filas = resultados[i * len(semillas) : (i + 1) * len(semillas)]
        medianas = [np.median([f[c] for f in filas]) for c in columnas]
        print(f"{nombre:<{ancho}}  " + "  ".join(f"{m:>15.3f}" for m in medianas))
    print(f"\nMedianas de {len(semillas)} semillas. Rangos reales orientativos: curtosis 5-20, agrupación 0.1-0.4, "
          "asimetría y efecto apalancamiento negativos.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mercado-abm", description="Mercado financiero artificial basado en agentes")
    sub = parser.add_subparsers(dest="orden", required=True)

    def comunes(p, con_fase=True):
        p.add_argument("--config", help="fichero TOML con parámetros (por defecto, los de config.py)")
        if con_fase:
            p.add_argument("--fase", type=int, choices=(1, 2), default=2, help="1 = sin los sesgos de la fase 2")
        p.add_argument("--semilla", type=int)
        p.add_argument("--dias", type=int, help="días de simulación (252 = 1 año)")
        p.add_argument("--agentes", type=int)

    p = sub.add_parser("simular", help="ejecuta una simulación, imprime los hechos estilizados y guarda gráficos")
    comunes(p)
    p.add_argument("--salida", default="resultados", help="carpeta de salida")
    p.add_argument("--sin-graficos", action="store_true")
    p.set_defaults(func=cmd_simular)

    p = sub.add_parser("comparar", help="mide el efecto de cada sesgo de la fase 2 con varias semillas")
    p.add_argument("--config")
    p.add_argument("--semillas", default="1,2,3,4")
    p.add_argument("--dias", type=int)
    p.add_argument("--agentes", type=int)
    p.add_argument("--procesos", type=int)
    p.set_defaults(func=cmd_comparar)

    p = sub.add_parser("graficos", help="regenera los gráficos de una simulación guardada")
    p.add_argument("carpeta")
    p.set_defaults(func=cmd_graficos)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
