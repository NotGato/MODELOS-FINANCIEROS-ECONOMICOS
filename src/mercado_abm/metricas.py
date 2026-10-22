"""Hechos estilizados: las regularidades estadísticas que comparte cualquier mercado real.

El modelo no se valida prediciendo precios, sino reproduciendo estos patrones
(Cont, 2001). Los rangos son orientativos para datos diarios de acciones e
índices desarrollados y sirven para ver de un vistazo si el modelo es realista.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DIAS_HABILES
from .simulacion import Resultados


@dataclass
class Hecho:
    nombre: str
    valor: float
    rango: tuple[float, float]
    explicacion: str

    @property
    def cumple(self) -> bool:
        return self.rango[0] <= self.valor <= self.rango[1]


def curtosis(x: np.ndarray) -> float:
    z = (x - x.mean()) / x.std()
    return float((z**4).mean())


def asimetria(x: np.ndarray) -> float:
    z = (x - x.mean()) / x.std()
    return float((z**3).mean())


def autocorrelacion(x: np.ndarray, retardo: int) -> float:
    return float(np.corrcoef(x[:-retardo], x[retardo:])[0, 1])


def indice_hill(x: np.ndarray, cola: float = 0.05) -> float:
    """Índice de cola de Hill sobre las pérdidas: ~3 en mercados reales ("ley cúbica")."""
    perdidas = np.sort(-x[x < 0])[::-1]
    k = max(int(cola * x.size), 10)
    extremos = perdidas[: k + 1]
    return float(1.0 / np.mean(np.log(extremos[:k] / extremos[k])))


def efecto_apalancamiento(r: np.ndarray, horizonte: int = 10) -> float:
    """Correlación media entre el retorno de hoy y la volatilidad de los próximos días.

    Es negativa en los mercados reales: las caídas suben la volatilidad más que las subidas.
    """
    return float(np.mean([np.corrcoef(r[:-k], np.abs(r[k:]))[0, 1] for k in range(1, horizonte + 1)]))


def maxima_caida(log_precio: np.ndarray) -> float:
    return float(1.0 - np.exp(-(np.maximum.accumulate(log_precio) - log_precio).max()))


def _por_empresa(f, r: np.ndarray) -> float:
    return float(np.median([f(r[:, j]) for j in range(r.shape[1])]))


def hechos_estilizados(res: Resultados) -> list[Hecho]:
    b = res.calentamiento
    r = res.retornos[b:]
    ri = res.retornos_indice()[b:]
    volumen = res.volumen[b:]
    correl = np.corrcoef(r.T)[np.triu_indices(r.shape[1], 1)]
    desalineamiento = np.abs(res.log_precio[b + 1 :] - res.log_valor[b:])
    raiz = np.sqrt(DIAS_HABILES)

    return [
        Hecho("Volatilidad anual (acciones, mediana)", float(np.median(r.std(axis=0))) * raiz, (0.15, 0.45),
              "Las acciones individuales oscilan un 20-40% al año."),
        Hecho("Volatilidad anual (índice)", float(ri.std()) * raiz, (0.10, 0.25),
              "La diversificación la reduce, pero no del todo: hay un factor común."),
        Hecho("Autocorrelación de retornos (retardo 1)", _por_empresa(lambda x: autocorrelacion(x, 1), r), (-0.05, 0.05),
              "Casi impredecible a corto plazo."),
        Hecho("Curtosis diaria (acciones, mediana)", _por_empresa(curtosis, r), (5.0, 60.0),
              "Colas pesadas: una normal tendría 3."),
        Hecho("Curtosis diaria (índice)", curtosis(ri), (4.0, 60.0), "Los cracs viven en estas colas."),
        Hecho("Índice de cola de Hill (acciones)", _por_empresa(indice_hill, r), (2.0, 5.0),
              "Ley cúbica de los retornos: en torno a 3."),
        Hecho("Asimetría (índice)", asimetria(ri), (-2.0, 0.0), "Los cracs son más bruscos que las subidas."),
        Hecho("Agrupación de volatilidad: autocorr. |r| retardo 1 (índice)", autocorrelacion(np.abs(ri), 1), (0.10, 0.45),
              "A días agitados les siguen días agitados."),
        Hecho("Memoria larga: autocorr. |r| retardo 50 (índice)", autocorrelacion(np.abs(ri), 50), (0.02, 0.25),
              "La agrupación decae muy despacio."),
        Hecho("Efecto apalancamiento (índice)", efecto_apalancamiento(ri), (-0.3, -0.005),
              "Tras las caídas sube la volatilidad."),
        Hecho("Correlación volumen-|retorno| (acciones)",
              float(np.median([np.corrcoef(volumen[:, j], np.abs(r[:, j]))[0, 1] for j in range(r.shape[1])])), (0.15, 0.9),
              "Se negocia más cuando el precio se mueve mucho."),
        Hecho("Correlación media entre acciones", float(correl.mean()), (0.15, 0.55),
              "Mercado, sector y sentimiento comunes."),
        Hecho("Desalineamiento mediano |ln(P/V)|", float(np.median(desalineamiento)), (0.05, 0.40),
              "Los precios se alejan del valor durante años (Shiller)."),
        Hecho("Máxima caída del índice", maxima_caida(res.log_indice()[b:]), (0.25, 0.75),
              "En 25 años suele haber al menos un mercado bajista severo."),
    ]


def informe(hechos: list[Hecho]) -> str:
    ancho = max(len(h.nombre) for h in hechos)
    lineas = [f"{'Hecho estilizado':<{ancho}}  {'Valor':>8}  {'Rango real':>15}", "-" * (ancho + 32)]
    for h in hechos:
        marca = "ok" if h.cumple else "--"
        rango = f"[{h.rango[0]:g}, {h.rango[1]:g}]"
        lineas.append(f"{h.nombre:<{ancho}}  {h.valor:>8.3f}  {rango:>15}  {marca}")
    cumplidos = sum(h.cumple for h in hechos)
    lineas.append(f"\nReproduce {cumplidos} de {len(hechos)} hechos estilizados.")
    return "\n".join(lineas)


def resumen_mercado(res: Resultados) -> dict[str, float]:
    """Cifras de contexto que no son hechos estilizados pero ayudan a leer una simulación."""
    b = res.calentamiento
    anos = (res.dias - b) / DIAS_HABILES
    ri = res.retornos_indice()[b:]
    log_valor = res.log_valor_indice()
    return {
        "crecimiento_anual_indice": float(ri.sum() / anos),
        "crecimiento_anual_valor": float((log_valor[-1] - log_valor[b]) / anos),
        "peor_dia_indice": float(ri.min()),
        "dias_caida_mayor_5pct": int((ri < -0.05).sum()),
        "fraccion_fundamentalistas": float(res.fracciones[b:, 0].mean()),
        "fraccion_chartistas": float(res.fracciones[b:, 1].mean()),
        "fraccion_ruido": float(res.fracciones[b:, 2].mean()),
        "llamadas_de_margen": int(res.llamadas_margen.sum()),
        "quiebras": int(res.quiebras.sum()),
    }
