"""Reglas de decisión de cada tipo de agente.

La demanda ``d`` de cada vínculo agente-empresa está en unidades adimensionales:
se convierte en una orden multiplicándola por el capital que el agente dedica a
esa empresa, su agresividad, su confianza y ``tamano_orden``.

* Fundamentalista: compra si su valoración supera el precio y se inclina por
  empresas de calidad (ROE alto, poca deuda).
* Chartista: sigue su regla técnica (momentum, medias, RSI, rupturas, soportes).
* Ruido: reacciona a noticias, al sentimiento y al máximo de 52 semanas.

Además, cada tipo comparte un ruido común por empresa (todos leen las mismas
noticias y los mismos gráficos): sin él, el ruido individual se cancelaría y
no habría volatilidad agregada.
"""

from __future__ import annotations

import numpy as np

from .agentes import CHARTISTA, FUNDAMENTALISTA, RUIDO, Poblacion
from .config import DIAS_HABILES, Config
from .economia import Economia


def valoracion_objetivo(pob: Poblacion, eco: Economia, precios: np.ndarray) -> np.ndarray:
    """log del valor que cada agente asignaría hoy a cada empresa que sigue, sin sentimiento.

    Los fundamentalistas DCF mezclan el crecimiento de largo plazo con el del
    último año (extrapolación); los de múltiplos aplican el PER mediano del
    sector, que contagia las burbujas sectoriales.
    """
    a, e = pob.agente, pob.empresa
    w = pob.extrapolacion[a]
    g = (1.0 - w) * eco.crecimiento[e] + w * eco.crecimiento_reciente[e]
    dcf = np.log(eco.valor_publicado(g, e))
    relativa = (np.log(eco.bpa_ttm) + eco.per_mediano_sector(precios))[e]
    base = np.where(pob.usa_multiplos[a], relativa, dcf)
    return base + pob.sesgo[a] + pob.opinion


def actualizar_valoraciones(pob: Poblacion, objetivo: np.ndarray) -> None:
    """Conservadurismo: la valoración se ancla a la anterior y se ajusta poco a poco."""
    pob.log_valor += pob.velocidad_ajuste[pob.agente] * (objetivo - pob.log_valor)


def ruido_comun(config: Config, sector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ruido común por tipo y empresa (3×M) con componentes de mercado, sector e idiosincrático."""
    ce = config.estrategias
    n_sectores = sector.max() + 1
    propio = np.sqrt(max(1.0 - ce.ruido_mercado - ce.ruido_sector, 0.0))
    z = (
        np.sqrt(ce.ruido_mercado) * rng.normal(size=(3, 1))
        + np.sqrt(ce.ruido_sector) * rng.normal(size=(3, n_sectores))[:, sector]
        + propio * rng.normal(size=(3, sector.size))
    )
    return np.asarray(ce.ruido)[:, None] * z


def demandas(
    config: Config,
    pob: Poblacion,
    log_precio: np.ndarray,
    sentimiento: np.ndarray,
    calidad: np.ndarray,
    senales: np.ndarray,
    noticias: np.ndarray,
    maximo_anual: np.ndarray,
    vol_habitual: np.ndarray,
    ruido: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Demanda de cada vínculo según el tipo actual de su agente."""
    ce, cc = config.estrategias, config.comportamiento
    a, e = pob.agente, pob.empresa

    percibido = pob.log_valor + cc.efecto_sentimiento_valor * pob.sensibilidad[a] * sentimiento[e]
    d_fund = ce.phi * (percibido - log_precio[e]) + ce.peso_calidad * calidad[e] + ruido[FUNDAMENTALISTA, e]
    d_chart = ce.chi * senales[pob.regla[a], e] + ruido[CHARTISTA, e]

    d_ruido = ruido[RUIDO] + ce.reaccion_noticias * noticias + cc.efecto_sentimiento_ruido * sentimiento
    if cc.anclaje:
        # Cerca del máximo anual cuesta pagar más; lejos de él, "está barata". Lo
        # normal es estar algo por debajo del máximo, tanto más cuanto más volátil es
        distancia = maximo_anual - log_precio
        neutra = cc.anclaje_distancia_neutra * vol_habitual * np.sqrt(DIAS_HABILES)
        d_ruido = d_ruido + cc.anclaje_maximo * np.tanh((distancia - neutra) / 0.15)

    tipo = pob.tipo[a]
    d = np.where(tipo == FUNDAMENTALISTA, d_fund, np.where(tipo == CHARTISTA, d_chart, d_ruido[e]))
    return d + ce.ruido_individual * rng.normal(size=d.size)
