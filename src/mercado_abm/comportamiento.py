"""Capa 3: la irracionalidad.

Cada sesgo se puede desactivar desde :class:`ConfigComportamiento` para medir su
efecto sobre los hechos estilizados (ver ``mercado-abm comparar``).
"""

from __future__ import annotations

import numpy as np

from .config import ConfigComportamiento


class Sentimiento:
    """Ánimo del mercado, acotado en (-1, 1), que se alimenta de los retornos recientes.

    Hay un sentimiento global y uno por sector. Persiste (euforia y pánico duran)
    y sesga las valoraciones y la demanda de los agentes de ruido.
    """

    def __init__(self, cfg: ConfigComportamiento, n_sectores: int, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.global_ = 0.0
        self.sector = np.zeros(n_sectores)

    def actualizar(self, retorno_mercado: float, retornos_sector: np.ndarray) -> None:
        if not self.cfg.sentimiento:
            return
        c = self.cfg
        self.global_ = float(
            np.tanh(
                c.persistencia_sentimiento * np.arctanh(np.clip(self.global_, -0.999, 0.999))
                + c.reaccion_sentimiento * retorno_mercado
                + c.ruido_sentimiento * self.rng.normal()
            )
        )
        self.sector = np.tanh(
            c.persistencia_sentimiento * np.arctanh(np.clip(self.sector, -0.999, 0.999))
            + c.reaccion_sentimiento * retornos_sector
            + c.ruido_sentimiento * self.rng.normal(size=self.sector.size)
        )

    def por_empresa(self, sector: np.ndarray) -> np.ndarray:
        return self.global_ + 0.5 * self.sector[sector]


def probabilidad_atencion(
    cfg: ConfigComportamiento,
    vol: np.ndarray,
    vol_habitual: np.ndarray,
    noticias: np.ndarray,
    confianza: np.ndarray,
    agente: np.ndarray,
    empresa: np.ndarray,
) -> np.ndarray:
    """Probabilidad de que cada agente opere hoy cada empresa que sigue (por vínculo).

    Con atención limitada, se mira más lo que se mueve de forma inusual (respecto
    a la volatilidad a la que uno está acostumbrado) y lo que sale en las
    noticias. ``vol`` es la volatilidad percibida, en la que las caídas pesan más
    si hay aversión a las pérdidas. Los agentes con exceso de confianza operan más.
    """
    if cfg.atencion_limitada:
        a = np.abs(noticias)
        sorpresa_vol = (vol / np.maximum(vol_habitual, 1e-6)) ** cfg.atencion_sensibilidad
        base = cfg.atencion_base * sorpresa_vol + cfg.atencion_noticias * a / (1.0 + a)
    else:
        base = np.full(vol.shape, cfg.atencion_fija)
    return np.clip(base[empresa] * np.sqrt(confianza)[agente], 0.0, cfg.atencion_max)


class CambioEstrategia:
    """Elección discreta de estrategia (Brock-Hommes) con contagio (Kirman).

    La aptitud de cada estrategia es su rendimiento especulativo reciente,
    saturado con tanh para que unos pocos días extremos no lo decidan todo; con
    aversión a las pérdidas, un día malo pesa ``aversion`` veces más que uno bueno.
    Cuando los precios se alejan mucho del valor de consenso, crece el miedo a
    una corrección y el atractivo de la estrategia fundamentalista.
    """

    def __init__(self, cfg: ConfigComportamiento, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.aptitud = np.zeros(3)

    def actualizar_aptitud(self, rendimiento: np.ndarray) -> None:
        c = self.cfg
        x = np.tanh(rendimiento / c.escala_rendimiento)
        valor = np.where(x < 0, c.aversion * x, x) if c.aversion_perdidas else x
        self.aptitud = c.memoria * self.aptitud + (1.0 - c.memoria) * valor

    def probabilidades(self, fracciones: np.ndarray, desalineacion: float) -> np.ndarray:
        """Probabilidad de elegir cada estrategia al revisar (logit sobre el atractivo).

        El contagio entra dentro del logit: cuantos más agentes usan una estrategia,
        más atractiva parece. Con intensidad suficiente aparecen regímenes que se
        refuerzan solos hasta que el desalineamiento los rompe.
        """
        c = self.cfg
        atractivo = c.intensidad_eleccion * self.aptitud + np.asarray(c.predisposicion)
        if c.contagio:
            atractivo = atractivo + c.contagio_intensidad * fracciones
        atractivo[0] += c.miedo_burbuja * desalineacion
        p = np.exp(atractivo - atractivo.max())
        p /= p.sum()
        return (1.0 - c.exploracion) * p + c.exploracion / 3.0

    def revisar(self, tipo: np.ndarray, rendimiento_medio: np.ndarray, bloqueados: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Una fracción de agentes reconsidera su estrategia; los que pierden, más a menudo."""
        c = self.cfg
        if not c.cambio_estrategia:
            return tipo
        prob = c.tasa_revision * (1.0 + c.revision_por_perdidas * (rendimiento_medio < 0))
        revisa = (self.rng.random(tipo.size) < prob) & ~bloqueados
        tipo = tipo.copy()
        tipo[revisa] = self.rng.choice(3, revisa.sum(), p=p)
        return tipo


def actualizar_confianza(
    cfg: ConfigComportamiento, confianza: np.ndarray, innata: np.ndarray, rendimiento: np.ndarray
) -> np.ndarray:
    """Sesgo de autoatribución: los aciertos suben la confianza más de lo que los fallos la bajan.

    ``rendimiento`` es la media móvil del rendimiento especulativo de cada agente.
    Se mide frente al resto (dividido por su dispersión), así que la volatilidad
    por sí sola no infla la confianza; un mercado alcista en el que casi todos
    aciertan, sí. Con el tiempo la confianza vuelve a su nivel innato.
    """
    if not cfg.exceso_confianza:
        return confianza
    x = np.tanh(rendimiento / (rendimiento.std() + 1e-12))
    aprendizaje = np.where(x > 0, cfg.confianza_aprendizaje_ganancias, cfg.confianza_aprendizaje_perdidas)
    confianza = confianza - cfg.confianza_reversion * (confianza - innata) + aprendizaje * x
    return np.clip(confianza, 0.5, cfg.confianza_max)


def ordenes_disposicion(
    disposicion: np.ndarray,
    n_esp: np.ndarray,
    coste: np.ndarray,
    log_precio: np.ndarray,
    agente: np.ndarray,
    empresa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Efecto disposición sobre las posiciones especulativas (por vínculo).

    Devuelve (orden en importe para realizar ganancias, máscara de posiciones en
    pérdidas). Las posiciones ganadoras se cierran antes; las perdedoras se
    mantienen más tiempo (eso lo aplica quien llama).
    """
    abierta = n_esp != 0
    lp = log_precio[empresa]
    ganancia = np.where(abierta, np.sign(n_esp) * (lp - np.log(np.where(abierta, coste, 1.0))), 0.0)
    importe = n_esp * np.exp(lp)
    realizar = -disposicion[agente] * np.tanh(np.maximum(ganancia, 0.0) / 0.1) * importe / 5.0
    return realizar, ganancia < 0


def llamadas_de_margen(
    cfg: ConfigComportamiento,
    exposicion: np.ndarray,
    riqueza: np.ndarray,
    limite: np.ndarray,
    riqueza_referencia: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fracción de cartera que cada agente debe vender mañana y quién quiebra.

    Si la exposición supera el límite más un colchón, el bróker obliga a vender
    hasta volver a ``objetivo_desapalancamiento`` del límite, como mucho
    ``liquidacion_diaria_max`` de la cartera al día. Quien quiebra lo vende todo.
    Estas ventas mueven el precio y pueden disparar más llamadas: así se forman
    las cascadas.
    """
    quiebra = riqueza < cfg.umbral_quiebra * riqueza_referencia
    fraccion = np.zeros_like(riqueza)
    if cfg.apalancamiento:
        tope = limite * np.maximum(riqueza, 0.0)
        llamada = exposicion > tope * (1.0 + cfg.colchon_margen)
        objetivo = cfg.objetivo_desapalancamiento * tope
        fraccion = np.where(llamada, 1.0 - objetivo / np.maximum(exposicion, 1e-12), 0.0)
        fraccion = np.clip(fraccion, 0.0, cfg.liquidacion_diaria_max)
    fraccion = np.where(quiebra, 1.0, fraccion)
    return fraccion, quiebra
