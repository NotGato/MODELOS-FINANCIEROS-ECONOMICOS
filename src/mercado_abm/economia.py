"""Capa 1: el mundo fundamental.

Cada empresa tiene ventas por acción, margen neto, valor contable y deuda que
evolucionan a diario con shocks macro (ciclo económico), sectoriales e
idiosincráticos. Los agentes solo ven los resultados trimestrales publicados,
con ruido contable; el valor "verdadero" usa la información real y nunca se
les muestra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DIAS_HABILES, DIAS_TRIMESTRE, ConfigEconomia


@dataclass(frozen=True)
class Sector:
    nombre: str
    codigo: str
    crecimiento: float  # crecimiento anual de las ventas a largo plazo
    margen: float  # margen neto medio
    roe: float  # rentabilidad sobre fondos propios a largo plazo
    beta: float
    deuda: float  # deuda / patrimonio objetivo
    ciclicidad: float  # sensibilidad de ventas y márgenes al ciclo
    vol: float  # volatilidad anual de las ventas


SECTORES = (
    Sector("Tecnología", "TEC", 0.060, 0.18, 0.22, 1.30, 0.3, 0.8, 0.18),
    Sector("Consumo", "CON", 0.030, 0.08, 0.15, 0.80, 0.6, 0.3, 0.08),
    Sector("Industria", "IND", 0.035, 0.10, 0.13, 1.10, 0.8, 1.0, 0.12),
    Sector("Financiero", "FIN", 0.030, 0.20, 0.11, 1.20, 1.5, 0.8, 0.12),
    Sector("Energía", "ENE", 0.020, 0.12, 0.12, 1.00, 0.7, 1.5, 0.20),
)


def valor_gordon(bpa, crecimiento, roe, coste_capital):
    """Valor por acción con Gordon ajustado por reinversión.

    V = BPA·(1+g)·(1 − g/ROE) / (r − g): para crecer al ritmo g hay que reinvertir
    g/ROE del beneficio, así que crecer solo crea valor si ROE > r.
    """
    g = np.clip(crecimiento, -0.05, coste_capital - 0.015)
    reparto = np.clip(1.0 - g / np.maximum(roe, 1e-3), 0.2, 1.0)
    return bpa * (1.0 + g) * reparto / (coste_capital - g)


def valor_dos_etapas(bpa, crecimiento_corto, crecimiento_largo, roe, coste_capital, anos=5):
    """DCF en dos etapas: ``anos`` años al crecimiento percibido y después Gordon.

    Es mucho menos sensible que Gordon a un crecimiento reciente extraordinario,
    porque ese crecimiento solo dura la primera etapa. Si ambos crecimientos
    coinciden, el resultado es exactamente :func:`valor_gordon`.
    """
    g1 = np.clip(crecimiento_corto, -0.2, 0.3)
    reparto1 = np.clip(1.0 - g1 / np.maximum(roe, 1e-3), 0.2, 1.0)
    crec = (1.0 + g1) / (1.0 + coste_capital)
    descuento = crec ** np.arange(1, anos + 1)[:, None] if np.ndim(crec) else crec ** np.arange(1, anos + 1)
    primera = bpa * reparto1 * np.sum(descuento, axis=0)
    terminal = valor_gordon(bpa * (1.0 + g1) ** anos, crecimiento_largo, roe, coste_capital)
    return primera + terminal / (1.0 + coste_capital) ** anos


class Economia:
    def __init__(self, cfg: ConfigEconomia, rng: np.random.Generator):
        if cfg.n_sectores > len(SECTORES):
            raise ValueError(f"Como máximo hay {len(SECTORES)} sectores definidos")
        self.cfg = cfg
        self.rng = rng
        self.sectores = SECTORES[: cfg.n_sectores]
        k, e = cfg.n_sectores, cfg.empresas_por_sector
        m = k * e
        self.n_empresas = m
        self.sector = np.repeat(np.arange(k), e)
        self.nombres = [f"{s.codigo}{i + 1}" for s in self.sectores for i in range(e)]

        def por_sector(attr):
            return np.array([getattr(self.sectores[s], attr) for s in self.sector])

        # Parámetros propios de cada empresa (heterogeneidad dentro del sector)
        self.crecimiento = por_sector("crecimiento") + rng.normal(0, 0.005, m)
        self.margen_objetivo = por_sector("margen") * rng.lognormal(0, 0.2, m)
        self.roe_objetivo = por_sector("roe") * rng.lognormal(0, 0.15, m)
        self.beta = por_sector("beta") + rng.normal(0, 0.1, m)
        self.deuda_objetivo = por_sector("deuda") * rng.lognormal(0, 0.3, m)
        self.ciclicidad = por_sector("ciclicidad") * rng.uniform(0.7, 1.3, m)
        self.vol_ventas = por_sector("vol") * rng.uniform(0.8, 1.2, m)
        self.retencion = np.clip(self.crecimiento / self.roe_objetivo, 0.05, 0.9)
        self.acciones = rng.lognormal(np.log(100.0), 0.8, m)  # millones de acciones

        # Estado real (no observable directamente)
        bpa0 = rng.lognormal(np.log(3.0), 0.4, m)
        self.ciclo = 0.0
        self.tasa = cfg.tasa_libre_riesgo
        self.margen = self.margen_objetivo.copy()
        self.ventas = bpa0 / self.margen
        self.valor_contable = bpa0 / self.roe_objetivo
        self.deuda = self.deuda_objetivo.copy()
        self.desfase = rng.integers(0, 21, m)  # día del trimestre en que publica
        # Beneficio ya acumulado en el trimestre en curso, como si la empresa existiera antes
        self._acumulado = bpa0 * (DIAS_TRIMESTRE - 1 - self.desfase) / DIAS_HABILES

        # Información publicada (lo que ven los agentes)
        self.historial_bpa = np.tile(bpa0 / 4.0, (8, 1))  # últimos 8 trimestres
        self.vc_publicado = self.valor_contable.copy()
        self.deuda_publicada = self.deuda.copy()
        self.publica_hoy = np.zeros(m, dtype=bool)
        self.sorpresa = np.zeros(m)  # log(BPA publicado / esperado) del último informe
        self.dividendo = np.zeros(m)  # dividendo por acción pagado hoy

    # ------------------------------------------------------------------ dinámica
    def avanzar(self, t: int) -> None:
        cfg, rng = self.cfg, self.rng
        dt = 1.0 / DIAS_HABILES
        raiz = np.sqrt(dt)
        m = self.n_empresas

        z_macro = rng.normal()
        z_sector = rng.normal(size=len(self.sectores))[self.sector]
        z_propio = rng.normal(size=m)

        self.ciclo += -cfg.reversion_ciclo * self.ciclo * dt + cfg.vol_ciclo * raiz * z_macro
        self.tasa += (
            cfg.reversion_tasa * (cfg.tasa_libre_riesgo - self.tasa) * dt
            + cfg.vol_tasa * raiz * rng.normal()
        )
        self.tasa = max(self.tasa, 0.0)

        peso_propio = np.sqrt(max(1.0 - cfg.peso_macro**2 - cfg.peso_sector**2, 0.0))
        shock = (
            cfg.peso_macro * self.ciclicidad * z_macro
            + cfg.peso_sector * z_sector
            + peso_propio * z_propio
        )
        saltos = (rng.random(m) < cfg.prob_salto_anual * dt) * rng.normal(
            cfg.salto_medio, cfg.salto_vol, m
        )
        self.ventas *= np.exp(
            (self.crecimiento - 0.5 * self.vol_ventas**2) * dt
            + self.vol_ventas * raiz * shock
            + saltos
        )

        # Los márgenes se expanden en los auges y se comprimen en las recesiones
        objetivo = self.margen_objetivo * np.clip(1.0 + 0.3 * self.ciclicidad * self.ciclo, 0.3, 1.7)
        self.margen += (
            cfg.reversion_margen * (objetivo - self.margen) * dt
            + cfg.vol_margen * self.margen_objetivo * raiz * rng.normal(size=m)
        )
        self.margen = np.clip(self.margen, 0.2 * self.margen_objetivo, 2.5 * self.margen_objetivo)

        self.deuda += 0.5 * (self.deuda_objetivo - self.deuda) * dt + 0.1 * self.deuda_objetivo * raiz * rng.normal(size=m)
        self.deuda = np.maximum(self.deuda, 0.0)

        bpa = self.bpa_actual
        # Gestión del capital: recompras o ampliaciones devuelven el ROE hacia su objetivo
        objetivo_vc = self.ventas * self.margen_objetivo / self.roe_objetivo
        self.valor_contable += self.retencion * bpa * dt + 0.3 * (objetivo_vc - self.valor_contable) * dt
        self._acumulado += bpa * dt
        self._publicar(t)

    def _publicar(self, t: int) -> None:
        self.publica_hoy = (t % DIAS_TRIMESTRE) == self.desfase
        self.dividendo[:] = 0.0
        idx = np.flatnonzero(self.publica_hoy)
        if idx.size == 0:
            return
        bpa_q = self._acumulado[idx] * np.exp(self.rng.normal(0, self.cfg.ruido_contable, idx.size))
        esperado = self.historial_bpa[-1, idx] * np.exp(self.crecimiento[idx] / 4.0)
        self.sorpresa[idx] = np.log(bpa_q / esperado)
        self.historial_bpa[:-1, idx] = self.historial_bpa[1:, idx]
        self.historial_bpa[-1, idx] = bpa_q
        self._acumulado[idx] = 0.0
        self.vc_publicado[idx] = self.valor_contable[idx]
        self.deuda_publicada[idx] = self.deuda[idx]
        self.dividendo[idx] = (1.0 - self.retencion[idx]) * bpa_q

    # --------------------------------------------------------- datos y ratios
    @property
    def bpa_actual(self) -> np.ndarray:
        """Beneficio por acción anualizado real del día (no publicado)."""
        return self.ventas * self.margen

    @property
    def bpa_ttm(self) -> np.ndarray:
        return self.historial_bpa[-4:].sum(axis=0)

    @property
    def crecimiento_reciente(self) -> np.ndarray:
        """Crecimiento interanual del BPA publicado (lo que se extrapola)."""
        return np.clip(np.log(self.bpa_ttm / self.historial_bpa[:4].sum(axis=0)), -0.3, 0.3)

    @property
    def roe_publicado(self) -> np.ndarray:
        return np.clip(self.bpa_ttm / self.vc_publicado, 0.02, 0.6)

    def coste_capital(self, deuda: np.ndarray) -> np.ndarray:
        return self.tasa + self.beta * self.cfg.prima_mercado + self.cfg.prima_deuda * np.minimum(deuda, 2.0)

    def valor_verdadero(self) -> np.ndarray:
        """Valor intrínseco con la información real; sirve de referencia, nadie lo ve.

        Usa el beneficio normalizado (margen de largo plazo) y suma el exceso de
        beneficio actual durante el tiempo que tarda el margen en volver a su media.
        """
        normalizado = self.ventas * self.margen_objetivo
        exceso = (self.bpa_actual - normalizado) / self.cfg.reversion_margen
        return exceso + valor_gordon(
            normalizado, self.crecimiento, self.roe_objetivo, self.coste_capital(self.deuda)
        )

    def valor_publicado(self, crecimiento_percibido: np.ndarray, empresa: np.ndarray | slice = slice(None)) -> np.ndarray:
        """Valor DCF en dos etapas con datos publicados y el crecimiento que percibe cada agente.

        ``empresa`` indica a qué empresa corresponde cada elemento de
        ``crecimiento_percibido`` (por defecto, una por empresa).
        """
        # Como un analista, se suaviza el ROE publicado hacia la media histórica de la empresa
        roe = 0.3 * self.roe_publicado + 0.7 * self.roe_objetivo
        return valor_dos_etapas(
            self.bpa_ttm[empresa],
            crecimiento_percibido,
            self.crecimiento[empresa],
            roe[empresa],
            self.coste_capital(self.deuda_publicada)[empresa],
        )

    def per_mediano_sector(self, precios: np.ndarray) -> np.ndarray:
        """log(PER) mediano del sector de cada empresa: la base de la valoración relativa."""
        log_per = np.log(precios / self.bpa_ttm)
        medianas = np.array([np.median(log_per[self.sector == s]) for s in range(len(self.sectores))])
        return medianas[self.sector]

    def calidad(self) -> np.ndarray:
        """Puntuación compuesta: ROE alto, poca deuda y crecimiento reciente (z-scores)."""

        def z(x):
            return (x - x.mean()) / (x.std() + 1e-12)

        bruto = z(self.roe_publicado) - z(self.deuda_publicada) + 0.5 * z(self.crecimiento_reciente)
        return z(bruto)

    def ratios(self, precios: np.ndarray) -> dict[str, np.ndarray]:
        ttm = self.bpa_ttm
        return {
            "per": precios / ttm,
            "precio_valor_contable": precios / self.vc_publicado,
            "roe": self.roe_publicado,
            "deuda_patrimonio": self.deuda_publicada,
            "rentabilidad_dividendo": (1.0 - self.retencion) * ttm / precios,
            "crecimiento_bpa": self.crecimiento_reciente,
        }
