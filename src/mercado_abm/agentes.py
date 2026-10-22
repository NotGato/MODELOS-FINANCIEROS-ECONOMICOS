"""Capa 2: la población de agentes.

Cada agente sigue solo unas pocas empresas, así que el estado se guarda por
*vínculo* agente-empresa (arrays de longitud L = número de vínculos) en vez de
en matrices N×M casi vacías. ``agente[l]`` y ``empresa[l]`` dicen a quién
pertenece cada vínculo; las sumas por agente o por empresa se hacen con
``np.bincount``.

Cada agente tiene rasgos fijos (riqueza, agresividad, sesgos, regla técnica,
empresas que sigue, apalancamiento máximo...) y un estado que cambia (tipo de
estrategia actual, cartera, caja, confianza). La cartera tiene dos partes:

* ``n_base``: cartera estratégica larga ("comprar y mantener"), que se reequilibra
  despacio contra inversores pasivos externos, sin impacto en precio.
* ``n_esp``: posiciones especulativas abiertas con órdenes que sí mueven el
  precio. Se van cerrando con un horizonte típico de cada tipo de agente.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .tecnico import REGLA_SUSTITUTA, REGLAS, REGLAS_ANCLAJE

FUNDAMENTALISTA, CHARTISTA, RUIDO = 0, 1, 2
TIPOS = ("fundamentalista", "chartista", "ruido")

RIQUEZA_MEDIA = 100.0
HORIZONTE_BASE = 60.0  # días para reequilibrar la cartera estratégica


class Poblacion:
    def __init__(self, config: Config, n_empresas: int, capitalizacion: np.ndarray, rng: np.random.Generator):
        ca, ce, cc = config.agentes, config.estrategias, config.comportamiento
        self.cfg = config
        self.rng = rng
        n, m = ca.n_agentes, n_empresas
        self.n, self.m = n, m

        self.tipo = rng.choice(3, n, p=np.asarray(ca.fracciones_iniciales) / sum(ca.fracciones_iniciales))

        # --- Rasgos fijos ---
        riqueza = rng.pareto(ca.pareto_riqueza, n) + 1.0
        self.riqueza_inicial = RIQUEZA_MEDIA * riqueza / riqueza.mean()
        self.agresividad = rng.lognormal(0.0, 0.5, n)
        self.sesgo = rng.normal(0.0, ce.sesgo_valoracion, n)
        self.sensibilidad = rng.uniform(0.5, 1.5, n)
        self.extrapolacion = rng.uniform(0.0, cc.extrapolacion_max, n) if cc.extrapolacion else np.zeros(n)
        if cc.conservadurismo:
            self.velocidad_ajuste = rng.uniform(*cc.velocidad_ajuste, n)
        else:
            self.velocidad_ajuste = np.ones(n)
        self.usa_multiplos = rng.random(n) < ca.frac_multiplos
        self.disposicion = rng.uniform(0.0, cc.disposicion_max, n) if cc.efecto_disposicion else np.zeros(n)
        self.puede_cortos = rng.random(n) >= ca.frac_sin_cortos

        self.limite = np.ones(n)
        self.exposicion_objetivo = np.full(n, ca.exposicion_base)
        if cc.apalancamiento:
            self._apalancar(np.flatnonzero(rng.random(n) < ca.frac_apalancados))

        reglas = rng.integers(len(REGLAS), size=n)
        if not cc.anclaje:
            for nombre in REGLAS_ANCLAJE:
                reglas[reglas == REGLAS.index(nombre)] = REGLAS.index(REGLA_SUSTITUTA)
        self.regla = reglas

        # Confianza de partida: unos agentes se creen mejores que otros
        if cc.exceso_confianza:
            self.confianza_innata = rng.lognormal(cc.confianza_media, cc.confianza_dispersion, n)
        else:
            self.confianza_innata = np.ones(n)

        # --- Vínculos: empresas seguidas (las grandes atraen más atención) ---
        k = rng.integers(ca.seguidas_min, ca.seguidas_max + 1, n).clip(max=m)
        peso = np.sqrt(capitalizacion) / np.sqrt(capitalizacion).sum()
        self.agente = np.repeat(np.arange(n), k)
        self.empresa = np.concatenate([rng.choice(m, size=ki, replace=False, p=peso) for ki in k])
        self.n_seguidas = k
        self.seguidores = np.bincount(self.empresa, minlength=m)
        n_vinculos = self.agente.size

        # --- Estado ---
        self.confianza = self.confianza_innata.copy()
        self.rendimiento_medio = np.zeros(n)  # media móvil del rendimiento especulativo (%)
        self.opinion = rng.normal(0.0, ce.dispersion_opiniones, n_vinculos)
        self.log_valor = np.zeros(n_vinculos)  # se inicializa en la simulación
        self.n_base = np.zeros(n_vinculos)
        self.n_esp = np.zeros(n_vinculos)
        self.coste = np.zeros(n_vinculos)  # precio medio de la posición especulativa
        self.venta_forzada = np.zeros(n)  # fracción de la cartera a liquidar mañana
        self.en_quiebra = np.zeros(n, dtype=bool)
        self.caja = self.riqueza_inicial.copy()

    def _apalancar(self, idx: np.ndarray) -> None:
        ca = self.cfg.agentes
        self.limite[idx] = self.rng.uniform(1.5, ca.apalancamiento_max, idx.size)
        self.exposicion_objetivo[idx] = ca.exposicion_base_apalancados * self.limite[idx]

    def reapalancar(self, dt: float) -> None:
        """Ciclo de apalancamiento: tras un crac hay pocos apalancados y se recuperan despacio.

        Cada día, una pequeña parte de los inversores sin apalancar empieza a
        apalancarse, hasta volver a la fracción objetivo (la complacencia de Minsky).
        """
        ca, cc = self.cfg.agentes, self.cfg.comportamiento
        candidatos = np.flatnonzero((self.limite == 1.0) & ~self.en_quiebra)
        hueco = ca.frac_apalancados - (self.limite > 1.0).mean()
        if hueco <= 0 or candidatos.size == 0:
            return
        prob = cc.reapalancamiento_anual * dt * hueco * self.n / candidatos.size
        self._apalancar(candidatos[self.rng.random(candidatos.size) < prob])

    # ---------------------------------------------------------------- agregados
    def por_agente(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.agente, weights=x, minlength=self.n)

    def por_empresa(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.empresa, weights=x, minlength=self.m)

    # ----------------------------------------------------------------- cartera
    def posiciones(self, precios: np.ndarray) -> np.ndarray:
        """Valor de mercado de cada vínculo (cartera estratégica + especulativa)."""
        return (self.n_base + self.n_esp) * precios[self.empresa]

    def riqueza(self, precios: np.ndarray) -> np.ndarray:
        return self.caja + self.por_agente(self.posiciones(precios))

    def exposicion(self, precios: np.ndarray) -> np.ndarray:
        return self.por_agente(np.abs(self.posiciones(precios)))

    def capital(self, riqueza: np.ndarray) -> np.ndarray:
        """Capital que cada agente dedica a cada empresa que sigue."""
        return (np.maximum(riqueza, 0.0) / self.n_seguidas)[self.agente]

    def objetivo_base(self, precios: np.ndarray, capitalizacion: np.ndarray, riqueza: np.ndarray) -> np.ndarray:
        """Acciones que querría tener en la cartera estratégica (pesos por capitalización)."""
        cap = capitalizacion[self.empresa]
        peso = cap / self.por_agente(cap)[self.agente]
        exposicion = self.exposicion_objetivo * np.maximum(riqueza, 0.0)
        return exposicion[self.agente] * peso / precios[self.empresa]

    def iniciar_cartera(self, precios: np.ndarray, capitalizacion: np.ndarray) -> None:
        self.n_base = self.objetivo_base(precios, capitalizacion, self.riqueza_inicial)
        self.caja = self.riqueza_inicial - self.por_agente(self.n_base * precios[self.empresa])

    # ------------------------------------------------------------ operaciones
    def ejecutar_especulativas(self, acciones: np.ndarray, precios: np.ndarray, coste_tx: float) -> None:
        """Ejecuta órdenes especulativas (en acciones) y actualiza el precio medio de compra."""
        precio = precios[self.empresa]
        previa = self.n_esp
        nueva = previa + acciones
        amplia = ((previa * acciones > 0) | (previa == 0)) & (acciones != 0)
        cruza = previa * nueva < 0
        medio = (np.abs(previa) * self.coste + np.abs(acciones) * precio) / np.maximum(np.abs(nueva), 1e-300)
        self.coste = np.where(amplia, medio, np.where(cruza, precio, self.coste))
        self.coste[nueva == 0] = 0.0
        self.n_esp = nueva
        importe = acciones * precio
        self.caja -= self.por_agente(importe + coste_tx * np.abs(importe))

    def liquidar_fraccion(self, fraccion: np.ndarray, precios: np.ndarray, coste_tx: float) -> None:
        """Ventas forzadas (llamadas de margen o quiebra): reduce toda la cartera en ``fraccion``."""
        f = fraccion[self.agente]
        importe = f * (self.n_base + self.n_esp) * precios[self.empresa]
        self.n_base *= 1.0 - f
        self.n_esp *= 1.0 - f
        self.caja += self.por_agente(importe - coste_tx * np.abs(importe))

    def reequilibrio_pasivo(self, objetivo_base: np.ndarray, cierre: np.ndarray, precios: np.ndarray) -> None:
        """Operaciones contra inversores pasivos externos (sin impacto en precio)."""
        nueva_base = self.n_base + (objetivo_base - self.n_base) / HORIZONTE_BASE
        nueva_esp = self.n_esp * (1.0 - cierre)
        # Quien no puede vender en corto nunca queda con una posición total negativa
        nueva_esp = np.where(self.puede_cortos[self.agente], nueva_esp, np.maximum(nueva_esp, -nueva_base))
        delta = (nueva_base - self.n_base) + (nueva_esp - self.n_esp)
        self.n_base, self.n_esp = nueva_base, nueva_esp
        self.caja -= self.por_agente(delta * precios[self.empresa])

    def reemplazar(self, idx: np.ndarray, fracciones: np.ndarray, escala_riqueza: float) -> None:
        """Un agente en quiebra sale del mercado y entra otro, sin apalancar, con capital nuevo."""
        if idx.size == 0:
            return
        vinculos = np.isin(self.agente, idx)
        self.limite[idx] = 1.0
        self.exposicion_objetivo[idx] = self.cfg.agentes.exposicion_base
        self.tipo[idx] = self.rng.choice(3, idx.size, p=fracciones)
        self.n_base[vinculos] = 0.0
        self.n_esp[vinculos] = 0.0
        self.coste[vinculos] = 0.0
        self.caja[idx] = self.riqueza_inicial[idx] * escala_riqueza
        self.confianza[idx] = self.confianza_innata[idx]
        self.rendimiento_medio[idx] = 0.0
        self.venta_forzada[idx] = 0.0
        self.en_quiebra[idx] = False
