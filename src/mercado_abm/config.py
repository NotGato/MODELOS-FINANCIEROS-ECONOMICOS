"""Parámetros del modelo agrupados por capa.

Todos los valores se pueden sobrescribir desde un fichero TOML cuyas tablas
coinciden con los campos de :class:`Config` (``[economia]``, ``[agentes]``...).
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

DIAS_HABILES = 252
DIAS_TRIMESTRE = 63


@dataclass
class ConfigSimulacion:
    dias: int = 6300  # ~25 años de sesiones
    calentamiento: int = 252  # días excluidos de las métricas
    historia_previa: int = 300  # precios sintéticos para inicializar indicadores
    semilla: int = 1


@dataclass
class ConfigEconomia:
    n_sectores: int = 5
    empresas_por_sector: int = 4
    tasa_libre_riesgo: float = 0.03
    vol_tasa: float = 0.003  # volatilidad anual del tipo libre de riesgo
    reversion_tasa: float = 0.2
    prima_mercado: float = 0.045
    prima_deuda: float = 0.01  # prima de riesgo por unidad de deuda/patrimonio
    vol_ciclo: float = 0.5  # volatilidad anual del ciclo macro
    reversion_ciclo: float = 0.5
    peso_macro: float = 0.5  # reparto de los shocks de ventas
    peso_sector: float = 0.4
    vol_margen: float = 0.2  # volatilidad anual relativa del margen
    reversion_margen: float = 1.0
    prob_salto_anual: float = 0.3  # shocks puntuales de negocio por empresa
    salto_medio: float = -0.05
    salto_vol: float = 0.15
    ruido_contable: float = 0.03  # ruido de los resultados publicados


@dataclass
class ConfigAgentes:
    n_agentes: int = 2000
    fracciones_iniciales: tuple[float, float, float] = (0.5, 0.3, 0.2)  # F, C, R
    pareto_riqueza: float = 1.5  # índice de cola de la riqueza inicial
    seguidas_min: int = 3  # empresas que sigue cada agente
    seguidas_max: int = 10
    frac_apalancados: float = 0.15
    apalancamiento_max: float = 4.0
    frac_sin_cortos: float = 0.4
    frac_multiplos: float = 0.2  # fundamentalistas que valoran por PER sectorial
    exposicion_base: float = 0.6  # cartera estratégica larga, en fracción del límite
    exposicion_base_apalancados: float = 0.7
    horizonte: tuple[float, float, float] = (120.0, 30.0, 10.0)  # días por tipo
    tamano_orden: float = 0.05  # fracción del capital por unidad de demanda


@dataclass
class ConfigEstrategias:
    phi: float = 0.5  # fuerza de la reversión al valor fundamental
    peso_calidad: float = 0.03  # inclinación por ROE alto y poca deuda
    chi: float = 0.4  # fuerza de las reglas técnicas
    ruido: tuple[float, float, float] = (0.75, 2.1, 1.0)  # ruido común por tipo
    ruido_mercado: float = 0.35  # parte del ruido común a todo el mercado
    ruido_sector: float = 0.2  # parte común al sector
    ruido_individual: float = 0.3
    dispersion_opiniones: float = 0.1  # ruido propio de cada valoración
    sesgo_valoracion: float = 0.05
    reaccion_noticias: float = 0.5  # sobrerreacción del agente de ruido a sorpresas


@dataclass
class ConfigMercado:
    impacto: float = 0.04  # retorno por unidad de demanda neta media
    umbral_impacto: float = 0.06  # por encima, el impacto pasa de lineal a raíz cuadrada
    sensibilidad_liquidez: float = 0.1  # la profundidad cae con la volatilidad
    vol_referencia: float = 0.013
    estres_max: float = 2.0  # tope del estrés de liquidez: el impacto se multiplica como mucho por 1 + 0.1·2
    limite_diario: float = 0.2  # interruptor de volatilidad: retorno log máximo en un día (0 = sin límite)
    coste_transaccion: float = 0.0005
    diferencial_prestamo: float = 0.02


@dataclass
class ConfigComportamiento:
    # --- Interruptores (fase 1: los cinco primeros; fase 2: todos) ---
    cambio_estrategia: bool = True
    contagio: bool = True
    sentimiento: bool = True
    atencion_limitada: bool = True
    extrapolacion: bool = True
    conservadurismo: bool = True
    aversion_perdidas: bool = True
    efecto_disposicion: bool = True
    anclaje: bool = True
    exceso_confianza: bool = True
    apalancamiento: bool = True

    # Cambio de estrategia (Brock-Hommes) y contagio (Kirman)
    intensidad_eleccion: float = 8.0
    predisposicion: tuple[float, float, float] = (0.0, 0.0, 0.0)  # atractivo fijo de F, C y R
    escala_rendimiento: float = 0.2  # % diario a partir del cual el rendimiento se satura
    memoria: float = 0.9
    contagio_intensidad: float = 2.0
    miedo_burbuja: float = 5.0
    tasa_revision: float = 0.02
    revision_por_perdidas: float = 1.0
    exploracion: float = 0.03

    # Sentimiento
    persistencia_sentimiento: float = 0.97
    reaccion_sentimiento: float = 3.0
    ruido_sentimiento: float = 0.02
    efecto_sentimiento_valor: float = 0.1
    efecto_sentimiento_ruido: float = 0.15

    # Atención
    atencion_base: float = 0.35  # probabilidad de operar con volatilidad habitual
    atencion_sensibilidad: float = 0.8  # exponente de (vol reciente / vol habitual)
    atencion_max: float = 0.9
    atencion_noticias: float = 0.3
    atencion_fija: float = 0.35  # se usa si atencion_limitada = False

    # Extrapolación y conservadurismo
    extrapolacion_max: float = 0.5
    velocidad_ajuste: tuple[float, float] = (0.02, 0.10)

    # Teoría prospectiva y efecto disposición
    aversion: float = 2.25  # peso de las pérdidas al evaluar estrategias (Kahneman-Tversky)
    asimetria_atencion: float = 1.5  # cuánto más llama la atención una caída que una subida
    disposicion_max: float = 0.5

    # Anclaje: caza de gangas frente al máximo de 52 semanas
    anclaje_maximo: float = 0.3
    anclaje_distancia_neutra: float = 0.5  # distancia típica al máximo anual, en volatilidades anuales

    # Exceso de confianza (innata, lognormal) con autoatribución
    confianza_media: float = 0.1
    confianza_dispersion: float = 0.2
    confianza_max: float = 2.5
    confianza_aprendizaje_ganancias: float = 0.004
    confianza_aprendizaje_perdidas: float = 0.002
    confianza_reversion: float = 1.0 / 250

    # Apalancamiento, llamadas de margen y quiebras
    colchon_margen: float = 0.15
    objetivo_desapalancamiento: float = 0.8
    liquidacion_diaria_max: float = 0.5  # fracción máxima de cartera que el bróker vende al día
    umbral_quiebra: float = 0.1
    reapalancamiento_anual: float = 0.25  # parte del hueco de apalancados que se recupera al año


@dataclass
class Config:
    simulacion: ConfigSimulacion = field(default_factory=ConfigSimulacion)
    economia: ConfigEconomia = field(default_factory=ConfigEconomia)
    agentes: ConfigAgentes = field(default_factory=ConfigAgentes)
    estrategias: ConfigEstrategias = field(default_factory=ConfigEstrategias)
    mercado: ConfigMercado = field(default_factory=ConfigMercado)
    comportamiento: ConfigComportamiento = field(default_factory=ConfigComportamiento)

    def como_dict(self) -> dict:
        return asdict(self)


SESGOS_FASE_2 = (
    "conservadurismo",
    "aversion_perdidas",
    "efecto_disposicion",
    "anclaje",
    "exceso_confianza",
    "apalancamiento",
)


def _aplicar(destino, valores: dict, ruta: str = "") -> None:
    nombres = {f.name: f for f in fields(destino)}
    for clave, valor in valores.items():
        if clave not in nombres:
            raise KeyError(f"Parámetro desconocido: {ruta}{clave}")
        actual = getattr(destino, clave)
        if is_dataclass(actual):
            if not isinstance(valor, dict):
                raise TypeError(f"{ruta}{clave} debe ser una tabla")
            _aplicar(actual, valor, f"{ruta}{clave}.")
        else:
            if isinstance(actual, tuple):
                valor = tuple(valor)
            elif isinstance(actual, float) and isinstance(valor, int):
                valor = float(valor)
            setattr(destino, clave, valor)


def actualizar(config: Config, valores: dict) -> Config:
    """Sobrescribe los campos de ``config`` con un dict anidado (mismo formato que el TOML)."""
    _aplicar(config, valores)
    return config


def cargar(ruta: str | Path | None = None) -> Config:
    config = Config()
    if ruta is not None:
        with open(ruta, "rb") as f:
            actualizar(config, tomllib.load(f))
    return config


def fase1(config: Config | None = None) -> Config:
    """Desactiva los sesgos de la fase 2 y deja la base de la fase 1."""
    config = config or Config()
    for sesgo in SESGOS_FASE_2:
        setattr(config.comportamiento, sesgo, False)
    return config
