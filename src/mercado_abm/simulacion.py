"""Bucle principal: une las cuatro capas y registra la historia del mercado."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import comportamiento as comp
from .agentes import TIPOS, Poblacion
from .config import DIAS_HABILES, Config
from .economia import Economia
from .estrategias import actualizar_valoraciones, demandas, ruido_comun, valoracion_objetivo
from .mercado import CreadorDeMercado
from .tecnico import Indicadores

_SERIES_EMPRESA = ("log_valor", "log_consenso", "bpa_ttm", "valor_contable", "volumen")
_SERIES_MERCADO = (
    "tasa",
    "ciclo",
    "sentimiento",
    "llamadas_margen",
    "quiebras",
    "confianza",
    "apalancamiento",
)


@dataclass
class Resultados:
    nombres: list[str]
    sectores: list[str]
    sector: np.ndarray
    acciones: np.ndarray
    calentamiento: int
    config: dict
    log_precio: np.ndarray  # (T+1)×M
    log_valor: np.ndarray  # T×M, valor verdadero
    log_consenso: np.ndarray  # T×M, valoración media de los agentes
    bpa_ttm: np.ndarray
    valor_contable: np.ndarray
    volumen: np.ndarray  # órdenes / capital de los seguidores
    fracciones: np.ndarray  # T×3
    riqueza_tipo: np.ndarray  # T×3, cuota de la riqueza total
    tasa: np.ndarray
    ciclo: np.ndarray
    sentimiento: np.ndarray
    llamadas_margen: np.ndarray
    quiebras: np.ndarray
    confianza: np.ndarray
    apalancamiento: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def dias(self) -> int:
        return self.log_valor.shape[0]

    @property
    def retornos(self) -> np.ndarray:
        return np.diff(self.log_precio, axis=0)

    def pesos_capitalizacion(self) -> np.ndarray:
        cap = self.acciones[None, :] * np.exp(self.log_precio[:-1])
        return cap / cap.sum(axis=1, keepdims=True)

    def retornos_indice(self) -> np.ndarray:
        """Retorno logarítmico diario de un índice ponderado por capitalización."""
        simple = np.expm1(self.retornos)
        return np.log1p((self.pesos_capitalizacion() * simple).sum(axis=1))

    def log_indice(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.retornos_indice())])

    def log_valor_indice(self) -> np.ndarray:
        """Valor fundamental agregado, normalizado para empezar donde el índice."""
        valor = (self.acciones[None, :] * np.exp(self.log_valor)).sum(axis=1)
        precio0 = (self.acciones * np.exp(self.log_precio[0])).sum()
        return np.log(valor / precio0)

    def guardar(self, carpeta: str | Path) -> Path:
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        arrays = {
            k: getattr(self, k)
            for k in ("sector", "acciones", "log_precio", "fracciones", "riqueza_tipo", *_SERIES_EMPRESA, *_SERIES_MERCADO)
        }
        np.savez_compressed(carpeta / "resultados.npz", **arrays)
        meta = {
            "nombres": self.nombres,
            "sectores": self.sectores,
            "calentamiento": self.calentamiento,
            "config": self.config,
        }
        (carpeta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return carpeta

    @classmethod
    def cargar(cls, carpeta: str | Path) -> Resultados:
        carpeta = Path(carpeta)
        meta = json.loads((carpeta / "meta.json").read_text())
        with np.load(carpeta / "resultados.npz") as datos:
            arrays = {k: datos[k] for k in datos.files}
        return cls(**meta, **arrays)


class Simulacion:
    def __init__(self, config: Config | None = None):
        self.config = config = config or Config()
        self.rng = rng = np.random.default_rng(config.simulacion.semilla)
        self.eco = Economia(config.economia, rng)
        m = self.eco.n_empresas

        precio0 = self.eco.valor_verdadero()
        cap0 = self.eco.acciones * precio0
        self.pob = Poblacion(config, m, cap0, rng)
        self.mm = CreadorDeMercado(config.mercado, config.agentes)
        self.sentimiento = comp.Sentimiento(config.comportamiento, len(self.eco.sectores), rng)
        self.cambio = comp.CambioEstrategia(config.comportamiento, rng)
        self.noticias = np.zeros(m)
        self.horizonte = np.asarray(config.agentes.horizonte, dtype=float)

        # Historia sintética previa para que los indicadores técnicos arranquen definidos
        h = max(config.simulacion.historia_previa, 260)
        camino = np.vstack([np.zeros((1, m)), np.cumsum(rng.normal(0, 0.012, (h - 1, m)), axis=0)])
        historia = np.log(precio0)[None, :] + camino - camino[-1]
        self.indicadores = Indicadores(historia, h + config.simulacion.dias + 1)

        self.pob.iniciar_cartera(precio0, cap0)
        self.pob.log_valor = valoracion_objetivo(self.pob, self.eco, precio0)

    # ------------------------------------------------------------------ bucle
    def ejecutar(self, progreso: Callable[[int, int], None] | None = None) -> Resultados:
        cfg = self.config
        dias, m = cfg.simulacion.dias, self.eco.n_empresas
        reg = {k: np.zeros((dias, m)) for k in _SERIES_EMPRESA}
        reg.update({k: np.zeros(dias) for k in _SERIES_MERCADO})
        reg["fracciones"] = np.zeros((dias, 3))
        reg["riqueza_tipo"] = np.zeros((dias, 3))
        log_precio = np.zeros((dias + 1, m))
        log_precio[0] = self.indicadores.actual

        for t in range(dias):
            self._paso(t, reg)
            log_precio[t + 1] = self.indicadores.actual
            if progreso is not None and (t + 1) % DIAS_HABILES == 0:
                progreso(t + 1, dias)

        return Resultados(
            nombres=self.eco.nombres,
            sectores=[s.nombre for s in self.eco.sectores],
            sector=self.eco.sector,
            acciones=self.eco.acciones,
            calentamiento=cfg.simulacion.calentamiento,
            config=cfg.como_dict(),
            log_precio=log_precio,
            **reg,
        )

    def _paso(self, t: int, reg: dict) -> None:
        cfg, rng = self.config, self.rng
        cc, ca = cfg.comportamiento, cfg.agentes
        eco, pob, ind = self.eco, self.pob, self.indicadores
        a, e = pob.agente, pob.empresa

        # 1. Fundamentales: avanza la economía; algunas empresas publican resultados
        eco.avanzar(t)
        log_p = ind.actual.copy()
        precio = np.exp(log_p)
        publica = eco.publica_hoy
        self.noticias *= 0.8
        if publica.any():
            self.noticias[publica] += np.clip(eco.sorpresa[publica] / 0.05, -5.0, 5.0)
            nuevas = publica[e]
            pob.opinion[nuevas] = rng.normal(0, cfg.estrategias.dispersion_opiniones, nuevas.sum())

        # 2. Valoraciones (ancladas a la anterior si hay conservadurismo)
        actualizar_valoraciones(pob, valoracion_objetivo(pob, eco, precio))
        consenso = pob.por_empresa(pob.log_valor) / np.maximum(pob.seguidores, 1)

        # 3. Atención y demandas
        posiciones = pob.posiciones(precio)
        riqueza_previa = pob.caja + pob.por_agente(posiciones)
        vol20 = ind.vol_ventana(20)
        operan = ~pob.en_quiebra & (pob.venta_forzada == 0)
        vol_percibida = ind.vol_percibida(cc.asimetria_atencion if cc.aversion_perdidas else 1.0)
        p_activo = comp.probabilidad_atencion(cc, vol_percibida, ind.vol_lenta, self.noticias, pob.confianza, a, e)
        activo = (rng.random(a.size) < p_activo) & operan[a]

        d = demandas(
            cfg,
            pob,
            log_p,
            self.sentimiento.por_empresa(eco.sector),
            eco.calidad(),
            ind.senales(),
            self.noticias,
            ind.maximo(DIAS_HABILES),
            ind.vol_lenta,
            ruido_comun(cfg, eco.sector, rng),
            rng,
        )
        capital = pob.capital(riqueza_previa)
        orden = d * ca.tamano_orden * (pob.agresividad * pob.confianza)[a] * capital * activo

        # Efecto disposición: realizar ganancias pronto, resistirse a cerrar pérdidas
        realizar, en_perdidas = comp.ordenes_disposicion(pob.disposicion, pob.n_esp, pob.coste, log_p, a, e)
        cierra_perdedora = (orden * pob.n_esp < 0) & en_perdidas
        orden = np.where(cierra_perdedora, orden * (1.0 - pob.disposicion[a]), orden)
        orden += realizar * activo

        orden = self._restricciones(orden, posiciones, riqueza_previa)
        total = orden - pob.venta_forzada[a] * posiciones

        # 4. Precio: el creador de mercado absorbe la demanda neta
        capital_empresa = pob.por_empresa(capital)
        liquidez = self.mm.liquidez(capital_empresa, vol20)
        r = self.mm.retornos(pob.por_empresa(total), liquidez)
        log_p_nuevo = log_p + r
        precio_nuevo = np.exp(log_p_nuevo)

        # 5. Contabilidad
        pnl_esp = pob.por_agente(pob.n_esp * (precio_nuevo - precio)[e])
        tx = cfg.mercado.coste_transaccion
        pob.liquidar_fraccion(pob.venta_forzada, precio_nuevo, tx)
        pob.ejecutar_especulativas(orden / precio[e], precio_nuevo, tx)
        cierre = (1.0 / self.horizonte[pob.tipo])[a] * np.where(en_perdidas, 1.0 - pob.disposicion[a], 1.0)
        cap_nueva = eco.acciones * precio_nuevo
        objetivo = pob.objetivo_base(precio_nuevo, cap_nueva, pob.riqueza(precio_nuevo))
        pob.reequilibrio_pasivo(objetivo, cierre, precio_nuevo)
        tipo_interes = np.where(pob.caja > 0, eco.tasa, eco.tasa + cfg.mercado.diferencial_prestamo)
        pob.caja += pob.caja * tipo_interes / DIAS_HABILES
        if publica.any():
            pob.caja += pob.por_agente((pob.n_base + pob.n_esp) * eco.dividendo[e])

        # 6. Aprendizaje: aptitud de las estrategias, confianza y cambio de estrategia
        rend = 100.0 * pnl_esp / np.maximum(riqueza_previa, 1e-9)
        pnl_tipo = np.bincount(pob.tipo, weights=pnl_esp, minlength=3)
        riqueza_tipo = np.bincount(pob.tipo, weights=np.maximum(riqueza_previa, 0.0), minlength=3)
        self.cambio.actualizar_aptitud(100.0 * pnl_tipo / np.maximum(riqueza_tipo, 1e-9))
        pob.rendimiento_medio = 0.95 * pob.rendimiento_medio + 0.05 * rend
        pob.confianza = comp.actualizar_confianza(cc, pob.confianza, pob.confianza_innata, pob.rendimiento_medio)

        fracciones = np.bincount(pob.tipo, minlength=3) / pob.n
        desalineacion = float(np.mean((log_p_nuevo - consenso) ** 2))
        p_tipos = self.cambio.probabilidades(fracciones, desalineacion)
        pob.tipo = self.cambio.revisar(pob.tipo, pob.rendimiento_medio, ~operan, p_tipos)

        # 7. Quiebras de ayer ya liquidadas -> entran agentes nuevos; nuevas llamadas de margen
        riqueza = pob.riqueza(precio_nuevo)
        escala_riqueza = float(np.median(riqueza / pob.riqueza_inicial))
        quebrados = np.flatnonzero(pob.en_quiebra)
        if quebrados.size:
            pob.reemplazar(quebrados, p_tipos, escala_riqueza)
            riqueza = pob.riqueza(precio_nuevo)
        exposicion = pob.exposicion(precio_nuevo)
        fraccion, quiebra = comp.llamadas_de_margen(
            cc, exposicion, riqueza, pob.limite, pob.riqueza_inicial * escala_riqueza
        )
        pob.venta_forzada, pob.en_quiebra = fraccion, quiebra
        if cc.apalancamiento:
            pob.reapalancar(1.0 / DIAS_HABILES)

        # 8. Sentimiento con los retornos de hoy e indicadores técnicos
        cap = eco.acciones * precio
        r_mercado = float((cap * r).sum() / cap.sum())
        r_sector = np.bincount(eco.sector, weights=cap * r) / np.bincount(eco.sector, weights=cap)
        self.sentimiento.actualizar(r_mercado, r_sector)
        ind.agregar(log_p_nuevo)

        # Registro
        positiva = np.maximum(riqueza, 0.0)
        reg["log_valor"][t] = np.log(eco.valor_verdadero())
        reg["log_consenso"][t] = consenso
        reg["bpa_ttm"][t] = eco.bpa_ttm
        reg["valor_contable"][t] = eco.vc_publicado
        reg["volumen"][t] = pob.por_empresa(np.abs(total)) / np.maximum(capital_empresa, 1e-12)
        reg["fracciones"][t] = fracciones
        reg["riqueza_tipo"][t] = np.bincount(pob.tipo, weights=positiva, minlength=3) / positiva.sum()
        reg["tasa"][t] = eco.tasa
        reg["ciclo"][t] = eco.ciclo
        reg["sentimiento"][t] = self.sentimiento.global_
        reg["llamadas_margen"][t] = int(((fraccion > 0) & ~quiebra).sum())
        reg["quiebras"][t] = int(quiebra.sum())
        reg["confianza"][t] = pob.confianza.mean()
        apalancados = pob.limite > 1
        if apalancados.any():
            reg["apalancamiento"][t] = np.median(exposicion[apalancados] / np.maximum(riqueza[apalancados], 1e-9))

    def _restricciones(self, orden: np.ndarray, posiciones: np.ndarray, riqueza: np.ndarray) -> np.ndarray:
        """Prohibición de cortos para parte de los agentes y límite de apalancamiento."""
        pob = self.pob
        a = pob.agente
        orden = np.where(pob.puede_cortos[a], orden, np.maximum(orden, -posiciones))

        aumento = np.maximum(np.abs(posiciones + orden) - np.abs(posiciones), 0.0)
        margen = np.maximum(pob.limite * np.maximum(riqueza, 0.0) - pob.por_agente(np.abs(posiciones)), 0.0)
        pedido = pob.por_agente(aumento)
        factor = np.where(pedido > margen, margen / np.maximum(pedido, 1e-12), 1.0)
        return np.where(aumento > 0, orden * factor[a], orden)


def simular(config: Config | None = None, progreso: Callable[[int, int], None] | None = None) -> Resultados:
    return Simulacion(config).ejecutar(progreso)


__all__ = ["Resultados", "Simulacion", "simular", "TIPOS"]
