"""Capa 4: formación de precios con un creador de mercado.

El creador de mercado ejecuta todas las órdenes y mueve el precio según la
demanda neta, relativa al capital de quienes siguen cada empresa. El impacto
es lineal para órdenes normales (tipo Kyle) y crece con la raíz cuadrada para
órdenes enormes, como las ventas forzadas de una cascada de llamadas de margen.
Cuando sube la volatilidad, la profundidad del mercado cae y el mismo volumen
mueve más el precio.
"""

from __future__ import annotations

import numpy as np

from .config import ConfigAgentes, ConfigMercado


class CreadorDeMercado:
    def __init__(self, cfg: ConfigMercado, cfg_agentes: ConfigAgentes):
        self.cfg = cfg
        # Con esta lambda, una demanda media de 1 mueve el precio ``impacto``
        self.lambda_ = cfg.impacto / cfg_agentes.tamano_orden
        self._capital_medio: np.ndarray | None = None

    def liquidez(self, capital_seguidores: np.ndarray, vol: np.ndarray) -> np.ndarray:
        """Profundidad de cada empresa.

        Sale del capital de quienes la siguen, pero no baja de la mitad de su media
        del último mes: aunque los especuladores se arruinen, otros inversores
        siguen dando contrapartida.
        """
        if self._capital_medio is None:
            self._capital_medio = capital_seguidores.copy()
        self._capital_medio += (capital_seguidores - self._capital_medio) / 20.0
        capital = np.maximum(capital_seguidores, 0.5 * self._capital_medio)
        estres = np.clip(vol / self.cfg.vol_referencia - 1.0, 0.0, self.cfg.estres_max)
        return capital / (1.0 + self.cfg.sensibilidad_liquidez * estres)

    def retornos(self, orden_neta: np.ndarray, liquidez: np.ndarray) -> np.ndarray:
        lineal = self.lambda_ * orden_neta / np.maximum(liquidez, 1e-12)
        # Por encima del umbral, raíz cuadrada (con la misma pendiente en el empalme)
        u = self.cfg.umbral_impacto
        x = np.abs(lineal)
        r = np.sign(lineal) * np.where(x <= u, x, 2.0 * np.sqrt(u * x) - u)
        if self.cfg.limite_diario > 0:
            r = np.clip(r, -self.cfg.limite_diario, self.cfg.limite_diario)
        return r
