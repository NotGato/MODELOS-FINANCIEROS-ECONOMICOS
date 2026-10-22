"""Indicadores técnicos vectorizados por empresa.

Cada regla devuelve una señal en [-1, 1] (positiva = comprar). Un chartista usa
una sola regla durante toda su vida, lo que reparte el mercado entre reglas de
seguimiento de tendencia y reglas contrarias.
"""

from __future__ import annotations

import numpy as np

REGLAS = (
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "medias_5_20",
    "medias_20_100",
    "rsi_14",
    "ruptura_55",
    "soportes_resistencias",
)
# Reglas que dependen de anclas psicológicas (números redondos)
REGLAS_ANCLAJE = ("soportes_resistencias",)
REGLA_SUSTITUTA = "momentum_20"  # se usa en su lugar si el anclaje está desactivado

_LAMBDA_VOL = 0.94
_LAMBDA_VOL_LENTA = 1.0 - 1.0 / 250  # memoria de ~1 año: la volatilidad "normal"
_LAMBDA_VOL_RECIENTE = 1.0 - 1.0 / 20  # memoria de ~1 mes


class Indicadores:
    def __init__(self, historia_log_precio: np.ndarray, capacidad: int):
        """``historia_log_precio`` (H×M) debe tener al menos 253 días para el máximo anual."""
        h, m = historia_log_precio.shape
        self.lp = np.empty((capacidad, m))
        self.lp[:h] = historia_log_precio
        self.suma = np.zeros((capacidad + 1, m))
        self.suma[1 : h + 1] = np.cumsum(historia_log_precio, axis=0)
        self.t = h - 1

        r = np.diff(historia_log_precio, axis=0)
        self.var = np.full(m, r[:20].var(axis=0) + 1e-8)
        self.var_lenta = r.var(axis=0) + 1e-8
        # Varianza reciente separada en días de bajada y de subida
        self.var_bajadas = 0.5 * self.var.copy()
        self.var_subidas = 0.5 * self.var.copy()
        self.media_subida = np.maximum(r[:14], 0).mean(axis=0)
        self.media_bajada = np.maximum(-r[:14], 0).mean(axis=0)
        for x in r[14:]:
            self._actualizar_estados(x)

    def _actualizar_estados(self, r: np.ndarray) -> None:
        self.var = _LAMBDA_VOL * self.var + (1 - _LAMBDA_VOL) * r**2
        self.var_lenta = _LAMBDA_VOL_LENTA * self.var_lenta + (1 - _LAMBDA_VOL_LENTA) * r**2
        bajada = r < 0
        self.var_bajadas = _LAMBDA_VOL_RECIENTE * self.var_bajadas + (1 - _LAMBDA_VOL_RECIENTE) * r**2 * bajada
        self.var_subidas = _LAMBDA_VOL_RECIENTE * self.var_subidas + (1 - _LAMBDA_VOL_RECIENTE) * r**2 * ~bajada
        self.media_subida = (13 * self.media_subida + np.maximum(r, 0)) / 14
        self.media_bajada = (13 * self.media_bajada + np.maximum(-r, 0)) / 14

    def agregar(self, log_precio: np.ndarray) -> None:
        r = log_precio - self.lp[self.t]
        self.t += 1
        self.lp[self.t] = log_precio
        self.suma[self.t + 1] = self.suma[self.t] + log_precio
        self._actualizar_estados(r)

    # ---------------------------------------------------------------- consultas
    @property
    def actual(self) -> np.ndarray:
        return self.lp[self.t]

    @property
    def vol(self) -> np.ndarray:
        """Volatilidad diaria EWMA (RiskMetrics)."""
        return np.sqrt(self.var)

    @property
    def vol_lenta(self) -> np.ndarray:
        """Volatilidad a la que los agentes están acostumbrados (EWMA de un año)."""
        return np.sqrt(self.var_lenta)

    def vol_percibida(self, aversion: float = 1.0) -> np.ndarray:
        """Volatilidad del último mes, con las bajadas pesando ``aversion`` veces más que las subidas.

        Los pesos están normalizados para que, con bajadas y subidas igual de
        grandes, el resultado coincida con la volatilidad simétrica.
        """
        peso_bajadas = 2.0 * aversion / (1.0 + aversion)
        peso_subidas = 2.0 / (1.0 + aversion)
        return np.sqrt(peso_bajadas * self.var_bajadas + peso_subidas * self.var_subidas)

    def vol_ventana(self, n: int = 20) -> np.ndarray:
        return np.diff(self.lp[self.t - n : self.t + 1], axis=0).std(axis=0)

    def media(self, n: int) -> np.ndarray:
        return (self.suma[self.t + 1] - self.suma[self.t + 1 - n]) / n

    def maximo(self, n: int) -> np.ndarray:
        return self.lp[self.t - n + 1 : self.t + 1].max(axis=0)

    def _momentum(self, n: int, vol: np.ndarray) -> np.ndarray:
        z = (self.actual - self.lp[self.t - n]) / (vol * np.sqrt(n))
        return np.tanh(z / 1.5)

    def _cruce_medias(self, corta: int, larga: int, vol: np.ndarray) -> np.ndarray:
        z = (self.media(corta) - self.media(larga)) / (vol * np.sqrt((larga - corta) / 3.0))
        return np.tanh(z / 1.5)

    def _rsi(self) -> np.ndarray:
        rsi = 100.0 - 100.0 / (1.0 + self.media_subida / np.maximum(self.media_bajada, 1e-12))
        return np.clip((50.0 - rsi) / 25.0, -1.0, 1.0)  # contraria: sobrecompra -> vender

    def _ruptura(self, n: int) -> np.ndarray:
        ventana = self.lp[self.t - n : self.t]
        alto, bajo = ventana.max(axis=0), ventana.min(axis=0)
        posicion = 2.0 * (self.actual - bajo) / np.maximum(alto - bajo, 1e-9) - 1.0
        return np.clip(posicion, -1.0, 1.0)

    def _soportes_resistencias(self) -> np.ndarray:
        """Compra cerca de un número redondo por debajo (soporte), vende cerca de uno por encima."""
        precio = np.exp(self.actual)
        paso = 10.0 ** np.floor(np.log10(precio)) / 2.0
        soporte = np.floor(precio / paso) * paso
        resistencia = soporte + paso
        return np.exp(-np.log(precio / soporte) / 0.01) - np.exp(-np.log(resistencia / precio) / 0.01)

    def senales(self) -> np.ndarray:
        """Matriz (n_reglas × M) con la señal de cada regla para cada empresa."""
        vol = np.maximum(self.vol, 1e-4)
        return np.vstack(
            [
                self._momentum(5, vol),
                self._momentum(20, vol),
                self._momentum(60, vol),
                self._cruce_medias(5, 20, vol),
                self._cruce_medias(20, 100, vol),
                self._rsi(),
                self._ruptura(55),
                self._soportes_resistencias(),
            ]
        )
