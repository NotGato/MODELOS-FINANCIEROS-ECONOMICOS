"""Mercado financiero artificial basado en agentes (fundamental + técnico + sesgos)."""

from .config import Config, cargar, fase1
from .simulacion import Resultados, Simulacion, simular

__all__ = ["Config", "Resultados", "Simulacion", "cargar", "fase1", "simular"]
__version__ = "0.1.0"
