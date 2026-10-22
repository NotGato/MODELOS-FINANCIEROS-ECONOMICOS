import pytest

from mercado_abm import Config, simular


def config_pequena(semilla: int = 3, dias: int = 600, agentes: int = 400) -> Config:
    config = Config()
    config.simulacion.dias = dias
    config.simulacion.semilla = semilla
    config.agentes.n_agentes = agentes
    return config


@pytest.fixture(scope="session")
def resultados_pequenos():
    return simular(config_pequena())
