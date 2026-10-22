import numpy as np

from mercado_abm import Resultados, Simulacion, fase1, simular
from mercado_abm.config import actualizar, cargar
from mercado_abm.metricas import curtosis, hechos_estilizados, indice_hill

from .conftest import config_pequena


def test_misma_semilla_mismo_resultado():
    a = simular(config_pequena(dias=150))
    b = simular(config_pequena(dias=150))
    np.testing.assert_array_equal(a.log_precio, b.log_precio)
    c = simular(config_pequena(semilla=4, dias=150))
    assert not np.array_equal(a.log_precio, c.log_precio)


def test_resultados_finitos_y_dimensiones(resultados_pequenos):
    res = resultados_pequenos
    assert res.log_precio.shape == (res.dias + 1, len(res.nombres))
    for serie in (res.log_precio, res.log_valor, res.log_consenso, res.volumen, res.fracciones):
        assert np.isfinite(serie).all()
    np.testing.assert_allclose(res.fracciones.sum(axis=1), 1.0)


def test_guardar_y_cargar(resultados_pequenos, tmp_path):
    resultados_pequenos.guardar(tmp_path)
    cargado = Resultados.cargar(tmp_path)
    np.testing.assert_array_equal(cargado.log_precio, resultados_pequenos.log_precio)
    assert cargado.nombres == resultados_pequenos.nombres


def test_fase1_no_tiene_apalancamiento_ni_llamadas_de_margen():
    res = simular(fase1(config_pequena(dias=300)))
    assert res.llamadas_margen.sum() == 0
    assert np.all(res.apalancamiento == 0)


def test_restricciones_de_cortos_y_apalancamiento_se_respetan():
    sim = Simulacion(config_pequena(dias=400))
    sim.ejecutar()
    pob = sim.pob
    precio = np.exp(sim.indicadores.actual)
    posiciones = pob.posiciones(precio)
    sin_cortos = ~pob.puede_cortos[pob.agente]
    assert posiciones[sin_cortos].min() >= -1e-9
    # Fuera de los que tienen una llamada de margen pendiente, nadie supera su límite con colchón
    riqueza = pob.riqueza(precio)
    exposicion = pob.exposicion(precio)
    sanos = (pob.venta_forzada == 0) & ~pob.en_quiebra
    margen = 1 + sim.config.comportamiento.colchon_margen
    assert np.all(exposicion[sanos] <= pob.limite[sanos] * riqueza[sanos] * margen + 1e-6)


def test_ficheros_de_configuracion(tmp_path):
    ruta = tmp_path / "c.toml"
    ruta.write_text("[agentes]\nn_agentes = 123\n[comportamiento]\nanclaje = false\n")
    config = cargar(ruta)
    assert config.agentes.n_agentes == 123 and config.comportamiento.anclaje is False
    actualizar(config, {"mercado": {"impacto": 0.05}})
    assert config.mercado.impacto == 0.05


def test_metricas_sobre_datos_conocidos():
    rng = np.random.default_rng(0)
    assert abs(curtosis(rng.normal(size=200_000)) - 3) < 0.1
    assert 2.0 < indice_hill(rng.standard_t(3, size=200_000)) < 4.0


def test_hechos_estilizados_basicos():
    """Una simulación por defecto acortada ya muestra colas pesadas y retornos impredecibles."""
    config = config_pequena(semilla=1, dias=2500, agentes=1500)
    hechos = {h.nombre: h for h in hechos_estilizados(simular(config))}
    assert hechos["Autocorrelación de retornos (retardo 1)"].cumple
    assert hechos["Curtosis diaria (acciones, mediana)"].valor > 4
    assert hechos["Agrupación de volatilidad: autocorr. |r| retardo 1 (índice)"].valor > 0.05
    assert hechos["Correlación media entre acciones"].cumple
