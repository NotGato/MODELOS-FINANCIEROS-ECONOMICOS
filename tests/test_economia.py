import numpy as np

from mercado_abm.config import DIAS_TRIMESTRE, ConfigEconomia
from mercado_abm.economia import Economia, valor_dos_etapas, valor_gordon


def test_dos_etapas_coincide_con_gordon_si_el_crecimiento_no_cambia():
    bpa = np.array([1.0, 2.0, 3.0])
    g = np.array([0.06, 0.03, 0.0])
    roe = np.array([0.22, 0.15, 0.10])
    r = np.array([0.09, 0.07, 0.08])
    np.testing.assert_allclose(valor_dos_etapas(bpa, g, g, roe, r), valor_gordon(bpa, g, roe, r))


def test_crecer_con_roe_menor_que_el_coste_de_capital_destruye_valor():
    sin_crecer = valor_gordon(1.0, 0.0, 0.05, 0.08)
    creciendo = valor_gordon(1.0, 0.03, 0.05, 0.08)
    assert creciendo < sin_crecer


def test_extrapolar_crecimiento_sube_la_valoracion_de_forma_acotada():
    base = valor_dos_etapas(1.0, 0.03, 0.03, 0.15, 0.08)
    optimista = valor_dos_etapas(1.0, 0.15, 0.03, 0.15, 0.08)
    assert base < optimista < 2 * base


def test_publicacion_trimestral_y_datos_coherentes():
    eco = Economia(ConfigEconomia(), np.random.default_rng(0))
    publicaciones = np.zeros(eco.n_empresas, dtype=int)
    for t in range(4 * DIAS_TRIMESTRE):
        antes = eco.historial_bpa.copy()
        eco.avanzar(t)
        cambiadas = np.any(eco.historial_bpa != antes, axis=0)
        assert np.array_equal(cambiadas, eco.publica_hoy)
        publicaciones += eco.publica_hoy
    assert np.all(publicaciones == 4)
    assert np.all(eco.bpa_ttm > 0)
    assert np.all(eco.valor_verdadero() > 0)


def test_el_roe_publicado_no_deriva_lejos_del_objetivo():
    eco = Economia(ConfigEconomia(), np.random.default_rng(1))
    for t in range(10 * 252):
        eco.avanzar(t)
    desvio = np.abs(np.log(eco.roe_publicado / eco.roe_objetivo))
    assert np.median(desvio) < 0.5
