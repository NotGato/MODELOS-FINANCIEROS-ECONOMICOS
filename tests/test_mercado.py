import numpy as np

from mercado_abm.config import ConfigAgentes, ConfigMercado
from mercado_abm.mercado import CreadorDeMercado


def creador(**cambios):
    return CreadorDeMercado(ConfigMercado(**cambios), ConfigAgentes())


def test_comprar_sube_y_vender_baja():
    mm = creador()
    r = mm.retornos(np.array([10.0, -10.0, 0.0]), np.full(3, 1000.0))
    assert r[0] > 0 > r[1]
    assert r[2] == 0


def test_impacto_lineal_bajo_el_umbral_y_concavo_por_encima():
    mm = creador(limite_diario=0.0)
    liquidez = np.ones(1)
    lineal = mm.lambda_ * np.array([0.01, 0.02])
    np.testing.assert_allclose(mm.retornos(np.array([0.01, 0.02]), np.ones(2)), lineal)

    umbral = mm.cfg.umbral_impacto
    orden_grande = 10 * umbral / mm.lambda_
    r_grande = mm.retornos(np.array([orden_grande]), liquidez)[0]
    assert umbral < r_grande < 10 * umbral
    # Continuidad en el empalme
    justo = umbral / mm.lambda_
    assert abs(mm.retornos(np.array([justo * 1.0001]), liquidez)[0] - umbral) < 1e-4


def test_la_liquidez_cae_con_la_volatilidad_pero_tiene_suelo():
    mm = creador()
    capital = np.full(2, 100.0)
    calma, estres = mm.liquidez(capital, np.array([0.01, 0.10]))
    assert estres < calma
    # Aunque el capital desaparezca de golpe, la profundidad no baja de la mitad de su media
    assert mm.liquidez(np.zeros(2), np.full(2, 0.01)).min() > 0


def test_interruptor_de_volatilidad():
    mm = creador(limite_diario=0.07)
    r = mm.retornos(np.array([1e6, -1e6]), np.ones(2))
    np.testing.assert_allclose(r, [0.07, -0.07])
