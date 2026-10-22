import numpy as np

from mercado_abm import comportamiento as comp
from mercado_abm.config import ConfigComportamiento


def test_probabilidades_de_estrategia_suman_uno_y_el_desalineamiento_favorece_a_fundamentalistas():
    cambio = comp.CambioEstrategia(ConfigComportamiento(), np.random.default_rng(0))
    fracciones = np.array([1 / 3, 1 / 3, 1 / 3])
    calma = cambio.probabilidades(fracciones, desalineacion=0.0)
    burbuja = cambio.probabilidades(fracciones, desalineacion=0.5)
    assert np.isclose(calma.sum(), 1) and np.isclose(burbuja.sum(), 1)
    assert burbuja[0] > calma[0]


def test_sin_cambio_de_estrategia_nadie_cambia():
    cfg = ConfigComportamiento(cambio_estrategia=False)
    cambio = comp.CambioEstrategia(cfg, np.random.default_rng(0))
    tipo = np.array([0, 1, 2] * 10)
    nuevo = cambio.revisar(tipo, -np.ones(30), np.zeros(30, dtype=bool), np.array([1.0, 0, 0]))
    assert np.array_equal(nuevo, tipo)


def test_aversion_a_perdidas_penaliza_mas_los_dias_malos():
    cfg = ConfigComportamiento(memoria=0.0)
    con = comp.CambioEstrategia(cfg, np.random.default_rng(0))
    con.actualizar_aptitud(np.array([0.3, -0.3, 0.0]))
    assert abs(con.aptitud[1]) > abs(con.aptitud[0])

    sin = comp.CambioEstrategia(ConfigComportamiento(memoria=0.0, aversion_perdidas=False), np.random.default_rng(0))
    sin.actualizar_aptitud(np.array([0.3, -0.3, 0.0]))
    assert np.isclose(abs(sin.aptitud[1]), abs(sin.aptitud[0]))


def test_autoatribucion_los_aciertos_suben_la_confianza_mas_de_lo_que_la_bajan_los_fallos():
    cfg = ConfigComportamiento()
    innata = np.ones(2)
    confianza = comp.actualizar_confianza(cfg, np.ones(2), innata, np.array([0.5, -0.5]))
    assert confianza[0] - 1 > 1 - confianza[1] > 0


def test_atencion_reacciona_a_volatilidad_inusual_y_a_noticias():
    cfg = ConfigComportamiento()
    agente = np.zeros(3, dtype=int)
    empresa = np.arange(3)
    p = comp.probabilidad_atencion(
        cfg,
        vol=np.array([0.01, 0.03, 0.01]),
        vol_habitual=np.full(3, 0.01),
        noticias=np.array([0.0, 0.0, 3.0]),
        confianza=np.ones(1),
        agente=agente,
        empresa=empresa,
    )
    assert p[1] > p[0] and p[2] > p[0]
    assert np.all(p <= cfg.atencion_max)


def test_llamada_de_margen_y_quiebra():
    cfg = ConfigComportamiento()
    exposicion = np.array([100.0, 300.0, 150.0])
    riqueza = np.array([100.0, 100.0, 1.0])
    limite = np.array([1.0, 2.0, 2.0])
    fraccion, quiebra = comp.llamadas_de_margen(cfg, exposicion, riqueza, limite, np.full(3, 100.0))
    assert fraccion[0] == 0  # dentro del límite
    assert 0 < fraccion[1] <= cfg.liquidacion_diaria_max  # 3x con límite 2x: venta parcial
    assert quiebra[2] and fraccion[2] == 1.0


def test_disposicion_vende_ganadoras_y_marca_perdedoras():
    n_esp = np.array([10.0, 10.0, 0.0])
    coste = np.array([50.0, 150.0, 0.0])
    realizar, en_perdidas = comp.ordenes_disposicion(
        np.array([0.5]), n_esp, coste, np.log(np.array([100.0])), np.zeros(3, dtype=int), np.zeros(3, dtype=int)
    )
    assert realizar[0] < 0  # ganadora: vende antes
    assert realizar[1] == 0 and en_perdidas[1]  # perdedora: la mantiene
    assert realizar[2] == 0 and not en_perdidas[2]
