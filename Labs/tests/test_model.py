import math

import pytest

from src.model import EPSILON, estados_binarios, prob_a_dado_b_e, prob_conjunta


def test_prob_a_dado_b_e_es_determinista_or():
	assert prob_a_dado_b_e(0, 0, 0) == 1.0
	assert prob_a_dado_b_e(1, 0, 0) == 0.0
	assert prob_a_dado_b_e(1, 1, 0) == 1.0
	assert prob_a_dado_b_e(0, 1, 0) == 0.0
	assert prob_a_dado_b_e(1, 0, 1) == 1.0
	assert prob_a_dado_b_e(0, 0, 1) == 0.0


def test_prob_conjunta_caso_simple_sin_eventos_raros():
	esperado = (1.0 - EPSILON) * (1.0 - EPSILON) * 1.0
	assert math.isclose(prob_conjunta(0, 0, 0), esperado, rel_tol=0.0, abs_tol=1e-12)


def test_prob_conjunta_caso_con_robo_y_alarma():
	esperado = EPSILON * (1.0 - EPSILON) * 1.0
	assert math.isclose(prob_conjunta(1, 0, 1), esperado, rel_tol=0.0, abs_tol=1e-12)


def test_prob_conjunta_asignacion_incompatible_es_cero():
	assert prob_conjunta(1, 0, 0) == 0.0
	assert prob_conjunta(0, 1, 0) == 0.0


def test_distribucion_conjunta_esta_normalizada():
	total = sum(prob_conjunta(b, e, a) for b, e, a in estados_binarios())
	assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize("b,e,a", [(-1, 0, 1), (0, 2, 1), (1, 0, 3)])
def test_prob_conjunta_valida_entradas_binarias(b, e, a):
	with pytest.raises(ValueError):
		prob_conjunta(b, e, a)
