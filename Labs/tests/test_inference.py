import math

import pytest

from src.inference import distribucion_posterior, inferencia_marginal


def test_prior_de_alarma():
	valor = inferencia_marginal({"a": 1}, {})
	assert math.isclose(valor, 0.0199, rel_tol=0.0, abs_tol=1e-12)


def test_diagnostico_simple():
	valor = inferencia_marginal({"b": 1}, {"a": 1})
	assert math.isclose(valor, 0.5025125628140703, rel_tol=0.0, abs_tol=1e-12)


def test_explain_away():
	valor = inferencia_marginal({"b": 1}, {"a": 1, "e": 1})
	assert math.isclose(valor, 0.01, rel_tol=0.0, abs_tol=1e-12)


def test_distribucion_posterior_normalizada():
	posterior = distribucion_posterior("b", {"a": 1})
	total = posterior[0] + posterior[1]
	assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize("query,evidencia", [({"x": 1}, {}), ({"b": 2}, {}), ({"b": 1, "a": 1}, {})])
def test_inferencia_valida_entradas(query, evidencia):
	with pytest.raises(ValueError):
		inferencia_marginal(query, evidencia)


def test_query_incompatible_con_evidencia_retorna_cero():
	assert inferencia_marginal({"a": 0}, {"a": 1}) == 0.0
