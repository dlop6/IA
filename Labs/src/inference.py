"""Capa de inferencia para la task 2."""

from src.model import estados_binarios, prob_conjunta


_VARIABLES = ("b", "e", "a")


def _validate_variable_name(name):
    if name not in _VARIABLES:
        raise ValueError(f"variable desconocida: {name}")


def _validate_assignment(assignment):
    for var, value in assignment.items():
        _validate_variable_name(var)
        if value not in (0, 1):
            raise ValueError(f"{var} debe ser 0 o 1")


def _state_to_dict(b, e, a):
    return {"b": b, "e": e, "a": a}


def _is_consistent(state, evidence):
    for key, value in evidence.items():
        if state[key] != value:
            return False
    return True


def _prob_state(state):
    return prob_conjunta(state["b"], state["e"], state["a"])


def inferencia_marginal(query, evidencia):
    _validate_assignment(query)
    _validate_assignment(evidencia)
    if len(query) != 1:
        raise ValueError("query debe tener exactamente una variable")
    query_var = next(iter(query.keys()))
    if query_var in evidencia and evidencia[query_var] != query[query_var]:
        return 0.0

    evidencia_total = dict(evidencia)
    evidencia_total.update(query)
    numerador = 0.0
    denominador = 0.0

    for b, e, a in estados_binarios():
        state = _state_to_dict(b, e, a)
        if _is_consistent(state, evidencia_total):
            numerador += _prob_state(state)
        if _is_consistent(state, evidencia):
            denominador += _prob_state(state)

    if denominador == 0.0:
        raise ValueError("la evidencia tiene probabilidad cero")
    return numerador / denominador


def distribucion_posterior(variable, evidencia):
    _validate_variable_name(variable)
    _validate_assignment(evidencia)
    return {
        0: inferencia_marginal({variable: 0}, evidencia),
        1: inferencia_marginal({variable: 1}, evidencia),
    }
