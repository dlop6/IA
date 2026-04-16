"""Capa de modelo para la task 2."""

from itertools import product


EPSILON = 0.01
_BINARY_VALUES = (0, 1)


def _validate_binary(value, name):
    if value not in _BINARY_VALUES:
        raise ValueError(f"{name} debe ser 0 o 1")


def prob_b(b):
    _validate_binary(b, "b")
    return EPSILON if b == 1 else 1.0 - EPSILON


def prob_e(e):
    _validate_binary(e, "e")
    return EPSILON if e == 1 else 1.0 - EPSILON


def prob_a_dado_b_e(a, b, e):
    _validate_binary(a, "a")
    _validate_binary(b, "b")
    _validate_binary(e, "e")
    alarma_esperada = int(b or e)
    return 1.0 if a == alarma_esperada else 0.0


def estados_binarios():
    return product(_BINARY_VALUES, repeat=3)


def prob_conjunta(b, e, a):
    return prob_b(b) * prob_e(e) * prob_a_dado_b_e(a, b, e)
