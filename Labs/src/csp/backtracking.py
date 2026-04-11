"""
Tarea 2.1 – Búsqueda Backtracking con Forward Checking

Encuentra una asignación válida de 8 microservicios a 3 servidores
respetando las restricciones de capacidad y anti-afinidad.
"""

from typing import Dict, List, Optional, Set

from .problem import (ANTI_AFFINITY_PAIRS, DOMAIN, MAX_PER_SERVER, VARIABLES,
                      Assignment, is_consistent)


def _get_neighbors(var: str) -> List[str]:
    """Retorna las variables que comparten una restricción anti-afinidad con *var*."""
    neighbors = []
    for a, b in ANTI_AFFINITY_PAIRS:
        if a == var:
            neighbors.append(b)
        elif b == var:
            neighbors.append(a)
    return neighbors


def _forward_check(
    assignment: Assignment,
    domains: Dict[str, List[str]],
    var: str,
    value: str,
) -> Optional[Dict[str, List[str]]]:
    """
    Después de asignar var=value, poda valores incompatibles de los dominios
    de vecinos no asignados (Forward Checking / Lookahead).

    Retorna una copia actualizada de los dominios, o None si algún dominio
    queda vacío (indicando que esta rama es inválida).
    """
    # Copia profunda de dominios para poder hacer backtrack fácilmente
    new_domains: Dict[str, List[str]] = {v: list(d) for v, d in domains.items()}

    # --- Poda por anti-afinidad ---
    for neighbor in _get_neighbors(var):
        if neighbor in assignment:
            continue  # ya asignado
        if value in new_domains[neighbor]:
            new_domains[neighbor].remove(value)
        if not new_domains[neighbor]:
            return None  # dominio vacío → camino sin salida

    # --- Poda por capacidad ---
    # Contar cuántos están asignados a *value* (la asignación ya incluye var)
    count = sum(1 for v in assignment.values() if v == value)
    if count >= MAX_PER_SERVER:
        # No hay más espacio en este servidor → eliminarlo de todos los dominios no asignados
        for v in VARIABLES:
            if v not in assignment and v != var:
                if value in new_domains[v]:
                    new_domains[v].remove(value)
                if not new_domains[v]:
                    return None

    return new_domains


def backtracking_search() -> Optional[Assignment]:
    """
    Punto de entrada. Retorna una asignación completa válida o None.
    """
    domains: Dict[str, List[str]] = {v: list(DOMAIN) for v in VARIABLES}
    return _backtrack({}, domains)


def _backtrack(
    assignment: Assignment,
    domains: Dict[str, List[str]],
) -> Optional[Assignment]:
    # Caso base: todas las variables asignadas
    if len(assignment) == len(VARIABLES):
        return dict(assignment)

    # Seleccionar siguiente variable no asignada (heurística MRV: menos valores restantes)
    unassigned = [v for v in VARIABLES if v not in assignment]
    var = min(unassigned, key=lambda v: len(domains[v]))

    for value in domains[var]:
        # Verificar consistencia de esta asignación parcial
        assignment[var] = value
        if is_consistent(assignment):
            # Forward check
            new_domains = _forward_check(assignment, domains, var, value)
            if new_domains is not None:
                result = _backtrack(assignment, new_domains)
                if result is not None:
                    return result
        del assignment[var]

    return None
