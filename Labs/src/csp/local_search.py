"""
Tarea 2.3 – Búsqueda Local mediante Modos Condicionales Iterados (ICM)

Comienza con una asignación completa aleatoria e iterativamente mueve cada
microservicio al servidor que minimiza las violaciones totales de restricciones.
Repite hasta converger o alcanzar un límite máximo de iteraciones.
"""

import random
from typing import Optional

from .problem import DOMAIN, VARIABLES, Assignment, is_valid, total_violations


def icm_search(
    max_iterations: int = 100,
    seed: Optional[int] = None,
) -> Optional[Assignment]:
    """
    Búsqueda Local ICM.

    Parámetros
    ----------
    max_iterations : int
        Número máximo de barridos completos sobre todas las variables.
    seed : int | None
        Semilla aleatoria para reproducibilidad.

    Retorna
    -------
    Una asignación válida si se encuentra, de lo contrario la mejor asignación
    alcanzada (que aún puede tener violaciones).
    """
    rng = random.Random(seed)

    # Asignación inicial aleatoria
    assignment: Assignment = {var: rng.choice(DOMAIN) for var in VARIABLES}

    for _ in range(max_iterations):
        if is_valid(assignment):
            return assignment

        changed = False

        for var in VARIABLES:
            current_value = assignment[var]
            current_violations = total_violations(assignment)

            best_value = current_value
            best_violations = current_violations

            for value in DOMAIN:
                if value == current_value:
                    continue
                assignment[var] = value
                v = total_violations(assignment)
                if v < best_violations:
                    best_violations = v
                    best_value = value

            assignment[var] = best_value
            if best_value != current_value:
                changed = True

        # Salida temprana si no hay mejora en un barrido completo
        if not changed:
            break

    return assignment if is_valid(assignment) else None
