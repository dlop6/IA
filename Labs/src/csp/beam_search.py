"""
Tarea 2.2 – Beam Search para CSP

Explora el árbol de asignaciones nivel por nivel (una variable por nivel),
conservando solo las mejores K asignaciones parciales en cada paso.

La heurística de poda selecciona candidatos que violan la menor
cantidad de restricciones.
"""

from typing import List, Optional

from .problem import DOMAIN, VARIABLES, Assignment, is_valid, total_violations


def beam_search(k: int = 3) -> Optional[Assignment]:
    """
    Beam Search con ancho de haz *k*.

    Retorna una asignación completa válida o None si el haz colapsa
    antes de encontrar una solución.
    """
    # Comenzar con la asignación vacía
    beam: List[Assignment] = [{}]

    for var in VARIABLES:
        candidates: List[Assignment] = []

        for assignment in beam:
            for value in DOMAIN:
                new_assignment = dict(assignment)
                new_assignment[var] = value
                candidates.append(new_assignment)

        # Ordenar por total de violaciones (ascendente) – menos violaciones = mejor
        candidates.sort(key=lambda a: total_violations(a))

        # Podar: conservar solo los mejores K candidatos
        beam = candidates[:k]

        if not beam:
            return None  # el haz colapsó

    # Entre las asignaciones completas sobrevivientes, elegir una válida
    for assignment in beam:
        if is_valid(assignment):
            return assignment

    # No se encontró solución válida en el haz
    return None
