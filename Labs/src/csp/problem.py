"""
Definición del Problema CSP: Asignación de Microservicios a Servidores

Variables: M1..M8 (8 microservicios)
Dominio: {S1, S2, S3} (3 servidores físicos)

Restricciones:
  1. Capacidad: Ningún servidor aloja más de 3 microservicios.
  2. Pares anti-afinidad: (M1,M2), (M3,M4), (M5,M6), (M1,M5)
"""

from typing import Dict, List, Optional, Set, Tuple

# ---------- Constantes ----------
VARIABLES = [f"M{i}" for i in range(1, 9)]
DOMAIN = ["S1", "S2", "S3"]

ANTI_AFFINITY_PAIRS: List[Tuple[str, str]] = [
    ("M1", "M2"),
    ("M3", "M4"),
    ("M5", "M6"),
    ("M1", "M5"),
]

MAX_PER_SERVER = 3

# Alias de tipo para una asignación (parcial o completa)
Assignment = Dict[str, str]


# ---------- Verificación de restricciones ----------

def count_server_usage(assignment: Assignment) -> Dict[str, int]:
    """Cuenta cuántos microservicios están asignados a cada servidor."""
    usage: Dict[str, int] = {s: 0 for s in DOMAIN}
    for server in assignment.values():
        usage[server] += 1
    return usage


def capacity_violations(assignment: Assignment) -> int:
    """Retorna el número de servidores que exceden el límite de capacidad."""
    usage = count_server_usage(assignment)
    return sum(1 for s in DOMAIN if usage[s] > MAX_PER_SERVER)


def anti_affinity_violations(assignment: Assignment) -> int:
    """Retorna el número de pares anti-afinidad ubicados en el mismo servidor."""
    violations = 0
    for m_a, m_b in ANTI_AFFINITY_PAIRS:
        if m_a in assignment and m_b in assignment:
            if assignment[m_a] == assignment[m_b]:
                violations += 1
    return violations


def total_violations(assignment: Assignment) -> int:
    """Total de violaciones de restricciones para una asignación (parcial o completa)."""
    return capacity_violations(assignment) + anti_affinity_violations(assignment)


def is_valid(assignment: Assignment) -> bool:
    """Verifica si una asignación *completa* satisface todas las restricciones."""
    if len(assignment) != len(VARIABLES):
        return False
    return total_violations(assignment) == 0


def is_consistent(assignment: Assignment) -> bool:
    """Verifica si una asignación *parcial* no viola ninguna restricción hasta el momento."""
    return total_violations(assignment) == 0


def weight(assignment: Assignment) -> int:
    """Retorna 1 si es válida, 0 en caso contrario (según la especificación del problema)."""
    return 1 if is_valid(assignment) else 0
