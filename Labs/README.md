# CC3045 – Inteligencia Artificial | Laboratorio 10

**LogiTrack — Patio Logístico Inteligente**  
Módulo de seguimiento probabilístico basado en Filtrado de Partículas (SIR).

---

## Estructura

```
Labs/
├── src/
│   └── lab10_task3.ipynb   # Implementación completa Task 3
└── output/                 # Artefactos generados (gráficas, etc.)
```

---

## Contenido del notebook

### Task 3.1 — Implementación base
- Filtrado de Partículas (SIR) desde cero en Python, sin `pgmpy`.
- `simular_vehiculo(pasos)` genera trayectoria real oculta y lecturas de sensor RFID ruidosas.
- Modelo de sensor: `P(sensor=h)=0.6`, `P(sensor=h±1)=0.2`, resto uniforme.
- Dinámica: movimiento ±1 carril o estático con probabilidad uniforme; reflexión en bordes.
- Visualización con `matplotlib`: trayectoria real, estimación (media de partículas) y nube de partículas.

### Task 3.2 — Análisis experimental K=5 vs K=20
- 50 simulaciones independientes de 30 pasos.
- Gráfica de error promedio absoluto por paso de tiempo.
- Análisis de los 5 peores escenarios con K=5.
- Tabla comparativa:

| Métrica | K=5 | K=20 |
|---|---|---|
| Error promedio | ~3.3 carriles | ~0.6 carriles |
| Error máximo | ~19 carriles | ~14 carriles |
| % sims con error > 5 carriles | ~46% | ~12% |

### Task 3.3 — Heurística de detección de colapso
- Métrica: `std(partículas)` — interpretación directa en carriles, O(K) en tiempo.
- `alerta_colapso(particulas, umbral)` → `True` cuando `std < 0.3`.
- Umbral calibrado experimentalmente: **0% de falsas alarmas** sobre 6 000 muestras.
- Demostración en 3 escenarios:
  - **Escenario A**: movimiento suave → sin alertas.
  - **Escenario B**: lectura errónea puntual → colapso recuperable.
  - **Escenario C**: sensor persistentemente erróneo → colapso irrecuperable.

### Task 3.4 — Dictamen ejecutivo
Recomendación técnica al gerente de operaciones sobre viabilidad de K=5 y plan de acción.

---

## Requisitos

```
numpy
matplotlib
```

Instalar con:
```bash
pip install -r requirements.txt
```

---

## Ejecución

Abrir `Labs/src/lab10_task3.ipynb` en Jupyter o VS Code y ejecutar todas las celdas en orden.
