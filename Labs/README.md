# Lab 9 - Task 2 (Burglary-Earthquake-Alarm)

Implementación de inferencia bayesiana exacta por enumeración para el modelo:
- B: Robo
- E: Terremoto
- A: Alarma

## Estructura
- `src/model.py`: distribución conjunta `P(B,E,A)`
- `src/inference.py`: `inferencia_marginal(query, evidencia)`
- `tests/test_model.py`: pruebas del modelo
- `tests/test_inference.py`: pruebas de inferencia
- `notebooks/task2.ipynb`: ejecución y demostración de resultados

## Ejecutar pruebas
```bash
pytest -q
```

## Ejecutar notebook
Abrir `notebooks/task2.ipynb` y correr celdas en orden.

## Resultados clave esperados
- `P(A=1) = 0.0199`
- `P(B=1 | A=1) = 0.5025`
- `P(B=1 | A=1, E=1) = 0.0100`

## Conclusión
Se demuestra el efecto explain away: al observar `A=1`, el robo parece probable; al agregar `E=1`, la probabilidad de robo cae significativamente.
