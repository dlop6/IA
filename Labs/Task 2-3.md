# Task 2 – Ingeniería de Datos


Utilizando Python (Pandas/NumPy), simule y procese un dataset. No se permite usar funciones mágicas de limpieza automática (como SimpleImputer de sklearn), deben hacerlo con lógica de programación para demostrar que entienden el proceso.

## 1. Generación de Dataset Sucio

a. Cree un DataFrame de 100 filas y 3 columnas: Edad, Salario y Compró_Producto (0 o 1).

b. Introduzca intencionalmente valores NaN (nulos) en el 10% de la columna Edad.

c. Genere un desbalance de clases en Compró_Producto: 90 filas deben ser '0' (No compró) y 10 filas '1' (Compró).

## 2. Manejo de Datos Faltantes (Imputación)

a. Escriba un algoritmo que recorra la columna Edad.

b. Si encuentra un valor faltante, rellénelo con el promedio de las edades existentes.

c. Pregunta extra en código (comentario): ¿En qué situación usar el promedio sería una mala idea y sería mejor usar la mediana?

## 3. Manejo de Datos Desbalanceados (Undersampling Manual)

a. Dado que la clase '0' es mayoritaria, implemente una función que realice Undersampling:
    i. Debe mantener todas las filas de la clase minoritaria ('1').
    ii. Debe seleccionar aleatoriamente un número de filas de la clase mayoritaria ('0') igual al número de filas de la clase minoritaria.
    iii. El resultado debe ser un nuevo DataFrame balanceado (aprox. 20 filas en total).

---

# Task 3 – Métricas de Desempeño

Escriba dos funciones en Python desde cero (usando math o numpy, pero sin usar sklearn.metrics) para calcular el error de dos listas de valores:

- `y_real = [100, 150, 200, 250, 300]` (Valores reales)
- `y_pred = [110, 140, 210, 240, 500]` (Predicciones - note el error masivo en el último dato)

## Requerimientos

1. Implemente la fórmula de RMSE vista en clase, como una función
2. Implemente la fórmula de MAE vista en clase, como una función
3. Comparación: Ejecute ambas funciones con los vectores dados.
    a. Imprima ambos resultados.
    b. Escriba un print final explicando: ¿Cuál de las dos métricas penalizó más el error del último dato (el 500)? ¿Por qué esto es importante si estamos prediciendo, por ejemplo, dosis de medicamentos?

---

## Entregas en Canvas

1. Documento PDF con las respuestas a cada task
2. Archivo `.py` o link a repositorio de GitHub (No se acepta entregas en otros medios)