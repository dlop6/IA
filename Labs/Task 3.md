# Task 3 - SVM y Árboles de Decisión

## Reglas importantes:

- Se permite el uso de librerías como scikit-learn para realizar un problema complejo con variables continuas.
- Dataset: [League of Legends Diamond Ranked Games (10 min)](link)
- Variable Objetivo: blueWins

## Instrucciones generales:

### 1. Limpieza y Pre-procesamiento

a. **Eliminación de Redundancia**: El dataset tiene columnas espejo (ej: redGoldDiff es el negativo de blueGoldDiff). Elimine las columnas redundantes o aquellas que causarían Data Leakage (información que no tendríamos al minuto 10 si solo viéramos el lado azul).

b. **Escalado Obligatorio para SVM**: Separe en Train/Test (80/20) y aplique StandardScaler a sus variables numéricas. Explique en un comentario por qué SVM necesita esto y los Árboles no tanto.

### 2. Support Vector Machines:

a. Entene un modelo SVM con **Kernel Lineal**.
b. Entene un modelo SVM con **Kernel RBF** (Radial Basis Function).
c. Compare el Accuracy de ambos en el set de prueba.
d. Pregunta de Análisis: Si el Kernel RBF funcionó mejor (o igual), ¿qué nos dice esto sobre la separabilidad lineal de las partidas de LoL?

### 3. Árboles de Decisión:

a. Entrene un DecisionTreeClassifier.
b. **Visualización**: Use plot_tree o exporte el árbol para visualizarlo. Si es muy grande, limite la profundidad (max_depth=3) para poder verlo.
c. **Feature Importance**: Extráiga y grafique las 5 variables más importantes según el árbol (Slide 55). ¿Tiene sentido para un jugador de LoL? (Ej: ¿Es más importante el oro o los asesinatos?).

### 4. Comparación Final:

a. ¿Qué modelo tuvo mejor desempeño numérico (Accuracy)?
b. Si usted tuviera que explicarle a un analista de e-sports por qué ganó un equipo, ¿qué modelo usaría: el SVM o el Árbol? Justifique.