Task 2 – Agente Connect Four con TD Learning
Para este laboratorio deberán tomar como punto de partida su implementación de Connect Four del
laboratorio anterior (que incluía Minimax y poda alpha-beta) y extenderla para incorporar un agente basado
en TD learning. A continuación se describen los requisitos mínimos y las decisiones de diseño que usted
debe tomar y justificar.
Task 2.1
Deberá implementar un agente que aprenda a jugar Connect Four mediante TD learning. Como mínimo, su
diseño debe contemplar y documentar las siguientes decisiones:
• Representación del estado. Defina cómo codificará el tablero de Connect Four como entrada para
su función de valor. Puede optar por una representación tabular (tabla Q), una función lineal sobre
features manuales, o una red neuronal. Cualquiera de las tres es válida, pero deberá justificar su
elección en términos de expresividad, costo computacional y viabilidad para el tamaño del espacio
de estados de Connect Four.
• Algoritmo de actualización. Elija entre TD(0)/SARSA (on-policy) o Q-learning (off-policy) e
impleméntelo correctamente. Recuerde que, como se discutió en clase, TD learning opera sobre
V^π(s; w) y requiere conocer Succ(s, a), mientras que Q-learning opera sobre Q̂_opt(s, a; w). Su
código debe reflejar esta distinción con claridad.
• Función de recompensa. Defina explícitamente las recompensas que utilizará. Como mínimo debe
considerar ganar, perder y empatar. Puede agregar recompensas intermedias si lo considera útil,
pero deberá argumentar si esto beneficia o perjudica el aprendizaje en este juego específico.
• Estrategia de exploración. Implemente una estrategia de exploración (ε-greedy u otra) y argumente
cómo ajustó sus parámetros a lo largo del entrenamiento.
• Ciclo de entrenamiento. Entrene al agente mediante self-play o contra un oponente fijo durante un
número suficiente de episodios. Documente cuántos episodios utilizó y por qué consideró que el
entrenamiento fue suficiente (puede apoyarse en curvas de aprendizaje u otras métricas).
Task 2.2
Una vez entrenado el agente, haga que compita en las siguientes tres condiciones, con un mínimo de 50
partidas por condición (150 en total).
• Condición A: Agente TD vs. Agente Minimax (del laboratorio anterior)
• Condición B: Agente TD vs. Agente Minimax con poda alpha-beta
• Condición C: Agente Minimax vs. Agente Minimax con poda alpha-beta (partidas de control)
Con los resultados de las 150 partidas, genere una visualización que muestre la distribución de victorias,
derrotas y empates para cada condición. La gráfica deberá incluir título, etiquetas de ejes y leyenda
adecuados. Entregue la gráfica en un PDF junto con su código.
Task 2.3
Grabe un video de no más de 10 minutos donde muestre lo siguiente:
• Una partida representativa de cada condición (3 partidas en total). Puede acelerar el video para
cumplir con el límite de tiempo.
• Una explicación breve de cómo funciona su agente TD a nivel conceptual.
• Un análisis de los resultados: ¿qué agente ganó más frecuentemente y por qué? ¿Cómo influyó la
estrategia de cada agente en ese resultado?.
