Task 2 – Agente Connect Four con TD Learning
Para esta fase, programarán el motor de inferencia del clásico problema presentado por Judea Pearl:
Terremotos, Robos y Alarmas.
El Modelo:
• 𝐵: Robo (Burglary) ∈ {0,1}
• 𝐸: Terremoto (Earthquake) ∈ {0,1}
• 𝐴: Alarma suena (Alarm) ∈ {0,1}
Parámetros dados para este laboratorio:
Asuma el valor de 𝜖 = 0.01 (probabilidad base de un evento raro).
• 𝑃(𝐵 = 1) = 𝜖
• 𝑃(𝐸 = 1) = 𝜖
• La alarma es determinista basada en la disyunción lógica (OR): La alarma suena si y solo si hay un
robo o hay un terremoto. 𝑃(𝐴 = 1 ∣ 𝐵, 𝐸) = [𝐴 = (𝐵 ∨ 𝐸)]. (Si 𝐵 = 1 o 𝐸 = 1, 𝑃(𝐴 = 1) = 1. De lo
contrario 𝑃(𝐴 = 1) = 0).
Task 2.1 - Generador de Distribución Conjunta
Escriba una función en Python prob_conjunta(b, e, a) que reciba el estado de las tres variables y retorne su
probabilidad conjunta
𝑃(𝐵 = 𝑏, 𝐸 = 𝑒, 𝐴 = 𝑎)
.
• Hint: Recuerde la regla de la cadena para Redes Bayesiana:
𝑃(𝐵, 𝐸, 𝐴) = 𝑃(𝐵) ⋅ 𝑃(𝐸) ⋅ 𝑃(𝐴 ∣ 𝐵, 𝐸)
Task 2.2 - Inferencia Marginal
Una vez entrenado el agente, haga que compita en las siguientes tres condiciones, con un mínimo de 50
partidas por
Implemente una función inferencia_marginal(query, evidencia) que aplique la marginalización sumando
sobre las variables ocultas. Debe poder calcular:

1. Prior del efecto: Calcule 𝑃(𝐴 = 1) (Probabilidad de que la alarma suene sin saber nada más). Para
   esto, deberá iterar (sumar) sobre todas las combinaciones posibles de 𝐵 y 𝐸.
   Task 2.3 - Demostración del Efecto "Explain Away"
   El objetivo de este laboratorio es que comprueben mediante código el efecto Explain Away.
   Utilizando su función del Task 2.2 y la definición de probabilidad condicional (𝑃(𝑋 ∣ 𝑌) = 𝑃(𝑋,𝑌)
   𝑃(𝑌) ), su
   programa debe calcular e imprimir con precisión de 4 decimales lo siguiente:
2. Diagnóstico Simple: Calcule 𝑃(𝐵 = 1 ∣ 𝐴 = 1). Imprima el resultado. (Interpretación: "La alarma
   sonó, ¿cuál es la probabilidad de que sea un robo?").
3. Efecto Explain Away: Calcule 𝑃(𝐵 = 1 ∣ 𝐴 = 1, 𝐸 = 1). Imprima el resultado. (Interpretación: "La
   alarma sonó, pero me acabo de enterar por las noticias que hubo un terremoto. ¿Cuál es ahora la
   probabilidad de que sea un robo?").
4. Conclusión: Compare ambos números impresos. Escriba un comentario explicando cómo el código
   acaba de demostrar numéricamente el concepto de Explain Away, validando que el aumento de
   certeza en una causa (Terremoto) reduce la probabilidad de la otra causa (Robo), a pesar de que
   Robo y Terremoto son variables independientes.





Se calculó la probabilidad marginal de que suene la alarma sin evidencia adicional, es decir, P(A=1)**P**(**A**=**1**). Al enumerar todos los estados posibles del modelo y sumar las probabilidades conjuntas donde A=1**A**=**1**, se obtiene:

P(A=1)=0.0199**P**(**A**=**1**)**=**0.0199

Esto significa que la alarma suena aproximadamente en el 1.99%**1.99%** de los casos bajo la distribución definida por el modelo.
