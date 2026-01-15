import random as rd


class Environment():
    def __init__(self):
        self.temperature = rd.randint(-10, 35)  
    
    def get_percept(self):
        return self.temperature
    
    def update(self, action):
        if action == "calentar":
            self.temperature += 1
        elif action == "enfriar":
            self.temperature -= 1
        else:
            pass 

class Agent():
    def __init__(self):
        pass
    
    def act(self, perception):
        if perception > 25:
            return "enfriar"
        elif perception < 18:
            return "calentar"
        else:
            return "esperar"

if __name__ == "__main__": # basicamente es un ejercicio para probar el agente y el entorno, especificamente un termostato simple.
    env = Environment()
    agent = Agent()
    
    for i in range(10):
        percept = env.get_percept() # obtener la percepcion de temperatura actual
        print(f"Percepción: {percept}°C")
        action = agent.act(percept) # decidir acción basada en la percepción
        print(f"Acción: {action}")
        env.update(action) # actualizar el entorno basado en la acción
        print(f"Temperatura actualizada: {env.get_percept()}°C\n")