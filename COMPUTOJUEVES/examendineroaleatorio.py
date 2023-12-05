import random
import numpy as np

# Initial parameters
object_num = 15
x_rate = 0.5
pob_size = 500
father_num = int(pob_size * x_rate)
generaciones = 1000
tasa_mutacion = 0.2
recom_rate = 0.5

profits = np.array([
    135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240
])
pesos = np.array([
    70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120
])
V = 750

def aptitud(cromosoma):
    total_peso = np.dot(cromosoma, pesos)
    if total_peso > V:
        return total_peso
    else:
        return np.dot(cromosoma, profits)

def generar_cromosoma():
    return np.random.choice([0, 1], object_num)

def reparar_cromosoma(cromosoma):
    # Intentar activar genes inactivos primero
    inactive_indices = np.where(cromosoma == 0)[0]
    for index in inactive_indices:
        cromosoma_temp = cromosoma.copy()
        cromosoma_temp[index] = 1
        if np.dot(cromosoma_temp, pesos) <= V:
            cromosoma = cromosoma_temp

    # Reducir peso si aún excede el límite
    while np.dot(cromosoma, pesos) > V:
        active_indices = np.where(cromosoma == 1)[0]
        if not active_indices.size:
            break
        random_index = random.choice(active_indices)
        cromosoma[random_index] = 0
        
    return cromosoma


def seleccion_aleatoria_ponderada_por_costos(poblacion, tasa_seleccion):
    aptitudes = [aptitud(c) for c in poblacion]
    suma_aptitudes = sum(aptitudes)
    probabilidades = [a / suma_aptitudes for a in aptitudes]
    indices_seleccionados = np.random.choice(len(poblacion), size=round(len(poblacion) * tasa_seleccion), p=probabilidades, replace=False)
    seleccionados = [poblacion[i] for i in indices_seleccionados]
    return seleccionados

def two_point_crossover(parent1, parent2):
    if random.random() < recom_rate:
        third = len(parent1) // 3
        two_thirds = 2 * third
        cut1 = random.randint(third - 1, third + 1)
        cut2 = random.randint(two_thirds - 1, two_thirds + 1)
        child1 = np.concatenate([parent1[:cut1], parent2[cut1:cut2], parent1[cut2:]])
        child2 = np.concatenate([parent2[:cut1], parent1[cut1:cut2], parent2[cut2:]])
        return child1, child2
    return parent1, parent2

def mutacion(cromosoma):
    if random.random() < tasa_mutacion:
        indice = random.randint(0, len(cromosoma) - 1)
        cromosoma[indice] = 1 if cromosoma[indice] == 0 else 0
    return cromosoma

elite_rate = 0.2
elite_size = int(pob_size * elite_rate)

# Execute 30 times
num_executions = 30
all_best_aptitudes = []
all_worst_aptitudes = []

for execution in range(num_executions):
    poblacion = [generar_cromosoma() for _ in range(pob_size)]
    mejor_hijo_global = None
    mejor_aptitud_global = float('-inf')
    peor_hijo_global = None
    peor_aptitud_global = float('inf')

    for gen in range(generaciones):
        poblacion = [reparar_cromosoma(cromosoma) for cromosoma in poblacion]
        aptitudes = [aptitud(cromosoma) for cromosoma in poblacion]
        poblacion = [x for _, x in sorted(zip(aptitudes, poblacion), key=lambda pair: pair[0], reverse=True)]
        elites = poblacion[:elite_size]
        seleccionados = seleccion_aleatoria_ponderada_por_costos(poblacion, x_rate)
        padres = seleccionados[:len(seleccionados)//2]
        madres = seleccionados[len(seleccionados)//2:]
        hijos = []

        for padre, madre in zip(padres, madres):
            hijo1, hijo2 = two_point_crossover(padre, madre)
            hijos.append(hijo1)
            hijos.append(hijo2)

        hijos_mutados = [mutacion(hijo) for hijo in hijos]
        poblacion = elites + hijos_mutados
        poblacion = poblacion[:pob_size]
        mejor_hijo_gen = max(hijos_mutados, key=aptitud)
        peor_hijo_gen = min(hijos_mutados, key=aptitud)

        if aptitud(mejor_hijo_gen) > mejor_aptitud_global:
            mejor_hijo_global = mejor_hijo_gen
            mejor_aptitud_global = aptitud(mejor_hijo_gen)

        if aptitud(peor_hijo_gen) < peor_aptitud_global:
            peor_hijo_global = peor_hijo_gen
            peor_aptitud_global = aptitud(peor_hijo_gen)

    all_best_aptitudes.append(mejor_aptitud_global)
    all_worst_aptitudes.append(peor_aptitud_global)
    print(f"Execution {execution + 1}:")
    print("\tBest fitness:", mejor_aptitud_global)
    print("\tWorst fitness:", peor_aptitud_global)
    print("="*40)

# Stats
print("Average Best Fitness:", np.mean(all_best_aptitudes))
print("Average Worst Fitness:", np.mean(all_worst_aptitudes))
print("Variance of Best Fitness:", np.var(all_best_aptitudes))
print("Variance of Worst Fitness:", np.var(all_worst_aptitudes))
