import random
import numpy as np

# Parámetros iniciales
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

def reparar_cromosoma(cromosoma):
    while np.dot(cromosoma, pesos) > V:
        indices = np.where(cromosoma == 1)[0]
        if not indices.size:
            break
        random_index = random.choice(indices)
        cromosoma[random_index] = 0
    return cromosoma

def torneo(piscina, k=3):
    padre = []
    madre = []
    flag = True
    for _ in range(len(piscina) // 2):
        competidores = random.sample(piscina, k)
        ganador = max(competidores, key=aptitud)
        if flag:
            padre.append(ganador)
        else:
            madre.append(ganador)
        flag = not flag
    return padre, madre

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

for ejecucion in range(30):  # Ejecutar todo 30 veces
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
        padres, madres = torneo(poblacion, k=5)
    
        hijos = []
        for padre, madre in zip(padres, madres):
            hijo1, hijo2 = two_point_crossover(padre, madre)
            hijos.append(hijo1)
            hijos.append(hijo2)
    
        hijos_mutados = [mutacion(hijo) for hijo in hijos]
        poblacion = elites + hijos_mutados
        poblacion = poblacion[:pob_size]

        mejor_hijo_gen = max(poblacion, key=aptitud)
        peor_hijo_gen = min(poblacion, key=aptitud)

        if aptitud(mejor_hijo_gen) > mejor_aptitud_global:
            mejor_hijo_global = mejor_hijo_gen
            mejor_aptitud_global = aptitud(mejor_hijo_gen)

        if aptitud(peor_hijo_gen) < peor_aptitud_global:
            peor_hijo_global = peor_hijo_gen
            peor_aptitud_global = aptitud(peor_hijo_gen)

    print(f"Ejecución #{ejecucion + 1}")
    print("Mejor hijo global de todas las generaciones:", mejor_hijo_global)
    print("Mejor aptitud global de todas las generaciones:", mejor_aptitud_global)
    print("Peor hijo global de todas las generaciones:", peor_hijo_global)
    print("Peor aptitud global de todas las generaciones:", peor_aptitud_global)
    print("-----")
