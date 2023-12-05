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

def aptitud(cromosoma):
    total_peso = np.dot(cromosoma, pesos)
    if total_peso > V:
        return V - total_peso
    else:
        return np.dot(cromosoma, profits)

def reparar_cromosoma(cromosoma):
    while np.dot(cromosoma, pesos) > V:
        indices = np.where(cromosoma == 1)[0]
        if not indices.size:
            break
        random_index = random.choice(indices)
        cromosoma[random_index] = 0
    return cromosoma

def seleccion_por_torneo(poblacion):
    seleccionados = random.sample(list(enumerate(poblacion)), 3)
    seleccionados = sorted(seleccionados, key=lambda x: aptitud(x[1]), reverse=True)
    return seleccionados[0][1]

def two_point_crossover(parent1, parent2):
    if random.random() < recom_rate:
        cut1, cut2 = sorted(np.random.choice(range(1, len(parent1)), 2, replace=False))
        child1 = np.concatenate([parent1[:cut1], parent2[cut1:cut2], parent1[cut2:]])
        child2 = np.concatenate([parent2[:cut1], parent1[cut1:cut2], parent2[cut2:]])
        return child1, child2
    return parent1, parent2

def mutacion(cromosoma):
    if random.random() < tasa_mutacion:
        indice = random.randint(0, len(cromosoma) - 1)
        cromosoma[indice] = 1 if cromosoma[indice] == 0 else 0
    return cromosoma

poblacion = [generar_cromosoma() for _ in range(pob_size)]

for gen in range(generaciones):
    # Reparar
    poblacion = [reparar_cromosoma(cromosoma) for cromosoma in poblacion]
    
    # Evaluar
    aptitudes = [aptitud(cromosoma) for cromosoma in poblacion]
    
    # Ordenar
    poblacion = [x for _, x in sorted(zip(aptitudes, poblacion), key=lambda pair: pair[0], reverse=True)]
    
    # Seleccion
    padres = [seleccion_por_torneo(poblacion) for _ in range(father_num)]
    
    hijos = []
    
    # Recombinacion
    for i in range(0, len(padres), 2):
        if i+1 < len(padres):  # asegurarse de que hay un par
            hijo1, hijo2 = two_point_crossover(padres[i], padres[i+1])
            hijos.append(hijo1)
            hijos.append(hijo2)
    
    # Mutacion
    hijos_mutados = [mutacion(hijo) for hijo in hijos]
    
    # Agregar hijos productos de mutación a población inicial
    poblacion += hijos_mutados
    poblacion = poblacion[:pob_size]  # Mantener el tamaño de la población

# Resultados finales
mejor_hijo_global = max(poblacion, key=aptitud)
peor_hijo_global = min(poblacion, key=aptitud)

print("Mejor hijo global:", mejor_hijo_global)
print("Mejor aptitud global:", aptitud(mejor_hijo_global))
print("Peor hijo global:", peor_hijo_global)
print("Peor aptitud global:", aptitud(peor_hijo_global))