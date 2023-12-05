import random
import numpy as np

# Parámetros iniciales
object_num = 10
x_rate = 0.5
pob_size = 20
father_num = int(pob_size * x_rate)

# Generar los pesos de los objetos de forma aleatoria
pesos = np.round(np.random.uniform(1, 10, object_num), 2)
print("Pesos:", pesos)

# Calcular los valores (profits) usando la fórmula dada
profits = pesos + 5
print("Profits:", profits)

# Tamaño de la mochila
V = pesos.sum() / 2
print("Tamaño de la mochila:", V)

# Generar una población inicial aleatoria, evitando cromosomas vacíos
def generar_cromosoma():
    cromosoma = np.zeros(object_num, dtype=int)
    while cromosoma.sum() == 0:
        cromosoma = np.random.choice([0, 1], object_num)
    return cromosoma

poblacion = [generar_cromosoma() for _ in range(pob_size)]

print("\nPoblación inicial:")
for cromosoma in poblacion:
    print(cromosoma)

def aptitud(cromosoma):
    """Función que calcula la aptitud de un cromosoma."""
    total_peso = np.dot(cromosoma, pesos)
    if total_peso > V:
        return 0
    else:
        return np.dot(cromosoma, profits)

# Reparar la población en caso de que un cromosoma exceda el peso de la mochila
def reparar_poblacion(poblacion):
    for cromosoma in poblacion:
        while np.dot(cromosoma, pesos) > V:
            indices = np.where(cromosoma == 1)[0]
            random_index = random.choice(indices)
            cromosoma[random_index] = 0

reparar_poblacion(poblacion)

print("\nPoblación reparada:")
for cromosoma in poblacion:
    fitness = aptitud(cromosoma)
    print(cromosoma, "- Fitness:", fitness)

# Mostrar los valores, pesos y fitness de cada cromosoma
print("\nDetalle de cada cromosoma:")
for cromosoma in poblacion:
    valor = np.dot(cromosoma, profits)
    peso = np.dot(cromosoma, pesos)
    print(cromosoma, "- Valor:", valor, "- Peso:", peso, "- Fitness:", aptitud(cromosoma))

def seleccion_por_torneo(poblacion):
    """Función que selecciona un padre usando el método de torneo."""
    seleccionados = random.sample(list(enumerate(poblacion)), 3)
    seleccionados = sorted(seleccionados, key=lambda x: aptitud(x[1]), reverse=True)
    return seleccionados[0]

# Seleccionar cromosomas usando el método de torneo
cromosomas_seleccionados_indices = [seleccion_por_torneo(poblacion) for _ in range(father_num)]

print("\nCromosomas seleccionados:")
for idx, cromosoma in cromosomas_seleccionados_indices:
    print(f"Índice del cromosoma: {idx}")
    print("Cromosoma:", cromosoma)
    print("Fitness:", aptitud(cromosoma))
    print("------")

# Dividir los cromosomas seleccionados en padres y madres
mitad = len(cromosomas_seleccionados_indices) // 2
indices_padres, indices_madres = cromosomas_seleccionados_indices[:mitad], cromosomas_seleccionados_indices[mitad:]

print("\nPadres:")
for idx, padre in indices_padres:
    print(padre, "- Fitness:", aptitud(padre))

print("\nMadres:")
for idx, madre in indices_madres:
    print(madre, "- Fitness:", aptitud(madre))

# Tasa de recombinación
recom_rate = 0.50

def two_point_crossover(parent1, parent2):
    while True:  # Continuar hasta que se realice la recombinación
        if random.random() < recom_rate:
            cut1, cut2 = sorted(np.random.choice(range(1, len(parent1)), 2, replace=False))
            child1 = np.concatenate([parent1[:cut1], parent2[cut1:cut2], parent1[cut2:]])
            child2 = np.concatenate([parent2[:cut1], parent1[cut1:cut2], parent2[cut2:]])
            return child1, child2, cut1, cut2

# Realizar la recombinación
print("\nResultado de la recombinación de dos puntos:")
for (idx1, padre), (idx2, madre) in zip(indices_padres, indices_madres):
    hijo1, hijo2, corte1, corte2 = two_point_crossover(padre, madre)
    print(f"Padre: {padre}")
    print(f"Madre: {madre}")
    if corte1 is not None:
        print(f"Puntos de corte: {corte1} y {corte2}")
    print(f"Hijo 1: {hijo1} - Fitness: {aptitud(hijo1)}")
    print(f"Hijo 2: {hijo2} - Fitness: {aptitud(hijo2)}")
    print("------")

# Mostrar el mejor hijo
hijos = [two_point_crossover(padre, madre) for (idx1, padre), (idx2, madre) in zip(indices_padres, indices_madres)]
hijos = [hijo for child_pair in hijos for hijo in child_pair[:2]]
mejor_hijo = max(hijos, key=aptitud)
print("\nMejor hijo:")
print(mejor_hijo, "- Fitness:", aptitud(mejor_hijo))
