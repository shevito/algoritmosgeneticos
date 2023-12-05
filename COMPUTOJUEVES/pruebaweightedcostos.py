import numpy as np
import random
import math

# Datos del problema de la mochila:
n = 10
pesos = np.random.uniform(1, 10, n)
profits = pesos + 5

def chromosome_weight(chromosome):
    """Función para calcular el peso total de un cromosoma."""
    return np.sum(pesos * chromosome)

def fitness(chromosome):
    """Función de aptitud basada en el problema de la mochila."""
    total_peso = np.dot(chromosome, pesos)
    if total_peso > V:
        return 0
    else:
        return np.dot(chromosome, profits)

def reparar(cromosoma):
    """Función de reparación basada en el problema de la mochila."""
    while chromosome_weight(cromosoma) > V:
        idx = np.random.randint(0, n)
        cromosoma[idx] = 0
    return cromosoma

def two_point_crossover(parent1, parent2, recom_rate):
    """Recombinación de dos puntos."""
    while True:
        if random.random() < recom_rate:
            cut1, cut2 = sorted(random.sample(range(1, len(parent1)), 2))
            child1 = parent1[:cut1] + parent2[cut1:cut2] + parent1[cut2:]
            child2 = parent2[:cut1] + parent1[cut1:cut2] + parent2[cut2:]
            return child1, child2, cut1, cut2
        
def weighted_cost_selection(population, xrate):
    POBLACION = len(population)
    costos = [sum(pesos[i] for i in range(len(c)) if c[i] == 1) for c in population]
    seleccionados = sorted(range(len(costos)), key=lambda k: costos[k])[:round(POBLACION * xrate)]
    return seleccionados

def generate_population(pop_size, chrom_length):
    """Generar una población con cromosomas aleatorios."""
    return [np.random.randint(2, size=chrom_length).tolist() for _ in range(pop_size)]

if __name__ == "__main__":
    tamaño_poblacion = 20
    V = np.sum(pesos) / 2
    print(f"Capacidad de la mochila: {V}")

    # Generación de la población
    poblacion = generate_population(tamaño_poblacion, n)
    print("Población generada:")
    for chrom in poblacion:
        print(chrom)

    # Proceso de reparación
    poblacion_reparada = [reparar(cromosoma.copy()) for cromosoma in poblacion]
    print("\nResultado de la función de reparación:")
    for chrom in poblacion_reparada:
        weight = chromosome_weight(chrom)
        print(f"Cromosoma: {chrom}, Peso: {weight}")

    xrate = 0.50
    selected_indices = weighted_cost_selection(poblacion_reparada, xrate)
    print("\nÍndices seleccionados con el método de selección aleatoria ponderada por costos:")
    for idx in selected_indices:
        print(idx)

    fathers = [poblacion_reparada[i] for idx, i in enumerate(selected_indices) if idx % 2 == 0]
    mothers = [poblacion_reparada[i] for idx, i in enumerate(selected_indices) if idx % 2 == 1]

    print("\nCromosomas de los padres seleccionados:")
    for father in fathers:
        print(f"Cromosoma: {father}, Aptitud: {fitness(father)}")

    print("\nCromosomas de las madres seleccionadas:")
    for mother in mothers:
        print(f"Cromosoma: {mother}, Aptitud: {fitness(mother)}")

    recom_rate = 0.50  # Tasa de recombinación
    offspring = []

    print("\nResultado de la recombinación de dos puntos:")
    for father, mother in zip(fathers, mothers):
        child1, child2, cut1, cut2 = two_point_crossover(father, mother, recom_rate)
        offspring.extend([child1, child2])
        print(f"\nPadre: {father}, Aptitud: {fitness(father)}")
        print(f"Madre: {mother}, Aptitud: {fitness(mother)}")
        if cut1 is not None and cut2 is not None:
            print(f"Puntos de recombinación: {cut1}, {cut2}")
        print(f"Hijo 1: {child1}, Aptitud: {fitness(child1)}")
        print(f"Hijo 2: {child2}, Aptitud: {fitness(child2)}")
        print("-" * 50)

    best_child = max(offspring, key=fitness)
    print("\nMejor hijo:")
    print(f"Cromosoma: {best_child}, Aptitud: {fitness(best_child)}")
