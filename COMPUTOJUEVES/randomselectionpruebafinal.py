import numpy as np
import random

# Datos del problema de la mochila:
n = 10
pesos = np.random.uniform(1, 10, n)
profits = pesos + 5

def chromosome_weight(chromosome):
    return np.sum(pesos * chromosome)

def aptitud(cromosoma):
    """Función que calcula la aptitud de un cromosoma."""
    total_peso = np.dot(cromosoma, pesos)
    if total_peso > V:
        return 0
    else:
        return np.dot(cromosoma, profits)

def generate_population(pop_size, chrom_length):
    return [np.random.randint(2, size=chrom_length).tolist() for _ in range(pop_size)]

def reparar(cromosoma):
    while chromosome_weight(cromosoma) > V:
        idx = np.random.randint(0, n)
        cromosoma[idx] = 0
    return cromosoma

def two_point_crossover(parent1, parent2, recom_rate):
    while True:  # Continuar hasta que se realice la recombinación
        if random.random() < recom_rate:
            cut1, cut2 = sorted(random.sample(range(1, len(parent1)), 2))
            child1 = parent1[:cut1] + parent2[cut1:cut2] + parent1[cut2:]
            child2 = parent2[:cut1] + parent1[cut1:cut2] + parent2[cut2:]
            return child1, child2, cut1, cut2

def random_selection(population, xrate):
    """Método de selección aleatoria."""
    selected = random.sample(population, int(len(population)*xrate))
    n_select = len(selected)
    n_fathers = n_select // 2
    if n_fathers == 0:
        n_fathers = 1
    fathers = selected[:n_fathers]
    mothers = selected[n_fathers:]
    return fathers, mothers

if __name__ == "__main__":
    tamaño_poblacion = 20
    V = np.sum(pesos) / 2

    print(f"Capacidad de la mochila: {V}")

    poblacion = generate_population(tamaño_poblacion, n)
    print("\nAptitud de la población generada:")
    for chrom in poblacion:
        print(f"Cromosoma: {chrom}, Aptitud: {aptitud(chrom)}")

    poblacion_reparada = [reparar(cromosoma.copy()) for cromosoma in poblacion]
    print("\nAptitud después de la función de reparación:")
    for chrom in poblacion_reparada:
        weight = chromosome_weight(chrom)
        print(f"Cromosoma: {chrom}, Peso: {weight}, Aptitud: {aptitud(chrom)}")

    xrate = 0.50
    fathers, mothers = random_selection(poblacion_reparada, xrate)
    print("\nAptitud de los padres seleccionados:")
    for father in fathers:
        print(f"Cromosoma: {father}, Aptitud: {aptitud(father)}")

    print("\nAptitud de las madres seleccionadas:")
    for mother in mothers:
        print(f"Cromosoma: {mother}, Aptitud: {aptitud(mother)}")

    offspring = []
    recom_rate = 0.50

    print("\nAptitud después de la recombinación de dos puntos:")
    for father, mother in zip(fathers, mothers):
        child1, child2, cut1, cut2 = two_point_crossover(father, mother, recom_rate)
        offspring.extend([child1, child2])
        print(f"\nPadre: {father}, Aptitud: {aptitud(father)}")
        print(f"Madre: {mother}, Aptitud: {aptitud(mother)}")
        if cut1 and cut2:
            print(f"Puntos de recombinación: {cut1}, {cut2}")
        print(f"Hijo 1: {child1}, Aptitud: {aptitud(child1)}")
        print(f"Hijo 2: {child2}, Aptitud: {aptitud(child2)}")
        print("-" * 50)

    best_child = max(offspring, key=aptitud)  
    print("\nMejor hijo basado en la aptitud:")
    print(f"{best_child}, Aptitud: {aptitud(best_child)}")
