import numpy as np
import random

# Datos del problema de la mochila:
n = 10
pesos = np.random.uniform(1, 10, n)
profits = pesos + 5

def aptitud(cromosoma):
    """Función que calcula la aptitud de un cromosoma."""
    total_peso = np.dot(cromosoma, pesos)
    if total_peso > V:
        return 0
    else:
        return np.dot(cromosoma, profits)

def cromosoma_peso(cromosoma):
    return np.dot(cromosoma, pesos)

def generate_population(pop_size, chrom_length):
    return [np.random.randint(2, size=chrom_length).tolist() for _ in range(pop_size)]

def reparar(cromosoma):
    while np.dot(cromosoma, pesos) > V:  # Utilizamos np.dot directamente aquí
        idx = np.random.randint(0, n)
        cromosoma[idx] = 0
    return cromosoma

def two_point_crossover(parent1, parent2):
    cut1, cut2 = sorted(random.sample(range(1, len(parent1)), 2))
    child1 = parent1[:cut1] + parent2[cut1:cut2] + parent1[cut2:]
    child2 = parent2[:cut1] + parent1[cut1:cut2] + parent2[cut2:]
    return child1, child2, cut1, cut2

def top_bottom_selection(population, xrate):
    selected = sorted(population, key=aptitud, reverse=False)[:int(len(population)*xrate)]
    n_select = len(selected)
    n_fathers = n_select // 2
    if n_fathers == 0:
        n_fathers = 1
    n_mothers = n_select - n_fathers
    fathers = selected[:n_fathers]
    mothers = selected[n_fathers:]
    return fathers, mothers

if __name__ == "__main__":
    tamaño_poblacion = 20
    V = np.sum(pesos) / 2

    print(f"Capacidad de la mochila: {V}")

    poblacion = generate_population(tamaño_poblacion, n)
    print("Población generada:")
    for chrom in poblacion:
        weight = cromosoma_peso(chrom)
        print(f"Cromosoma: {chrom}, Peso: {weight:.2f}")

    poblacion_reparada = [reparar(cromosoma.copy()) for cromosoma in poblacion]
    print("\nResultado de la función de reparación:")
    for chrom in poblacion_reparada:
        weight = cromosoma_peso(chrom)
        print(f"Cromosoma: {chrom}, Peso: {weight:.2f}, Aptitud: {aptitud(chrom):.2f}")

    xrate = 0.50
    fathers, mothers = top_bottom_selection(poblacion_reparada, xrate)
    print("\nResultado de la selección top-down:")
    print("Padres seleccionados:")
    for father in fathers:
        print(father)

    print("\nMadres seleccionadas:")
    for mother in mothers:
        print(mother)

    offspring = []
    recom_rate = 0.50

    print("\nResultado de la recombinación de dos puntos:")
    for father, mother in zip(fathers, mothers):
        child1, child2, cut1, cut2 = two_point_crossover(father, mother)
        while random.random() < recom_rate:
            child1, child2, cut1, cut2 = two_point_crossover(father, mother)

        offspring.extend([child1, child2])
        print(f"\nPadre: {father}")
        print(f"Madre: {mother}")
        print(f"Puntos de recombinación: {cut1}, {cut2}")
        print(f"Hijo 1: {child1}")
        print(f"Hijo 2: {child2}")
        print("-" * 50)

    best_child = max(offspring, key=aptitud)
    print("\nMejor hijo basado en el beneficio:")
    print(f"{best_child}, Aptitud: {aptitud(best_child):.2f}")
