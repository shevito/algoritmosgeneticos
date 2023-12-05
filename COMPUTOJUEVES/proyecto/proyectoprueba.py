import numpy as np

# Knapsack Problem Data
profits = np.array([
    135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240
])
weights = np.array([
    70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120
])
n_items = len(weights)  # number of items
C = 750  # total capacity of the knapsack

# ACO Parameters
n_ants = 100  # number of ants
n_iterations = 1000  # number of iterations
alpha = 1  # pheromone importance
beta = 2  # heuristic information importance
rho = 0.5  # pheromone evaporation rate
pheromone = np.ones(n_items) * 0.1  # initial pheromone level

# Heuristic information: profit-to-weight ratio
eta = profits / weights  # heuristic information

# Repair function to make the solution feasible
def repair_solution(solution, weights, capacity):
    total_weight = sum(solution * weights)
    while total_weight > capacity:
        # Find the item with the lowest profit-to-weight ratio that is in the solution
        ratio = np.where(solution == 1, profits / weights, np.inf)
        item_to_remove = np.argmin(ratio)
        # Remove this item from the solution
        solution[item_to_remove] = 0
        total_weight -= weights[item_to_remove]
    return solution

# ACO Algorithm
best_solution = np.zeros(n_items)
best_solution_profit = 0

for iteration in range(n_iterations):
    # Each ant constructs a solution
    for ant in range(n_ants):
        current_solution = np.zeros(n_items)
        current_capacity = C
        current_profit = 0

        while current_capacity > 0:
            # Calculate the probability for each item
            p = np.zeros(n_items)
            feasible = current_solution == 0
            for i in range(n_items):
                if feasible[i] and weights[i] <= current_capacity:
                    p[i] = (pheromone[i] ** alpha) * (eta[i] ** beta)
            p_sum = p.sum()
            if p_sum == 0:  # No items can be added anymore
                break
            p /= p_sum  # Normalize the probabilities
            
            # Select the next item based on the probabilities
            item = np.random.choice(n_items, p=p)
            if feasible[item] and weights[item] <= current_capacity:
                current_solution[item] = 1
                current_capacity -= weights[item]
                current_profit += profits[item]

        # Repair the solution if it is not feasible
        current_solution = repair_solution(current_solution, weights, C)

        # Update the best solution
        if current_profit > best_solution_profit:
            best_solution = current_solution
            best_solution_profit = current_profit

    # Update pheromones
    pheromone *= (1 - rho)  # Evaporation
    for i in range(n_items):
        if best_solution[i] == 1:
            # Increase pheromones for the items in the best solution
            pheromone[i] += 1 / (1 + (best_solution_profit / profits[i]))

print("Best Solution:", best_solution)
print("Best Solution Profit:", best_solution_profit)
