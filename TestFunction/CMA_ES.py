import cma
import numpy as np
from tabulate import tabulate


def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def himmelblau(x):
    return (x[0] ** 2.0 + x[1] - 11.0) ** 2.0 + (x[0] + x[1] ** 2.0 - 7.0) ** 2.0


def rastrigin(x):
    return 10 * len(x) + sum(x ** 2.0 - 10 * np.cos(2 * np.pi * x))


def objective_function(x):
    # Uncomment the line below for Rosenbrock
    # return rosenbrock(x)

    # Uncomment the line below for Himmelblau
    # return himmelblau(x)

    # Uncomment the line below for Rastrigin
    return rastrigin(x)
    pass


def run_cma_es(objective_function, dimension, population_size, max_generations, initial_step_size):
    es = cma.CMAEvolutionStrategy(dimension * [0.0], initial_step_size,
                                  {'popsize': population_size, 'maxiter': max_generations})

    best_fitness_history = []  # Store best fitness values for each generation
    best_solutions = []  # Store best solutions for each generation

    for generation in range(max_generations):
        solutions = es.ask()
        fitness_values = [objective_function(x) for x in solutions]
        es.tell(solutions, fitness_values)

        best_solution = es.result.xbest
        best_fitness = es.result.fbest
        best_fitness_history.append(best_fitness)  # Save best fitness for this generation
        best_solutions.append([best_fitness] + list(best_solution))  # Save best solution for this generation
    return best_fitness_history, best_solutions