import numpy as np
from matplotlib import pyplot as plt
import MPA as MPA
import time
from tabulate import tabulate
from CMA_ES import run_cma_es, rastrigin, rosenbrock, himmelblau

def run_mpa():
    pop = 300  # Population size for MPA
    MaxIter = 100  # Maximum number of iterations for MPA
    dim = 20  # Dimension for MPA
    lb = np.array([-5] * dim)  # Lower bound for MPA
    ub = np.array([5] * dim)  # Upper bound for MPA
    fobj = himmelblau

    # Run MPA
    start_time = time.time()
    GbestScore, GbestPositon, Curve = MPA.MPA(pop, dim, lb, ub, MaxIter, fobj)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print MPA results
    print(f"MPA Running Time：{elapsed_time:.4f} Seconds")
    print('Best result：', GbestScore)
    #print('Best X：', GbestPositon)
    #print('Curve：', Curve)

    # Plotting the convergence curve for MPA
    plt.figure()
    plt.plot(Curve)
    plt.title('MPA_Himmelblau_20 Dimensions')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.show()

def run_cma_es_main():
    dimension = 20
    population_size = 300
    max_generations = 100
    initial_step_size = 0.5

    start_time = time.time()

    best_fitness_history, best_solutions = run_cma_es(himmelblau, dimension, population_size, max_generations, initial_step_size)

    end_time = time.time()
    elapsed_time = end_time - start_time

    table_headers = ["Generation", "Best Fitness"] + [f"Dimension {i}" for i in range(1, dimension + 1)]
    table_data = []

    for generation, (fitness, solution) in enumerate(zip(best_fitness_history, best_solutions)):
        table_row = [generation + 1, round(fitness, 4)] + [round(val, 4) for val in solution[1:]]
        table_data.append(table_row)

    # Print the table
    #print(tabulate(table_data, headers=table_headers, tablefmt="fancy_grid"))
    print(f"CMA Running Time：{elapsed_time:.4f} Seconds")

    # Plotting the fitness history for CMA-ES
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('CMA-ES_Himmelblau_20 Dimensions')
    plt.show()

if __name__ == "__main__":
    run_mpa()
    run_cma_es_main()
