#@title Benchmark

import numpy as np
import random
from deap import base, creator, tools

# =============================================================================
# Hyperparameters
# =============================================================================
# NK Landscape
N = 20  #@param {type:"integer"}
K = 5  #@param {type:"integer"}
ADJACENT = True  #@param {type:"boolean"}
NK_SEED = 42  #@param {type:"integer"}

# GA
POP_SIZE = 100  #@param {type:"integer"}
N_GEN = 200  #@param {type:"integer"}
CXPB = 0.7  #@param {type:"number"}
MUTPB = 0.3  #@param {type:"number"}
TOURNSIZE = 3  #@param {type:"integer"}
MUT_SIGMA = 0.1  #@param {type:"number"}
MUT_INDPB = 0.2  #@param {type:"number"}
ELITE_SIZE = 5  #@param {type:"integer"}

# General
SEED = 42  #@param {type:"integer"}

random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Problem
# =============================================================================
problem = NKLandscape(n=N, k=K, adjacent=ADJACENT, seed=NK_SEED)

# =============================================================================
# DEAP Setup
# =============================================================================
if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=N)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=MUT_SIGMA, indpb=MUT_INDPB)

# =============================================================================
# Evolution
# =============================================================================
def binarize(ind):
    return [int(v >= 0.5) for v in ind]

def clamp(ind):
    for i in range(len(ind)):
        ind[i] = max(0.0, min(1.0, ind[i]))

pop = toolbox.population(n=POP_SIZE)

for ind in pop:
    ind.fitness.values = (-problem.fitness(binarize(ind)),)

history = {'fitness': {'mean': [], 'min': [], 'max': [], 'var': []}}

def record_stats(pop):
    fits = np.array([problem.fitness(binarize(ind)) for ind in pop])
    history['fitness']['mean'].append(float(fits.mean()))
    history['fitness']['min'].append(float(fits.min()))
    history['fitness']['max'].append(float(fits.max()))
    history['fitness']['var'].append(float(fits.var()))

record_stats(pop)

for gen in range(N_GEN):
    elite = tools.selBest(pop, ELITE_SIZE)
    elite = list(map(toolbox.clone, elite))

    offspring = toolbox.select(pop, len(pop) - ELITE_SIZE)
    offspring = list(map(toolbox.clone, offspring))

    for i in range(0, len(offspring) - 1, 2):
        if random.random() < CXPB:
            toolbox.mate(offspring[i], offspring[i + 1])
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    for ind in offspring:
        clamp(ind)

    for ind in offspring:
        if not ind.fitness.valid:
            ind.fitness.values = (-problem.fitness(binarize(ind)),)

    pop[:] = elite + offspring
    record_stats(pop)

    if (gen + 1) % 10 == 0:
        print(f"Gen {gen+1}/{N_GEN}, Fitness mean: {history['fitness']['mean'][-1]:.4f}, "
              f"min: {history['fitness']['min'][-1]:.4f}")

# =============================================================================
# Results
# =============================================================================
best_ind = max(pop, key=lambda x: x.fitness.values[0])
best_binary = binarize(best_ind)
print(f"\nBest solution: {best_binary}")
print(f"Best fitness (NK, higher=better): {-problem.fitness(best_binary):.4f}")

# =============================================================================
# Plot
# =============================================================================
import matplotlib.pyplot as plt

gens = range(len(history['fitness']['mean']))
plt.figure(figsize=(7, 5))
plt.plot(gens, history['fitness']['mean'], label='Mean')
plt.plot(gens, history['fitness']['min'], label='Min')
plt.plot(gens, history['fitness']['max'], label='Max')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness over Generations (Baseline GA)')
plt.legend()
plt.tight_layout()
plt.show()
