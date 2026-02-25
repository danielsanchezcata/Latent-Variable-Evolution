#@title Benchmark

import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools

# =============================================================================
# Hyperparameters
# =============================================================================
# NK Landscape
N = 20  #@param {type:"integer"}
K = 5  #@param {type:"integer"}
ADJACENT = True  #@param {type:"boolean"}
NK_SEED = 42  #@param {type:"integer"}

# Constraint
RATIO_MIN = 0.1  #@param {type:"number"}
RATIO_MAX = 0.3  #@param {type:"number"}
PENALTY_COEF = 1e3  #@param {type:"number"}

# GA
POP_SIZE = 100  #@param {type:"integer"}
N_GEN = 200  #@param {type:"integer"}
CXPB = 0.7  #@param {type:"number"}
MUTPB = 0.3  #@param {type:"number"}
TOURNSIZE = 3  #@param {type:"integer"}
FLIPBIT_INDPB = 0.05  #@param {type:"number"}
ELITE_SIZE = 5  #@param {type:"integer"}

# General
SEED = 42  #@param {type:"integer"}

random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Problem
# =============================================================================
problem = C2NKLandscape(n=N, k=K, adjacent=ADJACENT, seed=NK_SEED,
                        ratio_min=RATIO_MIN, ratio_max=RATIO_MAX, penalty_coef=PENALTY_COEF)

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
toolbox.register("attr_bit", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bit, n=N)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=FLIPBIT_INDPB)

# =============================================================================
# Evolution
# =============================================================================
def evaluate(ind):
    raw_fit = problem.fitness(ind)
    cv = problem.constraint_violation(ind)
    penalized_fit = raw_fit + PENALTY_COEF * cv
    return penalized_fit, raw_fit, cv

pop = toolbox.population(n=POP_SIZE)

for ind in pop:
    penalized_fit, _, _ = evaluate(ind)
    ind.fitness.values = (-penalized_fit,)

history = {
    'fitness': {'mean': [], 'min': [], 'max': [], 'var': []},
    'constraint': {'mean': [], 'min': [], 'max': [], 'var': []}
}

def record_stats(pop):
    raw_fits = np.array([problem.fitness(ind) for ind in pop])
    cvs = np.array([problem.constraint_violation(ind) for ind in pop])
    history['fitness']['mean'].append(float(raw_fits.mean()))
    history['fitness']['min'].append(float(raw_fits.min()))
    history['fitness']['max'].append(float(raw_fits.max()))
    history['fitness']['var'].append(float(raw_fits.var()))
    history['constraint']['mean'].append(float(cvs.mean()))
    history['constraint']['min'].append(float(cvs.min()))
    history['constraint']['max'].append(float(cvs.max()))
    history['constraint']['var'].append(float(cvs.var()))

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
        if not ind.fitness.valid:
            penalized_fit, _, _ = evaluate(ind)
            ind.fitness.values = (-penalized_fit,)

    pop[:] = elite + offspring
    record_stats(pop)

    if (gen + 1) % 10 == 0:
        print(f"Gen {gen+1}/{N_GEN}, Fitness mean: {history['fitness']['mean'][-1]:.4f}, "
              f"min: {history['fitness']['min'][-1]:.4f}, "
              f"Constraint mean: {history['constraint']['mean'][-1]:.4f}")

# =============================================================================
# Results
# =============================================================================
best_ind = max(pop, key=lambda x: x.fitness.values[0])
print(f"\nBest solution: {best_ind}")
print(f"Best fitness (NK, higher=better): {-problem.fitness(best_ind):.4f}")
print(f"Constraint violation: {problem.constraint_violation(best_ind):.4f}")
print(f"Ratio of 1s: {sum(best_ind)/N:.2f}")
print(f"Feasible: {problem.is_feasible(best_ind)}")

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

gens = range(len(history['fitness']['mean']))

ax = axes[0]
ax.plot(gens, history['fitness']['mean'], label='Mean')
ax.plot(gens, history['fitness']['min'], label='Min')
ax.plot(gens, history['fitness']['max'], label='Max')
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
ax.set_title('Fitness over Generations (Baseline GA)')
ax.legend()

ax = axes[1]
ax.plot(gens, history['constraint']['mean'], label='Mean')
ax.plot(gens, history['constraint']['min'], label='Min')
ax.plot(gens, history['constraint']['max'], label='Max')
ax.set_xlabel('Generation')
ax.set_ylabel('Constraint Violation')
ax.set_title('Constraint Violation over Generations (Baseline GA)')
ax.legend()

plt.tight_layout()
plt.show()
