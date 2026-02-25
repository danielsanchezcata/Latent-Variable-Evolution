#@title Example 

import numpy as np
import random
import torch
from math import comb
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

# Data Generation
NUM_DATA_POINTS = 1000  #@param {type:"integer"}

# Latent Module
LATENT_DIM = 10  #@param {type:"integer"}
HIDDEN_DIM = 64  #@param {type:"integer"}
EPOCHS = 100  #@param {type:"integer"}
BATCH_SIZE = 128  #@param {type:"integer"}
LR = 1e-3  #@param {type:"number"}
VAL_SPLIT = 0.2  #@param {type:"number"}
BETA_VAE = 1e-1  #@param {type:"number"}

# LVE GA
POP_SIZE_LVE = 100  #@param {type:"integer"}
N_GEN_LVE = 200  #@param {type:"integer"}
CXPB_LVE = 0.7  #@param {type:"number"}
MUTPB_LVE = 0.3  #@param {type:"number"}
TOURNSIZE_LVE = 3  #@param {type:"integer"}
INDPB_LVE = 0.2  #@param {type:"number"}
ELITE_SIZE = 5  #@param {type:"integer"}

# LVE Initialization
INIT_FROM_DATASET = False  #@param {type:"boolean"}
INIT_EPSILON = 0.0  #@param {type:"number"}

# General
SEED = 42  #@param {type:"integer"}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  #@param {type:"string"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# Problem
# =============================================================================
problem = C2NKLandscape(n=N, k=K, adjacent=ADJACENT, seed=NK_SEED,
                        ratio_min=RATIO_MIN, ratio_max=RATIO_MAX, penalty_coef=PENALTY_COEF)

# =============================================================================
# Data Generation (direct feasible sampling, no replacement)
# =============================================================================
toolbox_data = base.Toolbox()

def generate_fn(toolbox, problem):
    """Sample unique feasible binary strings proportional to C(N,m)."""
    n = problem.n
    m_min = int(np.ceil(problem.ratio_min * n))
    m_max = int(np.floor(problem.ratio_max * n))
    valid_ms = list(range(m_min, m_max + 1))
    
    weights = np.array([comb(n, m) for m in valid_ms], dtype=np.float64)
    weights /= weights.sum()
    
    dataset_set = set()
    dataset = []
    
    while len(dataset) < NUM_DATA_POINTS:
        m = np.random.choice(valid_ms, p=weights)
        ones_pos = tuple(sorted(np.random.choice(n, m, replace=False)))
        if ones_pos in dataset_set:
            continue
        dataset_set.add(ones_pos)
        x = np.zeros(n, dtype=np.float32)
        x[list(ones_pos)] = 1.0
        dataset.append(x)
    
    return dataset

# =============================================================================
# DEAP Setup for LVE (continuous latent space)
# =============================================================================
if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox_lve = base.Toolbox()
toolbox_lve.register("attr_latent", random.uniform, -2, 2)
toolbox_lve.register("individual", tools.initRepeat, creator.Individual, toolbox_lve.attr_latent, n=LATENT_DIM)
toolbox_lve.register("population", tools.initRepeat, list, toolbox_lve.individual)
toolbox_lve.register("select", tools.selTournament, tournsize=TOURNSIZE_LVE)
toolbox_lve.register("mate", tools.cxBlend, alpha=0.5)
toolbox_lve.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=INDPB_LVE)

# =============================================================================
# LVE Evolve Function
# =============================================================================
def evaluate_individual(ind, lve):
    """Evaluate individual with penalty for constraint violation."""
    problem = lve.data_generation.problem
    decoded = lve.decode([ind])[0]
    binary = (decoded >= 0.5).astype(int).tolist()
    raw_fit = problem.fitness(binary)
    cv = problem.constraint_violation(binary)
    penalized_fit = raw_fit + problem.penalty_coef * cv
    return penalized_fit, raw_fit, cv


def record_generation_stats(pop, lve, history):
    problem = lve.data_generation.problem
    raw_fitnesses = []
    constraint_violations = []
    for ind in pop:
        decoded = lve.decode([ind])[0]
        binary = (decoded >= 0.5).astype(int).tolist()
        raw_fitnesses.append(problem.fitness(binary))
        constraint_violations.append(problem.constraint_violation(binary))
    raw_fitnesses = np.array(raw_fitnesses)
    constraint_violations = np.array(constraint_violations)
    history['fitness']['mean'].append(float(raw_fitnesses.mean()))
    history['fitness']['min'].append(float(raw_fitnesses.min()))
    history['fitness']['max'].append(float(raw_fitnesses.max()))
    history['fitness']['var'].append(float(raw_fitnesses.var()))
    history['constraint']['mean'].append(float(constraint_violations.mean()))
    history['constraint']['min'].append(float(constraint_violations.min()))
    history['constraint']['max'].append(float(constraint_violations.max()))
    history['constraint']['var'].append(float(constraint_violations.var()))


def evolve_fn(toolbox, lve, pop_size, n_gen):
    """LVE evolution in latent space with penalty constraint handling."""
    history = {
        'fitness': {'mean': [], 'min': [], 'max': [], 'var': []},
        'constraint': {'mean': [], 'min': [], 'max': [], 'var': []}
    }
    
    pop = lve.init_population(pop_size)
    
    for ind in pop:
        penalized_fit, _, _ = evaluate_individual(ind, lve)
        ind.fitness.values = (-penalized_fit,)
    
    record_generation_stats(pop, lve, history)
    
    for gen in range(n_gen):
        elite = tools.selBest(pop, ELITE_SIZE)
        elite = list(map(toolbox.clone, elite))
        
        offspring = toolbox.select(pop, len(pop) - ELITE_SIZE)
        offspring = list(map(toolbox.clone, offspring))
        
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < CXPB_LVE:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB_LVE:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        for ind in offspring:
            if not ind.fitness.valid:
                penalized_fit, _, _ = evaluate_individual(ind, lve)
                ind.fitness.values = (-penalized_fit,)
        
        pop[:] = elite + offspring
        record_generation_stats(pop, lve, history)
        
        if (gen + 1) % 10 == 0:
            print(f"LVE Gen {gen+1}/{n_gen}, Fitness mean: {history['fitness']['mean'][-1]:.4f}, "
                  f"min: {history['fitness']['min'][-1]:.4f}, "
                  f"Constraint mean: {history['constraint']['mean'][-1]:.4f}")
    
    lve.evolution_history = history
    best_ind = max(pop, key=lambda x: x.fitness.values[0])
    decoded = lve.decode([best_ind])[0]
    return (decoded >= 0.5).astype(int)

# =============================================================================
# Run
# =============================================================================
print("Step 1: Data Generation")
data_gen = DataGeneration(problem, toolbox_data, generate_fn)
dataset = data_gen.generate()
print(f"Dataset size: {len(dataset)}")

print("\nStep 2: Train BinaryBetaVAE")
latent_module = BinaryBetaVAE(N, LATENT_DIM, HIDDEN_DIM, beta=BETA_VAE)
lve = LVE(data_gen, latent_module, toolbox_lve, device=DEVICE,
          init_from_dataset=INIT_FROM_DATASET, init_epsilon=INIT_EPSILON)
lve.dataset = dataset
loss_history = lve.train_module(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, val_split=VAL_SPLIT)

print("\nStep 3: Latent Variable Evolution")
best_solution = lve.evolve(POP_SIZE_LVE, N_GEN_LVE, evolve_fn)

print("\nResults")
print(f"Best solution: {best_solution}")
print(f"Best fitness (NK, higher=better): {-problem.fitness(best_solution):.4f}")
print(f"Constraint violation: {problem.constraint_violation(best_solution):.4f}")
print(f"Ratio of 1s: {sum(best_solution)/N:.2f}")
print(f"Feasible: {problem.is_feasible(best_solution)}")

lve.plot_evolution()
