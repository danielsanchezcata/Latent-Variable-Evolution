#@title test

import numpy as np
import random
import torch
from tqdm.auto import tqdm
from deap import base, creator, tools

# =============================================================================
# Hyperparameters
# =============================================================================
# Problem
DIM = 10  #@param {type:"integer"}
X_MIN = -50.0  #@param {type:"number"}
X_MAX = 50.0  #@param {type:"number"}
PENALTY_COEF = 1e2  #@param {type:"number"}

# Data Generation
NUM_DATA_POINTS = 1000  #@param {type:"integer"}
POP_SIZE_DATA = 100  #@param {type:"integer"}
N_GEN_DATA = 50  #@param {type:"integer"}
CXPB_DATA = 0.5  #@param {type:"number"}
MUTPB_DATA = 0.2  #@param {type:"number"}
TOURNSIZE_DATA = 3  #@param {type:"integer"}
INDPB_DATA = 0.2  #@param {type:"number"}

# Latent Module
LATENT_DIM = 5  #@param {type:"integer"}
HIDDEN_DIM = 32  #@param {type:"integer"}
EPOCHS = 100  #@param {type:"integer"}
BATCH_SIZE = 128  #@param {type:"integer"}
LR = 5e-3  #@param {type:"number"}
VAL_SPLIT = 0.2  #@param {type:"number"}
BETA_VAE = 1.0  #@param {type:"number"}

# Evolvability Loss
EVOL_SIGMA = 0.5  #@param {type:"number"}
EVOL_FitTHETA = 1e-4  #@param {type:"number"}
EVOL_NovPHI = 0  #@param {type:"number"}
EVOL_LocETA = 1e-4  #@param {type:"number"}
EVOL_K = 50  #@param {type:"integer"}
EVOL_N_OFFSPRING = 30  #@param {type:"integer"}

# LVE GA
POP_SIZE_LVE = 100  #@param {type:"integer"}
N_GEN_LVE = 200  #@param {type:"integer"}
CXPB_LVE = 0.7  #@param {type:"number"}
MUTPB_LVE = 0.3  #@param {type:"number"}
TOURNSIZE_LVE = 3  #@param {type:"integer"}
INDPB_LVE = 0.2  #@param {type:"number"}
ELITE_SIZE = 5  #@param {type:"integer"}

INIT_FROM_DATASET = False  #@param {type:"boolean"}
INIT_EPSILON = 0.0  #@param {type:"number"}

SEED = 42  #@param {type:"integer"}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  #@param {type:"string"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# Data Generation
# =============================================================================
if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox_data = base.Toolbox()
toolbox_data.register("attr_float", random.uniform, X_MIN, X_MAX)
toolbox_data.register("individual", tools.initRepeat, creator.Individual,
                      toolbox_data.attr_float, n=DIM)
toolbox_data.register("population", tools.initRepeat, list, toolbox_data.individual)
toolbox_data.register("select", tools.selTournament, tournsize=TOURNSIZE_DATA)
toolbox_data.register("mate", tools.cxBlend, alpha=0.5)
toolbox_data.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=INDPB_DATA)


def single_ga_run(toolbox, problem):
    pop = toolbox.population(n=POP_SIZE_DATA)
    for ind in pop:
        ind.fitness.values = (-abs(problem.constraint(ind)),)

    for gen in range(N_GEN_DATA):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < CXPB_DATA:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for mutant in offspring:
            if random.random() < MUTPB_DATA:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        for ind in offspring:
            for i in range(len(ind)):
                ind[i] = np.clip(ind[i], X_MIN, X_MAX)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = (-abs(problem.constraint(ind)),)
        pop[:] = offspring
        best = tools.selBest(pop, 1)[0]
        if best.fitness.values[0] == 0.0:
            break

    best = tools.selBest(pop, 1)[0]
    return best if problem.is_feasible(best) else None


def generate_fn(toolbox, problem):
    dataset = []
    pbar = tqdm(total=NUM_DATA_POINTS, desc="Generating data")
    while len(dataset) < NUM_DATA_POINTS:
        ind = single_ga_run(toolbox, problem)
        if ind is not None:
            dataset.append(np.array(ind, dtype=np.float32))
            pbar.update(1)
    pbar.close()
    return dataset


# =============================================================================
# LVE Toolbox
# =============================================================================
toolbox_lve = base.Toolbox()
toolbox_lve.register("attr_latent", random.uniform, -2, 2)
toolbox_lve.register("individual", tools.initRepeat, creator.Individual,
                     toolbox_lve.attr_latent, n=LATENT_DIM)
toolbox_lve.register("population", tools.initRepeat, list, toolbox_lve.individual)
toolbox_lve.register("select", tools.selTournament, tournsize=TOURNSIZE_LVE)
toolbox_lve.register("mate", tools.cxBlend, alpha=0.5)
toolbox_lve.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=INDPB_LVE)


# =============================================================================
# Evolve Function
# =============================================================================
def evaluate_individual(ind, lve):
    problem = lve.data_generation.problem
    decoded = lve.decode([ind])[0]
    raw_fit = problem.fitness(decoded)
    cv = problem.constraint_violation(decoded)
    penalized_fit = raw_fit + problem.penalty_coef * cv
    return penalized_fit, raw_fit, cv


def record_generation_stats(pop, lve, history):
    problem = lve.data_generation.problem
    raw_fits, cvs = [], []
    for ind in pop:
        decoded = lve.decode([ind])[0]
        raw_fits.append(problem.fitness(decoded))
        cvs.append(problem.constraint_violation(decoded))
    raw_fits = np.array(raw_fits)
    cvs = np.array(cvs)
    history['fitness']['mean'].append(float(raw_fits.mean()))
    history['fitness']['min'].append(float(raw_fits.min()))
    history['fitness']['max'].append(float(raw_fits.max()))
    history['fitness']['var'].append(float(raw_fits.var()))
    history['constraint']['mean'].append(float(cvs.mean()))
    history['constraint']['min'].append(float(cvs.min()))
    history['constraint']['max'].append(float(cvs.max()))
    history['constraint']['var'].append(float(cvs.var()))


def evolve_fn(toolbox, lve, pop_size, n_gen):
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
            print(f"LVE Gen {gen+1}/{n_gen}, "
                  f"Fitness mean: {history['fitness']['mean'][-1]:.4f}, "
                  f"Constraint mean: {history['constraint']['mean'][-1]:.4f}")

    lve.evolution_history = history
    best_ind = max(pop, key=lambda x: x.fitness.values[0])
    return lve.decode([best_ind])[0]


# =============================================================================
# Run
# =============================================================================
print("Step 1: Data Generation")
problem = SphereProblem(dim=DIM, constraint_handling='penalty', penalty_coef=PENALTY_COEF)
data_gen = DataGeneration(problem, toolbox_data, generate_fn)
dataset = data_gen.generate()
print(f"Dataset size: {len(dataset)}")

print("\nStep 2: Build Archive Behaviors")
latent_module = BetaVAE(DIM, LATENT_DIM, HIDDEN_DIM, beta=BETA_VAE)
lve = LVE(data_gen, latent_module, toolbox_lve, device=DEVICE,
          init_from_dataset=INIT_FROM_DATASET, init_epsilon=INIT_EPSILON)
lve.dataset = dataset

lve.scaler.fit(np.array(dataset, dtype=np.float32))
archive_np = lve.scaler.transform(np.array(dataset, dtype=np.float32))
archive_tensor = torch.tensor(archive_np, dtype=torch.float32)

def sphere_fitness_fn(x_normalized):
    """x_normalized: (batch, DIM) tensor in normalized space. Returns (batch,) tensor."""
    scale = torch.tensor(lve.scaler.scale_, dtype=torch.float32, device=x_normalized.device)
    min_ = torch.tensor(lve.scaler.data_min_, dtype=torch.float32, device=x_normalized.device)
    x = x_normalized * scale + min_
    raw_fit = x.pow(2).mean(dim=1)
    cv = torch.clamp((45.0 - x).mean(dim=1), min=0.0)
    return raw_fit + PENALTY_COEF * cv

evol_loss = EvolvabilityLoss(
    archive_behaviors=archive_tensor,
    fitness_fn=sphere_fitness_fn,
    sigma=EVOL_SIGMA,
    theta=EVOL_FitTHETA,
    phi=EVOL_NovPHI,
    eta=EVOL_LocETA,
    k=EVOL_K,
    n_offspring=EVOL_N_OFFSPRING
)

print("\nStep 3: Train BetaVAE")
loss_history = lve.train_module(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                                val_split=VAL_SPLIT, evolvability_loss=evol_loss)

print("\nStep 4: Latent Variable Evolution")
best_solution = lve.evolve(POP_SIZE_LVE, N_GEN_LVE, evolve_fn)

print("\nResults")
print(f"Best solution fitness: {problem.fitness(best_solution):.4f}")
print(f"Constraint violation: {problem.constraint_violation(best_solution):.4f}")
print(f"Feasible: {problem.is_feasible(best_solution)}")

lve.plot_evolution()
