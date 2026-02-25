#@title BipedalWalker CMA-ES Baseline

import numpy as np
import random

# =============================================================================
# Hyperparameters
# =============================================================================
# BipedalWalker
MAX_STEPS = 1600  #@param {type:"integer"}
N_EPISODES = 3  #@param {type:"integer"}

# Agent Architecture
HIDDEN_DIMS = [64, 64]  #@param
OUTPUT_ACTIVATION = 'tanh'  #@param {type:"string"}

# CMA-ES
SIGMA = 1.0  #@param {type:"number"}
LAMBDA = 50  #@param {type:"integer"}
N_GEN = 300  #@param {type:"integer"}

# General
SEED = 42  #@param {type:"integer"}

random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Setup
# =============================================================================
architecture = [24] + HIDDEN_DIMS + [4]

problem = BipedalWalkerProblem(max_steps=MAX_STEPS, n_episodes=N_EPISODES, seed=SEED)

search = BipedalWalker_CMAES(
    problem=problem,
    architecture=architecture,
    agent_class=MLP_Agent,
    sigma=SIGMA,
    lambda_=LAMBDA,
    n_gen=N_GEN,
    output_activation=OUTPUT_ACTIVATION
)

# =============================================================================
# Run CMA-ES
# =============================================================================
print("Running CMA-ES on BipedalWalker")
print(f"Architecture: {architecture}")
print(f"Weight dim: {search.weight_dim}")
print(f"Sigma: {SIGMA}, Lambda: {LAMBDA}, Generations: {N_GEN}")
print()

final_population = search.sample()

# =============================================================================
# Results
# =============================================================================
print(f"\nBest fitness (neg mean reward): {search.best_fitness:.1f}")
print(f"Best mean reward: {-search.best_fitness:.1f}")
print(f"Final behavior coverage: {search.history['behavior_coverage'][-1]:.4f}")
print(f"Population returned: {len(final_population)} individuals")

# =============================================================================
# Plot
# =============================================================================
search.plot_history()

# =============================================================================
# Show Best Agent
# =============================================================================
search.show_best()
