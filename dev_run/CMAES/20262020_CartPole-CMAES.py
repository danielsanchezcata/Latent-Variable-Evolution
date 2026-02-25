#@title CartPole CMA-ES Baseline

import numpy as np
import random

# =============================================================================
# Hyperparameters
# =============================================================================
# CartPole
MAX_STEPS = 500  #@param {type:"integer"}
N_EPISODES = 3  #@param {type:"integer"}

# Agent Architecture
HIDDEN_DIMS = [32]  #@param

# CMA-ES
SIGMA = 1.0  #@param {type:"number"}
LAMBDA = 50  #@param {type:"integer"}
N_GEN = 100  #@param {type:"integer"}

# General
SEED = 42  #@param {type:"integer"}

random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Setup
# =============================================================================
architecture = [4] + HIDDEN_DIMS + [2]

problem = CartPoleProblem(max_steps=MAX_STEPS, n_episodes=N_EPISODES, seed=SEED)

search = CartPole_CMAES(
    problem=problem,
    architecture=architecture,
    agent_class=MLP_Agent,
    sigma=SIGMA,
    lambda_=LAMBDA,
    n_gen=N_GEN
)

# =============================================================================
# Run CMA-ES
# =============================================================================
print("Running CMA-ES on CartPole")
print(f"Architecture: {architecture}")
print(f"Weight dim: {search.weight_dim}")
print(f"Sigma: {SIGMA}, Lambda: {LAMBDA}, Generations: {N_GEN}")
print()

final_population = search.sample()

# =============================================================================
# Results
# =============================================================================
print(f"\nBest fitness (neg mean steps): {search.best_fitness:.1f}")
print(f"Best mean steps alive: {-search.best_fitness:.1f}")
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
