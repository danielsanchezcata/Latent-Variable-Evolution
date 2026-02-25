import numpy as np
import random
import torch

# =============================================================================
# Hyperparameters
# =============================================================================
MAX_STEPS = 200  #@param {type:"integer"}
N_EPISODES = 5  #@param {type:"integer"}

ARCHITECTURE = [4, 32, 2]  #@param
OUTPUT_ACTIVATION = 'argmax'  #@param {type:"string"}

BIN_RANGES = [(-2.4, 2.4), (0.0, 1.0)]  #@param
BIN_SIZES = [20, 20]  #@param

TOP_K = 5  #@param {type:"integer"}
N_SAMPLES = 200  #@param {type:"integer"}
MUTATION_SIGMA = 0.3  #@param {type:"number"}

N_STEPS = 100  #@param {type:"integer"}

SEED = 42  #@param {type:"integer"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# Setup
# =============================================================================
fitness_fn = lambda info: -np.mean(info['reward'])

collector = CartPoleCollector(max_steps=MAX_STEPS, n_episodes=N_EPISODES, seed=SEED)
bd = CartPoleBD_v1(bin_ranges=BIN_RANGES, bin_sizes=BIN_SIZES)
bm = MAPElitesBM(behavior_descriptor=bd, fitness_fn=fitness_fn, top_k=TOP_K)

sp = UniBinUniMemPSE(
    agent_class=MLP_Agent,
    architecture=ARCHITECTURE,
    agent_kwargs={'output_activation': OUTPUT_ACTIVATION},
    mutation_sigma=MUTATION_SIGMA,
    n_samples=N_SAMPLES,
)

me = MAPElite(
    search_phase=sp,
    collector=collector,
    behavior_matching=bm,
)

# =============================================================================
# Run
# =============================================================================
agent_tmp = MLP_Agent(ARCHITECTURE)
weight_dim = agent_tmp.get_weight_dim()
print(f"Weight dim: {weight_dim}")
print(f"Architecture: {ARCHITECTURE}")
print()

me.run(n_steps=N_STEPS)

# =============================================================================
# Results
# =============================================================================
f_min, f_mean, f_max = bm.fitness_stats()
print(f"\nFinal archive size: {bm.archive_size()}")
print(f"Final coverage: {bm.coverage():.4f}")
print(f"Best fitness (neg reward): {f_min:.2f}")
print(f"Best reward: {-f_min:.2f}")

# =============================================================================
# Plot
# =============================================================================
me.plot_history()
