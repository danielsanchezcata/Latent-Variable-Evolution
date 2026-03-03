#@title EvoGym Walker MAP-Elites Baseline
#@markdown Evolves soft robot morphologies on Walker-v0 using MAP-Elites.
#@markdown BD: material composition ratios. Fitness: horizontal displacement.
#@markdown Open-loop sinusoidal controller (no neural network).

import numpy as np
import random
import time
import os
from dev.SSLVE.Main import MAPElite
from dev.SSLVE.SearchPhases import UniBinUniMemPSE
from dev.SSLVE.AgentModules import EvoGymAgent
from dev.SSLVE.BehaviorMatchings import MAPElitesBM
from dev.SSLVE.BehaviorDescriptors import EvoGymBD_Composition
from dev.SSLVE.Collectors import EvoGymCollector

# =============================================================================
# Hyperparameters
# =============================================================================
QUICK_EXPERIMENT = True  #@param {type:"boolean"}

# EvoGym
GRID_SHAPE = (5, 5)  #@param
TASK_NAME = 'Walker-v0'  #@param {type:"string"}
MAX_STEPS = 500  #@param {type:"integer"}
N_EPISODES = 1  #@param {type:"integer"}
FREQ = 2.0  #@param {type:"number"}

# Behavior Descriptor (3D): (frac_rigid, frac_soft, frac_actuator)
BIN_RANGES = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  #@param
BIN_SIZES = [10, 10, 10]  #@param

# MAP-Elites
TOP_K = 3  #@param {type:"integer"}
N_SAMPLES = 200  #@param {type:"integer"}
MUTATION_SIGMA = 0.3  #@param {type:"number"}
N_STEPS = 100  #@param {type:"integer"}

# General
SEED = 42  #@param {type:"integer"}
OUTPUT_DIR = "results/evogym_map_v1"  #@param {type:"string"}

if QUICK_EXPERIMENT:
    MAX_STEPS = 300
    N_SAMPLES = 20
    N_STEPS = 20

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Setup
# =============================================================================
N_VOXELS = GRID_SHAPE[0] * GRID_SHAPE[1]

fitness_fn = lambda info: -float(np.mean(info['displacement']))

init_fn = lambda: np.random.uniform(-1, 4, N_VOXELS)

collector = EvoGymCollector(
    task_name=TASK_NAME, max_steps=MAX_STEPS,
    n_episodes=N_EPISODES, seed=SEED, freq=FREQ,
)
bd = EvoGymBD_Composition(bin_ranges=BIN_RANGES, bin_sizes=BIN_SIZES)
bm = MAPElitesBM(behavior_descriptor=bd, fitness_fn=fitness_fn, top_k=TOP_K)

sp = UniBinUniMemPSE(
    agent_class=EvoGymAgent,
    architecture=GRID_SHAPE,
    agent_kwargs={'freq': FREQ},
    mutation_sigma=MUTATION_SIGMA,
    n_samples=N_SAMPLES,
    init_fn=init_fn,
)

me = MAPElite(search_phase=sp, collector=collector, behavior_matching=bm)

# =============================================================================
# Run
# =============================================================================
print(f"Grid shape: {GRID_SHAPE} ({N_VOXELS} voxels)")
print(f"Task: {TASK_NAME}")
print(f"Bins: {BIN_SIZES} (total={bd.total_bins()})")
print(f"TOP_K={TOP_K}, N_SAMPLES={N_SAMPLES}, N_STEPS={N_STEPS}")
print(f"Quick mode: {QUICK_EXPERIMENT}")
print(f"Output dir: {OUTPUT_DIR}")
print()

t0 = time.perf_counter()
me.run(n_steps=N_STEPS)
elapsed = time.perf_counter() - t0
print(f"\nTotal runtime: {elapsed / 60.0:.2f} min ({elapsed:.1f} sec)")

# =============================================================================
# Results
# =============================================================================
f_min, f_mean, f_max = bm.fitness_stats()
print(f"\nFinal archive size: {bm.archive_size()}")
print(f"Final coverage: {bm.coverage():.4f}")
print(f"Best fitness (lower is better): {f_min:.4f}")
print(f"Best displacement: {-f_min:.4f}")

# =============================================================================
# Plot
# =============================================================================
main_plot_path = os.path.join(OUTPUT_DIR, "map_history.png")
me.plot_history(save_path=main_plot_path)
print(f"Saved plot: {main_plot_path}")
