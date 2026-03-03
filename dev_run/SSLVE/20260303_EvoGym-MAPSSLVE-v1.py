#@title EvoGym Walker MAP-SSLVE
#@markdown Evolves soft robot morphologies on Walker-v0 using SSLVE.
#@markdown BD: material composition ratios. Fitness: horizontal displacement.
#@markdown Open-loop sinusoidal controller (no neural network).

import numpy as np
import random
import time
import os
import torch

from dev.SSLVE.Main import SSLVE
from dev.SSLVE.SearchPhases import UniBinUniMemFixedMix
from dev.SSLVE.AgentModules import EvoGymAgent
from dev.SSLVE.BehaviorMatchings import MAPElitesBM
from dev.SSLVE.BehaviorDescriptors import EvoGymBD_Composition
from dev.SSLVE.Collectors import EvoGymCollector
from dev.SSLVE.LatentModules import BetaVAE_SSLVE

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
MUTATION_SIGMA = 0.3  #@param {type:"number"}
N_PSE = 50  #@param {type:"integer"}
N_LVE_MUTATION = 75  #@param {type:"integer"}
N_LVE_CROSSOVER = 75  #@param {type:"integer"}

# Latent Module
LATENT_DIM = 8  #@param {type:"integer"}
HIDDEN_DIMS = [32, 16]  #@param
BETA = 1e-3  #@param {type:"number"}
GAMMA_SSL = 1e-3  #@param {type:"number"}
EPOCHS = 100  #@param {type:"integer"}
BATCH_SIZE = 128  #@param {type:"integer"}
LR = 1e-3  #@param {type:"number"}

# SSLVE
N_STEPS = 20  #@param {type:"integer"}

# General
SEED = 42  #@param {type:"integer"}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "results/evogym_mapsslve_v1"  #@param {type:"string"}

if QUICK_EXPERIMENT:
    MAX_STEPS = 300
    N_PSE = 10
    N_LVE_MUTATION = 10
    N_LVE_CROSSOVER = 10
    EPOCHS = 50
    BATCH_SIZE = 64
    N_STEPS = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
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

lm = BetaVAE_SSLVE(
    input_dim=N_VOXELS,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS,
    beta=BETA,
    gamma_ssl=GAMMA_SSL,
)

sp = UniBinUniMemFixedMix(
    agent_class=EvoGymAgent,
    architecture=GRID_SHAPE,
    agent_kwargs={'freq': FREQ},
    mutation_sigma=MUTATION_SIGMA,
    n_pse=N_PSE,
    n_lve_mutation=N_LVE_MUTATION,
    n_lve_crossover=N_LVE_CROSSOVER,
    init_fn=init_fn,
)

sslve = SSLVE(
    search_phase=sp,
    collector=collector,
    behavior_matching=bm,
    latent_module=lm,
    device=DEVICE,
)

# =============================================================================
# Run
# =============================================================================
print(f"Grid shape: {GRID_SHAPE} ({N_VOXELS} voxels)")
print(f"Task: {TASK_NAME}")
print(f"Latent dim: {LATENT_DIM}")
print(f"Device: {DEVICE}")
print(f"Bins: {BIN_SIZES} (total={bd.total_bins()})")
print(f"Samples per step: {N_PSE} PSE + {N_LVE_MUTATION} LVE mut + {N_LVE_CROSSOVER} LVE xo = {N_PSE + N_LVE_MUTATION + N_LVE_CROSSOVER}")
print(f"Quick mode: {QUICK_EXPERIMENT}")
print(f"Output dir: {OUTPUT_DIR}")
print()

train_kwargs = {
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'verbose': True,
}

t0 = time.perf_counter()
histories = sslve.run(n_steps=N_STEPS, train_kwargs=train_kwargs)
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
main_plot_path = os.path.join(OUTPUT_DIR, "sslve_history.png")
sslve.plot_history(save_path=main_plot_path)
print(f"Saved plot: {main_plot_path}")
