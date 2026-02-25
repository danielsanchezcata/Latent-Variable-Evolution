import numpy as np
import random
import torch

from dev.SSLVE.Main import MAPElite
from dev.SSLVE.SearchPhases import UniBinUniMemPSE
from dev.SSLVE.AgentModules import MLP_Agent
from dev.SSLVE.BehaviorMatchings import MAPElitesBM
from dev.SSLVE.BehaviorDescriptors import CarRacingBD_v1
from dev.SSLVE.Collectors import CarRacingCollector

# =============================================================================
# Hyperparameters
# =============================================================================
# CarRacing
MAX_STEPS = 1000  # @param {type:"integer"}
N_EPISODES = 1  # @param {type:"integer"}

# Agent Architecture
# State = [speed, angular_vel, cos(head_err), sin(head_err),
#          centerline_dist, prev_steer, prev_gas, prev_brake]
ARCHITECTURE = [8, 64, 64, 3]  # @param
OUTPUT_ACTIVATION = 'car_racing'  # @param {type:"string"}

# Behavior Descriptor
BIN_RANGES = [(0.0, 1.0), (0.0, 0.6), (0.0, 1.0)]  # @param
BIN_SIZES = [12, 12, 12]  # @param
FULL_THROTTLE_THRESHOLD = 0.95  # @param {type:"number"}

# MAP-Elites
TOP_K = 3  # @param {type:"integer"}
N_SAMPLES = 120  # @param {type:"integer"}
MUTATION_SIGMA = 0.25  # @param {type:"number"}
N_STEPS = 20  # @param {type:"integer"}

# Fitness (to minimize)
OFFTRACK_PENALTY = 400.0  # @param {type:"number"}
TRACK_LIMIT_PENALTY = 5.0  # @param {type:"number"}
INCOMPLETE_LAP_PENALTY = 600.0  # @param {type:"number"}

# General
SEED = 42  # @param {type:"integer"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# Setup
# =============================================================================
def fitness_fn(info):
    mean_steps = float(np.mean(info['steps']))
    mean_offtrack_ratio = float(np.mean(info['offtrack_ratio']))
    mean_track_viol = float(np.mean(info['track_limit_violations']))
    mean_completion = float(np.mean(info['lap_completion']))

    return (
        mean_steps
        + OFFTRACK_PENALTY * mean_offtrack_ratio
        + TRACK_LIMIT_PENALTY * mean_track_viol
        + INCOMPLETE_LAP_PENALTY * (1.0 - mean_completion)
    )


collector = CarRacingCollector(max_steps=MAX_STEPS, n_episodes=N_EPISODES, seed=SEED)
bd = CarRacingBD_v1(
    bin_ranges=BIN_RANGES,
    bin_sizes=BIN_SIZES,
    full_throttle_threshold=FULL_THROTTLE_THRESHOLD,
)
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
agent_tmp = MLP_Agent(ARCHITECTURE, output_activation=OUTPUT_ACTIVATION)
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
print(f"Best fitness (lower is better): {f_min:.2f}")

# =============================================================================
# Plot
# =============================================================================
me.plot_history()
