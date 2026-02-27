import numpy as np
import random
import torch
import time
import os
import matplotlib.pyplot as plt

from dev.SSLVE.Main import MAPElite
from dev.SSLVE.SearchPhases import UniBinUniMemPSE
from dev.SSLVE.AgentModules import MLP_Agent
from dev.SSLVE.BehaviorMatchings import MAPElitesBM
from dev.SSLVE.BehaviorDescriptors import CarRacingBD_v1
from dev.SSLVE.Collectors import CarRacingCollector

# =============================================================================
# Hyperparameters
# =============================================================================
# Execution mode
# QUICK_EXPERIMENT=True is recommended first in Colab.
QUICK_EXPERIMENT = False  # @param {type:"boolean"}

# CarRacing
MAX_STEPS = 600  # @param {type:"integer"}
N_EPISODES = 1  # @param {type:"integer"}

# Agent Architecture
# State = [speed, angular_vel, cos(head_err), sin(head_err),
#          centerline_dist, prev_steer, prev_gas, prev_brake]
ARCHITECTURE = [8, 64, 64, 3]  # @param
OUTPUT_ACTIVATION = 'car_racing'  # @param {type:"string"}

# Behavior Descriptor
BIN_RANGES = [(0.0, 1.0), (0.0, 0.8), (0.0, 45.0)]  # @param
BIN_SIZES = [32, 32, 24]  # @param

# MAP-Elites
TOP_K = 3  # @param {type:"integer"}
N_SAMPLES = 256  # @param {type:"integer"}
MUTATION_SIGMA = 0.45  # @param {type:"number"}
N_STEPS = 40  # @param {type:"integer"}

# Fitness (to minimize)
OFFTRACK_PENALTY = 400.0  # @param {type:"number"}
TRACK_LIMIT_PENALTY = 5.0  # @param {type:"number"}
INCOMPLETE_LAP_PENALTY = 600.0  # @param {type:"number"}
REWARD_WEIGHT = 1.0  # @param {type:"number"}

# General
SEED = 42  # @param {type:"integer"}
OUTPUT_DIR = "results/carracing_map_v1"  # @param {type:"string"}

if QUICK_EXPERIMENT:
    MAX_STEPS = 300
    N_SAMPLES = 96
    N_STEPS = 12

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Setup
# =============================================================================
def fitness_fn(info):
    mean_reward = float(np.mean(info['reward']))
    mean_steps = float(np.mean(info['steps']))
    mean_offtrack_ratio = float(np.mean(info['offtrack_ratio']))
    mean_track_viol = float(np.mean(info['track_limit_violations']))
    mean_completion = float(np.mean(info['lap_completion']))

    return (
        -REWARD_WEIGHT * mean_reward
        + mean_steps
        + OFFTRACK_PENALTY * mean_offtrack_ratio
        + TRACK_LIMIT_PENALTY * mean_track_viol
        + INCOMPLETE_LAP_PENALTY * (1.0 - mean_completion)
    )


def plot_map_training_summary(history, save_path):
    """Additional MAP training plot with best-so-far fitness and coverage."""
    if not history['fitness_min']:
        return

    steps = np.arange(len(history['fitness_min']))
    best_so_far = np.minimum.accumulate(np.array(history['fitness_min']))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(steps, history['fitness_min'], label='Per-step best')
    ax.plot(steps, best_so_far, label='Best-so-far', linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Fitness (lower better)')
    ax.set_title('MAP Fitness Progress')
    ax.legend()

    ax = axes[1]
    ax.plot(steps, history['coverage'], label='Coverage')
    ax.plot(steps, history['archive_size'], label='Archive size')
    ax.set_xlabel('Step')
    ax.set_title('MAP Exploration Progress')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


collector = CarRacingCollector(max_steps=MAX_STEPS, n_episodes=N_EPISODES, seed=SEED)
bd = CarRacingBD_v1(
    bin_ranges=BIN_RANGES,
    bin_sizes=BIN_SIZES,
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
print(f"Bins: {BIN_SIZES} (total={bd.total_bins()})")
print(f"TOP_K={TOP_K}")
print(f"Mutation sigma={MUTATION_SIGMA}")
print(f"Quick mode: {QUICK_EXPERIMENT}")
print(f"MAX_STEPS={MAX_STEPS}, N_SAMPLES={N_SAMPLES}, N_STEPS={N_STEPS}")
print("Note: MAP baseline rollout/evaluation is CPU-bound (Box2D); GPU has little effect here.")
print(f"Plot output dir: {OUTPUT_DIR}")
max_rollout_steps = N_STEPS * N_SAMPLES * N_EPISODES * MAX_STEPS
print(f"Max rollout steps budget: {max_rollout_steps}")
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
print(f"Best fitness (lower is better): {f_min:.2f}")

# =============================================================================
# Plot
# =============================================================================
main_plot_path = os.path.join(OUTPUT_DIR, "map_history.png")
extra_plot_path = os.path.join(OUTPUT_DIR, "map_training_summary.png")
me.plot_history(save_path=main_plot_path)
plot_map_training_summary(me.history, extra_plot_path)
print(f"Saved plot: {main_plot_path}")
print(f"Saved plot: {extra_plot_path}")
