import numpy as np
import random
import torch

from dev.SSLVE.Main import SSLVE
from dev.SSLVE.SearchPhases import UniBinUniMemLVE
from dev.SSLVE.AgentModules import MLP_Agent
from dev.SSLVE.BehaviorMatchings import MAPElitesBM
from dev.SSLVE.BehaviorDescriptors import CarRacingBD_v1
from dev.SSLVE.Collectors import CarRacingCollector
from dev.SSLVE.LatentModules import BetaVAE_SSLVE

# =============================================================================
# Hyperparameters
# =============================================================================
# Execution mode
# QUICK_EXPERIMENT=True is recommended first in Colab to verify everything runs.
QUICK_EXPERIMENT = True  # @param {type:"boolean"}
REQUIRE_GPU = True  # @param {type:"boolean"}

# CarRacing
MAX_STEPS = 1000  # @param {type:"integer"}
N_EPISODES = 1  # @param {type:"integer"}

# Agent Architecture
# State = [speed, angular_vel, cos(head_err), sin(head_err),
#          centerline_dist, prev_steer, prev_gas, prev_brake]
ARCHITECTURE = [8, 64, 64, 3]  # @param
OUTPUT_ACTIVATION = 'car_racing'  # @param {type:"string"}

# Behavior Descriptor
BIN_RANGES = [(0.0, 1.0), (0.0, 0.6), (0.0, 40.0)]  # @param
BIN_SIZES = [24, 24, 24]  # @param

# MAP-Elites
TOP_K = 10  # @param {type:"integer"}
N_SAMPLES = 192  # @param {type:"integer"}
MUTATION_SIGMA = 0.25  # @param {type:"number"}

# Fitness (to minimize)
OFFTRACK_PENALTY = 400.0  # @param {type:"number"}
TRACK_LIMIT_PENALTY = 5.0  # @param {type:"number"}
INCOMPLETE_LAP_PENALTY = 600.0  # @param {type:"number"}
REWARD_WEIGHT = 1.0  # @param {type:"number"}

# Latent Module
LATENT_DIM = 64  # @param {type:"integer"}
HIDDEN_DIMS = [256, 128]  # @param
BETA = 1e-2  # @param {type:"number"}
GAMMA_SSL = 1e-3  # @param {type:"number"}
EPOCHS = 80  # @param {type:"integer"}
BATCH_SIZE = 256  # @param {type:"integer"}
LR = 1e-3  # @param {type:"number"}

# SSLVE
N_STEPS = 20  # @param {type:"integer"}

# General
SEED = 42  # @param {type:"integer"}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not detected. In Colab set Runtime -> Change runtime type -> GPU, then rerun."
    )

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

if QUICK_EXPERIMENT:
    MAX_STEPS = 300
    N_SAMPLES = 64
    EPOCHS = 24
    BATCH_SIZE = 128
    N_STEPS = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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


collector = CarRacingCollector(max_steps=MAX_STEPS, n_episodes=N_EPISODES, seed=SEED)
bd = CarRacingBD_v1(
    bin_ranges=BIN_RANGES,
    bin_sizes=BIN_SIZES,
)
bm = MAPElitesBM(behavior_descriptor=bd, fitness_fn=fitness_fn, top_k=TOP_K)

agent_tmp = MLP_Agent(ARCHITECTURE, output_activation=OUTPUT_ACTIVATION)
weight_dim = agent_tmp.get_weight_dim()

lm = BetaVAE_SSLVE(
    input_dim=weight_dim,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS,
    beta=BETA,
    gamma_ssl=GAMMA_SSL,
)

sp = UniBinUniMemLVE(
    agent_class=MLP_Agent,
    architecture=ARCHITECTURE,
    agent_kwargs={'output_activation': OUTPUT_ACTIVATION},
    mutation_sigma=MUTATION_SIGMA,
    n_samples=N_SAMPLES,
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
print(f"Weight dim: {weight_dim}")
print(f"Latent dim: {LATENT_DIM}")
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Architecture: {ARCHITECTURE}")
print(f"Bins: {BIN_SIZES} (total={bd.total_bins()})")
print(f"TOP_K={TOP_K}")
print(f"Quick mode: {QUICK_EXPERIMENT}")
print(f"MAX_STEPS={MAX_STEPS}, N_SAMPLES={N_SAMPLES}, EPOCHS={EPOCHS}, N_STEPS={N_STEPS}")
print()

train_kwargs = {
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'verbose': True,
}

histories = sslve.run(n_steps=N_STEPS, train_kwargs=train_kwargs)

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
sslve.plot_history()
