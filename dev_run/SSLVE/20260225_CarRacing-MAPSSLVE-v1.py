import numpy as np
import random
import torch
import time
import os
import matplotlib.pyplot as plt

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
MAX_STEPS = 300  # @param {type:"integer"}
N_EPISODES = 1  # @param {type:"integer"}

# Agent Architecture
# State = [speed, angular_vel, cos(head_err), sin(head_err),
#          centerline_dist, prev_steer, prev_gas, prev_brake]
ARCHITECTURE = [8, 64, 64, 3]  # @param
OUTPUT_ACTIVATION = 'car_racing'  # @param {type:"string"}

# Behavior Descriptor
BIN_RANGES = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.8), (0.0, 45.0)]  # @param
BIN_SIZES = [25, 25, 25, 25]  # @param

# MAP-Elites
TOP_K = 3  # @param {type:"integer"}
N_SAMPLES = 200  # @param {type:"integer"}
MUTATION_SIGMA = 0.3  # @param {type:"number"}

# Fitness (to minimize)
STEP_COST = 1.0  # @param {type:"number"}
TRACK_LIMIT_PENALTY = 25.0  # @param {type:"number"}
COMPLETION_REWARD = 1200.0  # @param {type:"number"}
INCOMPLETE_LAP_PENALTY = 1200.0  # @param {type:"number"}

# Latent Module
LATENT_DIM = 64  # @param {type:"integer"}
HIDDEN_DIMS = [256, 128]  # @param
BETA = 1e-3  # @param {type:"number"}
GAMMA_SSL = 1e-3  # @param {type:"number"}
EPOCHS = 50  # @param {type:"integer"}
BATCH_SIZE = 256  # @param {type:"integer"}
LR = 1e-3  # @param {type:"number"}

# SSLVE
N_STEPS = 20  # @param {type:"integer"}

# General
SEED = 42  # @param {type:"integer"}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "results/carracing_mapsslve_v1"  # @param {type:"string"}

if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not detected. In Colab set Runtime -> Change runtime type -> GPU, then rerun."
    )

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

if QUICK_EXPERIMENT:
    MAX_STEPS = 100
    N_SAMPLES = 50
    EPOCHS = 25
    BATCH_SIZE = 128
    N_STEPS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Setup
# =============================================================================
def fitness_fn(info):
    mean_steps = float(np.mean(info['steps']))
    mean_track_viol = float(np.mean(info['track_limit_violations']))
    mean_completion = float(np.mean(info['lap_completion']))

    # Reward-shaped objective:
    # + completion reward, - step cost, - track-limit penalties, - incomplete-lap penalty.
    reward_score = (
        COMPLETION_REWARD * mean_completion
        - STEP_COST * mean_steps
        - TRACK_LIMIT_PENALTY * mean_track_viol
        - INCOMPLETE_LAP_PENALTY * (1.0 - mean_completion)
    )
    # MAPElitesBM minimizes fitness, so negate reward.
    return -reward_score


def plot_sslve_lm_step_summary(histories, save_path):
    """
    Plot final LM losses per SSLVE step.
    histories: list of dicts returned by SSLVE.run()
    """
    if not histories:
        return

    steps = []
    train_total = []
    train_recon = []
    train_kl = []
    train_ssl = []

    for i, h in enumerate(histories):
        if not h or 'train_total' not in h or len(h['train_total']) == 0:
            continue
        steps.append(i + 1)
        train_total.append(h['train_total'][-1])
        train_recon.append(h['train_recon'][-1])
        train_kl.append(h['train_kl'][-1])
        if 'train_ssl' in h and len(h['train_ssl']) > 0:
            train_ssl.append(h['train_ssl'][-1])

    if not steps:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(steps, train_total, label='Train total')
    ax.plot(steps, train_recon, label='Train recon')
    ax.plot(steps, train_kl, label='Train KL')
    if len(train_ssl) == len(steps):
        ax.plot(steps, train_ssl, label='Train SSL')
    ax.set_xlabel('SSLVE step')
    ax.set_ylabel('Loss')
    ax.set_title('LM Final Loss per SSLVE Step')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_sslve_lm_last_epoch_curve(histories, save_path):
    """Plot epoch-wise LM losses for the last SSLVE step."""
    if not histories:
        return

    last = None
    for h in reversed(histories):
        if h and 'train_total' in h and len(h['train_total']) > 0:
            last = h
            break

    if last is None:
        return

    epochs = np.arange(1, len(last['train_total']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(epochs, last['train_total'], label='Train total')
    ax.plot(epochs, last['train_recon'], label='Train recon')
    ax.plot(epochs, last['train_kl'], label='Train KL')
    if 'train_ssl' in last:
        ax.plot(epochs, last['train_ssl'], label='Train SSL')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('LM Loss Curves (Last SSLVE Step)')
    ax.legend()

    ax = axes[1]
    if 'val_total' in last and len(last['val_total']) > 0:
        val_epochs = np.arange(1, len(last['val_total']) + 1)
        ax.plot(val_epochs, last['val_total'], label='Val total')
        ax.plot(val_epochs, last['val_recon'], label='Val recon')
        ax.plot(val_epochs, last['val_kl'], label='Val KL')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('LM Validation Curves (Last SSLVE Step)')
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
print(f"Mutation sigma={MUTATION_SIGMA}")
print(f"Quick mode: {QUICK_EXPERIMENT}")
print(f"MAX_STEPS={MAX_STEPS}, N_SAMPLES={N_SAMPLES}, EPOCHS={EPOCHS}, N_STEPS={N_STEPS}")
print(f"Plot output dir: {OUTPUT_DIR}")
max_rollout_steps = N_STEPS * N_SAMPLES * N_EPISODES * MAX_STEPS
print(f"Max rollout steps budget: {max_rollout_steps}")
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
print(f"Best fitness (lower is better): {f_min:.2f}")

# =============================================================================
# Plot
# =============================================================================
main_plot_path = os.path.join(OUTPUT_DIR, "sslve_history.png")
lm_step_plot_path = os.path.join(OUTPUT_DIR, "sslve_lm_step_summary.png")
lm_epoch_plot_path = os.path.join(OUTPUT_DIR, "sslve_lm_last_step_curves.png")
sslve.plot_history(save_path=main_plot_path)
plot_sslve_lm_step_summary(histories, lm_step_plot_path)
plot_sslve_lm_last_epoch_curve(histories, lm_epoch_plot_path)
print(f"Saved plot: {main_plot_path}")
print(f"Saved plot: {lm_step_plot_path}")
print(f"Saved plot: {lm_epoch_plot_path}")
