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

TOP_K = 3  #@param {type:"integer"}
MUTATION_SIGMA = 0.3  #@param {type:"number"}
N_PSE = 75  #@param {type:"integer"}
N_LVE_MUTATION = 50  #@param {type:"integer"}
N_LVE_CROSSOVER = 75  #@param {type:"integer"}

LATENT_DIM = 32  #@param {type:"integer"}
HIDDEN_DIMS = [96]  #@param
BETA = 1e-1  #@param {type:"number"}
GAMMA_SSL = 1e-3  #@param {type:"number"}
EPOCHS = 200  #@param {type:"integer"}
BATCH_SIZE = 256  #@param {type:"integer"}
LR = 1e-3  #@param {type:"number"}

N_STEPS = 100  #@param {type:"integer"}

SEED = 42  #@param {type:"integer"}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

agent_tmp = MLP_Agent(ARCHITECTURE)
weight_dim = agent_tmp.get_weight_dim()

lm = BetaVAE_SSLVE(
    input_dim=weight_dim,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS,
    beta=BETA,
    gamma_ssl=GAMMA_SSL,
)

sp = UniBinUniMemFixedMix(
    agent_class=MLP_Agent,
    architecture=ARCHITECTURE,
    agent_kwargs={'output_activation': OUTPUT_ACTIVATION},
    mutation_sigma=MUTATION_SIGMA,
    n_pse=N_PSE,
    n_lve_mutation=N_LVE_MUTATION,
    n_lve_crossover=N_LVE_CROSSOVER,
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
print(f"Architecture: {ARCHITECTURE}")
print(f"Samples per step: {N_PSE} PSE + {N_LVE_MUTATION} LVE mut + {N_LVE_CROSSOVER} LVE xo = {N_PSE + N_LVE_MUTATION + N_LVE_CROSSOVER}")
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
print(f"Best fitness (neg reward): {f_min:.2f}")
print(f"Best reward: {-f_min:.2f}")

# =============================================================================
# Plot
# =============================================================================
sslve.plot_history()
