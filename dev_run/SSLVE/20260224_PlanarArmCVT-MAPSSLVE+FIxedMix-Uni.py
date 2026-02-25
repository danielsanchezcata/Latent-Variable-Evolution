#@title PlanarArmCVT MAP SSLVE FixedMix
import numpy as np
import random
import torch

# =============================================================================
# Hyperparameters
# =============================================================================
N_JOINTS = 100  #@param {type:"integer"}

N_BINS = 1950  #@param {type:"integer"}
CENTERS = "Precomputed_CVT_1950"  #@param {type:"string"}
print(f"""You chose {CENTERS} as your CVT configuration. Note below:
- Precomputed_CVT_1950 ... Precomputed CVT centers at seed 42. Only N_BINS = 1950 is supported.
- CVT ... It requires new configuration of CVT. It supports any N_BINS, but it takes time. Saving the obtained bd.center for later use is highly recommended.
- random ... Quick and sufficient for quick run. However, uniformity as in CVT is not guranteed, so behavior coverage evaluation is a bit degraded. 
""")

TOP_K = 3  #@param {type:"integer"}
MUTATION_SIGMA = 0.3  #@param {type:"number"}
N_PSE = 50  #@param {type:"integer"}
N_LVE_MUTATION = 75  #@param {type:"integer"}
N_LVE_CROSSOVER = 75  #@param {type:"integer"}

LATENT_DIM = 32  #@param {type:"integer"}
HIDDEN_DIMS = [64]  #@param
BETA = 1e-3  #@param {type:"number"}
GAMMA_SSL = 1e-3  #@param {type:"number"}
EPOCHS = 100  #@param {type:"integer"}
BATCH_SIZE = 256  #@param {type:"integer"}
LR = 1e-3  #@param {type:"number"}

N_STEPS = 20  #@param {type:"integer"}

SEED = 42  #@param {type:"integer"}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# Setup
# =============================================================================
fitness_fn = lambda info: info['angle_variance']

init_fn = lambda: np.random.uniform(-np.pi, np.pi, N_JOINTS)

collector = PlanarArmCollector(n_joints=N_JOINTS)
bd = PlanarArmBD_CVT(n_bins=N_BINS, centers=CENTERS)
bm = MAPElitesBM(behavior_descriptor=bd, fitness_fn=fitness_fn, top_k=TOP_K)

lm = BetaVAE_SSLVE(
    input_dim=N_JOINTS,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS,
    beta=BETA,
    gamma_ssl=GAMMA_SSL,
)

sp = UniBinUniMemFixedMix(
    agent_class=PlanarArmAgent,
    architecture=N_JOINTS,
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
print(f"N joints: {N_JOINTS}")
print(f"Latent dim: {LATENT_DIM}")
print(f"Device: {DEVICE}")
print(f"N bins: {N_BINS} ({CENTERS})")
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
print(f"Best fitness (angle variance): {f_min:.4f}")

# =============================================================================
# Plot
# =============================================================================
sslve.plot_history()
