#@title PlanarArmCVT MAP-Elite PSE
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
fitness_fn = lambda info: info['angle_variance']

init_fn = lambda: np.random.uniform(-np.pi, np.pi, N_JOINTS)

collector = PlanarArmCollector(n_joints=N_JOINTS)
bd = PlanarArmBD_CVT(n_bins=N_BINS, centers=CENTERS)
bm = MAPElitesBM(behavior_descriptor=bd, fitness_fn=fitness_fn, top_k=TOP_K)

sp = UniBinUniMemPSE(
    agent_class=PlanarArmAgent,
    architecture=N_JOINTS,
    mutation_sigma=MUTATION_SIGMA,
    n_samples=N_SAMPLES,
    init_fn=init_fn,
)

me = MAPElite(
    search_phase=sp,
    collector=collector,
    behavior_matching=bm,
)

# =============================================================================
# Run
# =============================================================================
print(f"N joints: {N_JOINTS}")
print(f"N bins: {N_BINS} ({CENTERS})")
print()

me.run(n_steps=N_STEPS)

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
me.plot_history()
