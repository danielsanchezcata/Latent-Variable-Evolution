import numpy as np
import random
import torch

# =============================================================================
# Hyperparameters
# =============================================================================
# BipedalWalker
MAX_STEPS = 200  #@param {type:"integer"}
N_EPISODES = 1  #@param {type:"integer"}

# Agent Architecture
ARCHITECTURE = [24, 64, 64, 4]  #@param
OUTPUT_ACTIVATION = 'tanh'  #@param {type:"string"}

# Behavior Descriptor
BIN_RANGES = [(0.0, 1.0), (0.0, 1.0), (-1.5, 1.5)]  #@param
BIN_SIZES = [12, 12, 12]  #@param

# MAP-Elites
TOP_K = 3  #@param {type:"integer"}

# CMA-ME Improvement Emitter
N_EMITTERS = 5  #@param {type:"integer"}
N_GENERATIONS = 10  #@param {type:"integer"}
SIGMA_INIT = 0.5  #@param {type:"number"}
LAMBDA_ = 20  #@param {type:"integer"}
N_OUTPUT = 20  #@param {type:"integer"}
N_INIT_SAMPLES = 200  #@param {type:"integer"}
SEPARABLE = True  #@param {type:"boolean"}

# MAP-Elite
N_STEPS = 100  #@param {type:"integer"}

# General
SEED = 42  #@param {type:"integer"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# Setup
# =============================================================================
fitness_fn = lambda info: -np.mean(info['reward'])

collector = BipedalWalkerCollector(max_steps=MAX_STEPS, n_episodes=N_EPISODES, seed=SEED)
bd = BipedalWalkerBD_v1(bin_ranges=BIN_RANGES, bin_sizes=BIN_SIZES)
bm = MAPElitesBM(behavior_descriptor=bd, fitness_fn=fitness_fn, top_k=TOP_K)

sp = UniBinUniMemCMAMEimpPSE(
    agent_class=MLP_Agent,
    architecture=ARCHITECTURE,
    agent_kwargs={'output_activation': OUTPUT_ACTIVATION},
    n_emitters=N_EMITTERS,
    n_generations=N_GENERATIONS,
    sigma_init=SIGMA_INIT,
    lambda_=LAMBDA_,
    n_output=N_OUTPUT,
    n_init_samples=N_INIT_SAMPLES,
    separable=SEPARABLE,
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
print(f"Emitters: {N_EMITTERS}, Generations: {N_GENERATIONS}, Lambda: {LAMBDA_}")
print(f"Internal evals per step: {N_EMITTERS * N_GENERATIONS * LAMBDA_}")
print(f"Output samples per step: {N_EMITTERS * N_OUTPUT}")
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

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Find best theta
best_idx = int(np.argmin(bm.fitnesses))
best_theta = bm.dataset[best_idx]
agent = sp.make_agent(best_theta)
print(f"Best fitness: {bm.fitnesses[best_idx]:.2f}, Reward: {-bm.fitnesses[best_idx]:.2f}")

# Render
env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
obs = env.reset(seed=42)
if isinstance(obs, tuple):
    obs = obs[0]
frames = [env.render()]
for _ in range(300):
    action = agent.act(obs)
    step_result = env.step(action)
    obs, reward, terminated, truncated = step_result[0], step_result[1], step_result[2], step_result[3]
    frames.append(env.render())
    if terminated or truncated:
        break
env.close()

# Animate
fig, ax = plt.subplots()
ax.axis('off')
im = ax.imshow(frames[0])
def update(i):
    im.set_data(frames[i])
    return [im]
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)
plt.close()
HTML(ani.to_jshtml())
