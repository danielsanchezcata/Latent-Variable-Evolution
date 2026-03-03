#@title EvoGym Walker MAP-Elites Baseline
#@markdown Evolves soft robot morphologies on Walker-v0 using MAP-Elites.
#@markdown BD: material composition ratios. Fitness: horizontal displacement.
#@markdown Open-loop sinusoidal controller (no neural network).

# =============================================================================
# Install (Colab only)
# =============================================================================
# !pip install evogym

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import gymnasium as gym

# =============================================================================
# Hyperparameters
# =============================================================================
QUICK_EXPERIMENT = True  #@param {type:"boolean"}

GRID_SHAPE = (5, 5)  #@param
TASK_NAME = 'Walker-v0'  #@param {type:"string"}
MAX_STEPS = 500  #@param {type:"integer"}
N_EPISODES = 1  #@param {type:"integer"}
FREQ = 2.0  #@param {type:"number"}

BIN_RANGES = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  #@param
BIN_SIZES = [10, 10, 10]  #@param

TOP_K = 3  #@param {type:"integer"}
N_SAMPLES = 200  #@param {type:"integer"}
MUTATION_SIGMA = 0.3  #@param {type:"number"}
N_STEPS = 100  #@param {type:"integer"}

SEED = 42  #@param {type:"integer"}

if QUICK_EXPERIMENT:
    MAX_STEPS = 300
    N_SAMPLES = 20
    N_STEPS = 20

random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Agent Module
# =============================================================================
class EvoGymAgent:
    """
    Agent for EvoGym morphology evolution.
    Genome is a continuous vector that maps to a 2D voxel grid.
    Uses open-loop sinusoidal actuation (no neural network).
    """
    EMPTY = 0
    RIGID = 1
    SOFT = 2
    H_ACT = 3
    V_ACT = 4

    def __init__(self, architecture, **kwargs):
        self.grid_shape = architecture
        self.n_voxels = architecture[0] * architecture[1]
        self.raw_genome = None
        self.freq = kwargs.get('freq', 2.0)

    def get_weight_dim(self):
        return self.n_voxels

    def set_weights(self, flat_weights):
        self.raw_genome = flat_weights

    def _continuous_to_material(self, val):
        if val < -0.5:
            return self.EMPTY
        elif val < 0.5:
            return self.SOFT
        elif val < 1.5:
            return self.RIGID
        elif val < 2.5:
            return self.H_ACT
        else:
            return self.V_ACT

    def _largest_connected_component(self, body):
        rows, cols = body.shape
        visited = np.zeros_like(body, dtype=bool)
        components = []
        for r in range(rows):
            for c in range(cols):
                if body[r, c] != self.EMPTY and not visited[r, c]:
                    component = []
                    queue = [(r, c)]
                    visited[r, c] = True
                    while queue:
                        cr, cc = queue.pop(0)
                        component.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < rows and 0 <= nc < cols
                                    and not visited[nr, nc]
                                    and body[nr, nc] != self.EMPTY):
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                    components.append(component)
        if not components:
            return body
        largest = max(components, key=len)
        largest_set = set(largest)
        result = np.full_like(body, self.EMPTY)
        for r, c in largest_set:
            result[r, c] = body[r, c]
        return result

    def decode_body(self):
        raw = np.array([self._continuous_to_material(v) for v in self.raw_genome])
        body = raw.reshape(self.grid_shape)
        body = self._largest_connected_component(body)
        connections = np.zeros_like(body)
        rows, cols = body.shape
        for r in range(rows):
            for c in range(cols):
                if body[r, c] != self.EMPTY:
                    connections[r, c] = 1
        has_actuator = np.any((body == self.H_ACT) | (body == self.V_ACT))
        has_voxel = np.any(body != self.EMPTY)
        is_valid = bool(has_actuator and has_voxel)
        return body, connections, is_valid

    def get_actuator_phases(self, body):
        rows, cols = body.shape
        phases = []
        for r in range(rows):
            for c in range(cols):
                if body[r, c] in (self.H_ACT, self.V_ACT):
                    phase = (r / max(rows - 1, 1)) * np.pi
                    phases.append(phase)
        return phases


# =============================================================================
# Collector
# =============================================================================
class EvoGymCollector:
    """Collector for EvoGym soft robot locomotion with open-loop sinusoidal controller."""

    def __init__(self, task_name='Walker-v0', max_steps=500, n_episodes=1,
                 seed=None, freq=2.0):
        self.task_name = task_name
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed = seed
        self.freq = freq

    def collect(self, agent):
        import evogym.envs  # noqa: F401

        body, connections, is_valid = agent.decode_body()

        n_rigid = int(np.sum(body == 1))
        n_soft = int(np.sum(body == 2))
        n_h_act = int(np.sum(body == 3))
        n_v_act = int(np.sum(body == 4))
        n_empty = int(np.sum(body == 0))
        n_filled = n_rigid + n_soft + n_h_act + n_v_act

        material_info = {
            'n_rigid': n_rigid, 'n_soft': n_soft,
            'n_h_act': n_h_act, 'n_v_act': n_v_act,
            'n_empty': n_empty, 'n_filled': n_filled,
        }

        if not is_valid:
            return {
                'reward': [0.0] * self.n_episodes,
                'displacement': [0.0] * self.n_episodes,
                'steps': [0] * self.n_episodes,
                **material_info,
            }

        phases = agent.get_actuator_phases(body)
        n_actuators = len(phases)

        all_rewards = []
        all_displacements = []
        all_steps = []

        for ep in range(self.n_episodes):
            seed = self.seed + ep if self.seed is not None else None
            try:
                env = gym.make(self.task_name, body=body, connections=connections,
                               render_mode=None)
                obs, _ = env.reset(seed=seed)
            except Exception:
                all_rewards.append(0.0)
                all_displacements.append(0.0)
                all_steps.append(0)
                continue

            total_reward = 0.0
            initial_x = env.unwrapped.get_pos_com_obs("robot")[0]

            for t in range(self.max_steps):
                action = np.array([
                    0.6 + 0.5 * (1.0 + np.sin(
                        2.0 * np.pi * self.freq * t / 40.0 + phases[i]
                    ))
                    for i in range(n_actuators)
                ])
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            final_x = env.unwrapped.get_pos_com_obs("robot")[0]
            displacement = float(final_x - initial_x)
            env.close()

            all_rewards.append(float(total_reward))
            all_displacements.append(displacement)
            all_steps.append(t + 1)

        return {
            'reward': all_rewards,
            'displacement': all_displacements,
            'steps': all_steps,
            **material_info,
        }


# =============================================================================
# Behavior Descriptor
# =============================================================================
class EvoGymBD_Composition:
    """3D BD: (fraction_rigid, fraction_soft, fraction_actuator) among non-empty voxels."""

    def __init__(self, bin_ranges=None, bin_sizes=None):
        if bin_ranges is None:
            bin_ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        if bin_sizes is None:
            bin_sizes = [10, 10, 10]
        self.bin_ranges = bin_ranges
        self.bin_sizes = bin_sizes

    def describe(self, info):
        n_filled = info['n_filled']
        if n_filled == 0:
            return (0.0, 0.0, 0.0)
        frac_rigid = info['n_rigid'] / n_filled
        frac_soft = info['n_soft'] / n_filled
        frac_act = (info['n_h_act'] + info['n_v_act']) / n_filled
        return (frac_rigid, frac_soft, frac_act)

    def discretize(self, descriptor):
        bin_id = []
        for val, (lo, hi), n_bins in zip(descriptor, self.bin_ranges, self.bin_sizes):
            clamped = np.clip(val, lo, hi)
            idx = int((clamped - lo) / (hi - lo) * n_bins)
            idx = min(idx, n_bins - 1)
            bin_id.append(idx)
        return tuple(bin_id)

    def total_bins(self):
        result = 1
        for n in self.bin_sizes:
            result *= n
        return result


# =============================================================================
# Behavior Matching (MAP-Elites archive)
# =============================================================================
class MAPElitesBM:
    """MAP-Elites archive with top-k per bin."""

    def __init__(self, behavior_descriptor, fitness_fn, top_k=10):
        self.behavior_descriptor = behavior_descriptor
        self.fitness_fn = fitness_fn
        self.top_k = top_k
        self.bins = {}
        self.dataset = []
        self.fitnesses = []
        self.bin_ids = []
        self.bins_idx = {}

    def update(self, thetas, infos):
        for theta, info in zip(thetas, infos):
            descriptor = self.behavior_descriptor.describe(info)
            bin_id = self.behavior_descriptor.discretize(descriptor)
            fitness = self.fitness_fn(info)
            if bin_id not in self.bins:
                self.bins[bin_id] = []
            self.bins[bin_id].append((theta, fitness))
            if len(self.bins[bin_id]) > self.top_k:
                self.bins[bin_id].sort(key=lambda x: x[1])
                self.bins[bin_id] = self.bins[bin_id][:self.top_k]
        self._rebuild()

    def _rebuild(self):
        self.dataset = []
        self.fitnesses = []
        self.bin_ids = []
        self.bins_idx = {}
        for bin_id, entries in self.bins.items():
            self.bins_idx[bin_id] = []
            for theta, fitness in entries:
                idx = len(self.dataset)
                self.dataset.append(theta)
                self.fitnesses.append(fitness)
                self.bin_ids.append(bin_id)
                self.bins_idx[bin_id].append(idx)

    def coverage(self):
        return len(self.bins) / self.behavior_descriptor.total_bins()

    def archive_size(self):
        return len(self.dataset)

    def fitness_stats(self):
        if not self.fitnesses:
            return 0.0, 0.0, 0.0
        f = np.array(self.fitnesses)
        return float(f.min()), float(f.mean()), float(f.max())


# =============================================================================
# Search Phase (Parameter Space Evolution)
# =============================================================================
class UniBinUniMemPSE:
    """Uniform bin, uniform member, Gaussian noise in parameter space."""

    def __init__(self, agent_class, architecture, agent_kwargs=None,
                 mutation_sigma=0.1, n_samples=50, init_fn=None):
        self.agent_class = agent_class
        self.architecture = architecture
        self.agent_kwargs = agent_kwargs or {}
        self.mutation_sigma = mutation_sigma
        self.n_samples = n_samples
        self.init_fn = init_fn

    def make_agent(self, theta):
        agent = self.agent_class(self.architecture, **self.agent_kwargs)
        agent.set_weights(theta)
        return agent

    def _init(self):
        if self.init_fn is not None:
            return self.init_fn()
        return np.random.uniform(-1, 4, self.architecture[0] * self.architecture[1])

    def sample(self, behavior_matching=None, **kwargs):
        if behavior_matching is None or len(behavior_matching.bins) == 0:
            return [self._init() for _ in range(self.n_samples)]
        bin_ids = list(behavior_matching.bins_idx.keys())
        candidates = []
        for _ in range(self.n_samples):
            bid = bin_ids[np.random.randint(len(bin_ids))]
            members = behavior_matching.bins_idx[bid]
            idx = members[np.random.randint(len(members))]
            parent = behavior_matching.dataset[idx]
            child = parent + self.mutation_sigma * np.random.randn(len(parent))
            candidates.append(child)
        return candidates


# =============================================================================
# MAP-Elites Orchestrator
# =============================================================================
class MAPElite:
    """MAP-Elites without latent module."""

    def __init__(self, search_phase, collector, behavior_matching):
        self.SP = search_phase
        self.CO = collector
        self.BM = behavior_matching
        self.history = {
            'fitness_min': [], 'fitness_mean': [], 'fitness_max': [],
            'coverage': [], 'archive_size': [],
        }

    def step(self):
        thetas = self.SP.sample(behavior_matching=self.BM)
        infos = []
        for theta in thetas:
            agent = self.SP.make_agent(theta)
            info = self.CO.collect(agent)
            infos.append(info)
        self.BM.update(thetas, infos)
        f_min, f_mean, f_max = self.BM.fitness_stats()
        self.history['fitness_min'].append(f_min)
        self.history['fitness_mean'].append(f_mean)
        self.history['fitness_max'].append(f_max)
        self.history['coverage'].append(self.BM.coverage())
        self.history['archive_size'].append(self.BM.archive_size())
        print(f"Archive: {self.BM.archive_size()}, "
              f"Bins: {len(self.BM.bins)}, "
              f"Coverage: {self.BM.coverage():.4f}, "
              f"Fitness min/mean/max: {f_min:.2f}/{f_mean:.2f}/{f_max:.2f}")

    def run(self, n_steps):
        for t in range(n_steps):
            print(f"\n--- MAP-Elite Step {t+1}/{n_steps} ---")
            self.step()

    def plot_history(self, save_path=None):
        if not self.history['fitness_min']:
            print("No history available.")
            return
        steps = range(len(self.history['fitness_min']))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax = axes[0]
        ax.plot(steps, self.history['fitness_min'], label='Min')
        ax.plot(steps, self.history['fitness_mean'], label='Mean')
        ax.plot(steps, self.history['fitness_max'], label='Max')
        ax.set_xlabel('Step'); ax.set_ylabel('Fitness')
        ax.set_title('Fitness over Steps'); ax.legend()
        ax = axes[1]
        ax.plot(steps, self.history['coverage'])
        ax.set_xlabel('Step'); ax.set_ylabel('Coverage')
        ax.set_title('Behavior Coverage')
        ax = axes[2]
        ax.plot(steps, self.history['archive_size'])
        ax.set_xlabel('Step'); ax.set_ylabel('Size')
        ax.set_title('Archive Size')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


# =============================================================================
# Setup
# =============================================================================
fitness_fn = lambda info: -float(np.mean(info['displacement']))

init_fn = lambda: np.random.uniform(-1, 4, GRID_SHAPE[0] * GRID_SHAPE[1])

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
print(f"Grid shape: {GRID_SHAPE}")
print(f"Task: {TASK_NAME}")
print(f"Bins: {BIN_SIZES} (total={bd.total_bins()})")
print(f"TOP_K={TOP_K}, N_SAMPLES={N_SAMPLES}, N_STEPS={N_STEPS}")
print(f"Quick mode: {QUICK_EXPERIMENT}")
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
me.plot_history()
