#@title EvoGym Walker MAP-SSLVE
#@markdown Evolves soft robot morphologies on Walker-v0 using SSLVE.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
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
MUTATION_SIGMA = 0.3  #@param {type:"number"}
N_PSE = 50  #@param {type:"integer"}
N_LVE_MUTATION = 75  #@param {type:"integer"}
N_LVE_CROSSOVER = 75  #@param {type:"integer"}

LATENT_DIM = 8  #@param {type:"integer"}
HIDDEN_DIMS = [32, 16]  #@param
BETA = 1e-3  #@param {type:"number"}
GAMMA_SSL = 1e-3  #@param {type:"number"}
EPOCHS = 100  #@param {type:"integer"}
BATCH_SIZE = 128  #@param {type:"integer"}
LR = 1e-3  #@param {type:"number"}

N_STEPS = 20  #@param {type:"integer"}

SEED = 42  #@param {type:"integer"}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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


# =============================================================================
# Agent Module
# =============================================================================
class EvoGymAgent:
    EMPTY = 0; RIGID = 1; SOFT = 2; H_ACT = 3; V_ACT = 4

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
        if val < -0.5: return self.EMPTY
        elif val < 0.5: return self.SOFT
        elif val < 1.5: return self.RIGID
        elif val < 2.5: return self.H_ACT
        else: return self.V_ACT

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
        if not components: return body
        largest = max(components, key=len)
        largest_set = set(largest)
        result = np.full_like(body, self.EMPTY)
        for r, c in largest_set: result[r, c] = body[r, c]
        return result

    def decode_body(self):
        raw = np.array([self._continuous_to_material(v) for v in self.raw_genome])
        body = raw.reshape(self.grid_shape)
        body = self._largest_connected_component(body)
        connections = np.zeros_like(body)
        for r in range(body.shape[0]):
            for c in range(body.shape[1]):
                if body[r, c] != self.EMPTY: connections[r, c] = 1
        has_actuator = np.any((body == self.H_ACT) | (body == self.V_ACT))
        has_voxel = np.any(body != self.EMPTY)
        return body, connections, bool(has_actuator and has_voxel)

    def get_actuator_phases(self, body):
        rows, cols = body.shape
        phases = []
        for r in range(rows):
            for c in range(cols):
                if body[r, c] in (self.H_ACT, self.V_ACT):
                    phases.append((r / max(rows - 1, 1)) * np.pi)
        return phases


# =============================================================================
# Collector
# =============================================================================
class EvoGymCollector:
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
        n_rigid = int(np.sum(body == 1)); n_soft = int(np.sum(body == 2))
        n_h_act = int(np.sum(body == 3)); n_v_act = int(np.sum(body == 4))
        n_empty = int(np.sum(body == 0))
        n_filled = n_rigid + n_soft + n_h_act + n_v_act
        material_info = {'n_rigid': n_rigid, 'n_soft': n_soft, 'n_h_act': n_h_act,
                         'n_v_act': n_v_act, 'n_empty': n_empty, 'n_filled': n_filled}
        if not is_valid:
            return {'reward': [0.0]*self.n_episodes, 'displacement': [0.0]*self.n_episodes,
                    'steps': [0]*self.n_episodes, **material_info}

        phases = agent.get_actuator_phases(body)
        n_actuators = len(phases)
        all_rewards, all_displacements, all_steps = [], [], []

        for ep in range(self.n_episodes):
            seed = self.seed + ep if self.seed is not None else None
            try:
                env = gym.make(self.task_name, body=body, connections=connections,
                               render_mode=None)
                obs, _ = env.reset(seed=seed)
            except Exception:
                all_rewards.append(0.0); all_displacements.append(0.0); all_steps.append(0)
                continue
            total_reward = 0.0
            initial_x = env.unwrapped.get_pos_com_obs("robot")[0]
            for t in range(self.max_steps):
                action = np.array([
                    0.6 + 0.5 * (1.0 + np.sin(2.0*np.pi*self.freq*t/40.0 + phases[i]))
                    for i in range(n_actuators)])
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated: break
            final_x = env.unwrapped.get_pos_com_obs("robot")[0]
            env.close()
            all_rewards.append(float(total_reward))
            all_displacements.append(float(final_x - initial_x))
            all_steps.append(t + 1)

        return {'reward': all_rewards, 'displacement': all_displacements,
                'steps': all_steps, **material_info}


# =============================================================================
# Behavior Descriptor
# =============================================================================
class EvoGymBD_Composition:
    def __init__(self, bin_ranges=None, bin_sizes=None):
        self.bin_ranges = bin_ranges or [(0.0,1.0),(0.0,1.0),(0.0,1.0)]
        self.bin_sizes = bin_sizes or [10,10,10]

    def describe(self, info):
        n_filled = info['n_filled']
        if n_filled == 0: return (0.0, 0.0, 0.0)
        return (info['n_rigid']/n_filled, info['n_soft']/n_filled,
                (info['n_h_act']+info['n_v_act'])/n_filled)

    def discretize(self, descriptor):
        bin_id = []
        for val, (lo, hi), n_bins in zip(descriptor, self.bin_ranges, self.bin_sizes):
            clamped = np.clip(val, lo, hi)
            idx = min(int((clamped - lo) / (hi - lo) * n_bins), n_bins - 1)
            bin_id.append(idx)
        return tuple(bin_id)

    def total_bins(self):
        r = 1
        for n in self.bin_sizes: r *= n
        return r


# =============================================================================
# Behavior Matching
# =============================================================================
class MAPElitesBM:
    def __init__(self, behavior_descriptor, fitness_fn, top_k=10):
        self.behavior_descriptor = behavior_descriptor
        self.fitness_fn = fitness_fn
        self.top_k = top_k
        self.bins = {}; self.dataset = []; self.fitnesses = []
        self.bin_ids = []; self.bins_idx = {}

    def update(self, thetas, infos):
        for theta, info in zip(thetas, infos):
            descriptor = self.behavior_descriptor.describe(info)
            bin_id = self.behavior_descriptor.discretize(descriptor)
            fitness = self.fitness_fn(info)
            if bin_id not in self.bins: self.bins[bin_id] = []
            self.bins[bin_id].append((theta, fitness))
            if len(self.bins[bin_id]) > self.top_k:
                self.bins[bin_id].sort(key=lambda x: x[1])
                self.bins[bin_id] = self.bins[bin_id][:self.top_k]
        self._rebuild()

    def _rebuild(self):
        self.dataset = []; self.fitnesses = []; self.bin_ids = []; self.bins_idx = {}
        for bin_id, entries in self.bins.items():
            self.bins_idx[bin_id] = []
            for theta, fitness in entries:
                idx = len(self.dataset)
                self.dataset.append(theta); self.fitnesses.append(fitness)
                self.bin_ids.append(bin_id); self.bins_idx[bin_id].append(idx)

    def coverage(self): return len(self.bins) / self.behavior_descriptor.total_bins()
    def archive_size(self): return len(self.dataset)
    def fitness_stats(self):
        if not self.fitnesses: return 0.0, 0.0, 0.0
        f = np.array(self.fitnesses)
        return float(f.min()), float(f.mean()), float(f.max())


# =============================================================================
# Latent Module (Beta-VAE with SSL)
# =============================================================================
class BetaVAE_SSLVE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=None, beta=1.0, gamma_ssl=1.0):
        super().__init__()
        if hidden_dims is None: hidden_dims = [32]
        self.input_dim = input_dim; self.latent_dim = latent_dim
        self.beta = beta; self.gamma_ssl = gamma_ssl
        enc_layers = []; in_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h_dim)); enc_layers.append(nn.ReLU())
            in_dim = h_dim
        self.encoder_net = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        dec_layers = []; in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h_dim)); dec_layers.append(nn.ReLU())
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder_net = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder_net(x); return self.fc_mu(h)

    def encode_dist(self, x):
        h = self.encoder_net(x)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), min=-20, max=2)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z): return self.decoder_net(z)

    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def nograd_encode_dist(self, x):
        with torch.no_grad(): return self.encode_dist(x)

    def loss(self, x, x_recon, mu=None, logvar=None, z=None, return_parts=False, **kw):
        recon = F.mse_loss(x_recon, x, reduction='mean')
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + self.beta * kl
        if return_parts: return {'total': total, 'recon': recon, 'kl': kl}
        return total

    def ssl_loss(self, mu_batch, logvar_batch, mu_neighbors, logvar_neighbors, counts):
        total_neighbors = mu_neighbors.size(0)
        if total_neighbors == 0: return torch.tensor(0.0, device=mu_batch.device)
        counts_t = torch.tensor(counts, dtype=torch.long, device=mu_batch.device)
        mu_b = torch.repeat_interleave(mu_batch, counts_t, dim=0)
        logvar_b = torch.repeat_interleave(logvar_batch, counts_t, dim=0)
        var_b = logvar_b.exp(); var_n = logvar_neighbors.exp()
        kl = 0.5*(logvar_b - logvar_neighbors + var_n/var_b + (mu_neighbors-mu_b).pow(2)/var_b - 1.0)
        kl_per = kl.mean(dim=1)
        loss = torch.tensor(0.0, device=mu_batch.device); n_active = 0; offset = 0
        for c in counts:
            if c > 0: loss = loss + kl_per[offset:offset+c].mean(); n_active += 1
            offset += c
        return loss / n_active if n_active > 0 else torch.tensor(0.0, device=mu_batch.device)

    def fit(self, dataset, bin_ids=None, bins=None, epochs=100, batch_size=32,
            lr=1e-3, device='cpu', verbose=True, val_split=0.2):
        self.to(device)
        use_ssl = self.gamma_ssl > 0 and bin_ids is not None and bins is not None
        data = torch.tensor(np.array(dataset), dtype=torch.float32)
        n = len(data)
        if n == 0: return {}
        batch_size = min(batch_size, n)
        n_val = int(n * val_split)
        if n - n_val < batch_size: n_val = 0
        n_train = n - n_val
        indices = torch.randperm(n).tolist()
        train_idx = indices[:n_train]; val_idx = indices[n_train:]
        train_loader = torch.utils.data.DataLoader(train_idx, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_idx, batch_size=batch_size, shuffle=False, drop_last=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {'train_total':[],'train_recon':[],'train_kl':[],'val_total':[],'val_recon':[],'val_kl':[]}
        if use_ssl: history['train_ssl'] = []

        for epoch in range(epochs):
            self.train()
            ep_losses = {'total':0.0,'recon':0.0,'kl':0.0}
            if use_ssl: ep_losses['ssl'] = 0.0
            for idx_batch in train_loader:
                idx_batch = idx_batch.tolist()
                batch = data[idx_batch].to(device)
                optimizer.zero_grad()
                x_recon, mu, logvar, z = self.forward(batch)
                loss_dict = self.loss(batch, x_recon, mu=mu, logvar=logvar, z=z, return_parts=True)
                if use_ssl:
                    neighbor_list = []; counts = []
                    for i in idx_batch:
                        bid = bin_ids[i]
                        ni = [j for j in bins[bid] if j != i]
                        counts.append(len(ni))
                        if ni: neighbor_list.append(data[ni])
                    if neighbor_list:
                        stacked = torch.cat(neighbor_list, dim=0).to(device)
                        mu_n, logvar_n = self.nograd_encode_dist(stacked)
                    else:
                        mu_n = torch.empty(0, self.latent_dim, device=device)
                        logvar_n = torch.empty(0, self.latent_dim, device=device)
                    ssl = self.ssl_loss(mu, logvar, mu_n, logvar_n, counts)
                    total = loss_dict['total'] + self.gamma_ssl * ssl
                    ep_losses['ssl'] += ssl.item()
                else:
                    total = loss_dict['total']
                total.backward(); optimizer.step()
                ep_losses['total'] += loss_dict['total'].item()
                ep_losses['recon'] += loss_dict['recon'].item()
                ep_losses['kl'] += loss_dict['kl'].item()
            nb = len(train_loader)
            for k in ep_losses: history[f'train_{k}'].append(ep_losses[k]/nb)
            self.eval()
            vl = {'total':0.0,'recon':0.0,'kl':0.0}
            with torch.no_grad():
                for idx_batch in val_loader:
                    batch = data[idx_batch.tolist()].to(device)
                    x_r, mu, lv, z = self.forward(batch)
                    ld = self.loss(batch, x_r, mu=mu, logvar=lv, z=z, return_parts=True)
                    for k in vl: vl[k] += ld[k].item()
            nvb = len(val_loader)
            if nvb > 0:
                for k in vl: history[f'val_{k}'].append(vl[k]/nvb)
            if verbose and (epoch+1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} | Train: {history['train_total'][-1]:.4f}"
                if use_ssl: msg += f", SSL: {history['train_ssl'][-1]:.4f}"
                print(msg)
        self.eval(); return history


# =============================================================================
# Search Phase (Fixed Mix: PSE + LVE mutation + LVE crossover)
# =============================================================================
class UniBinUniMemFixedMix:
    def __init__(self, agent_class, architecture, agent_kwargs=None,
                 mutation_sigma=0.3, n_pse=0, n_lve_mutation=0, n_lve_crossover=0,
                 init_fn=None):
        self.agent_class = agent_class; self.architecture = architecture
        self.agent_kwargs = agent_kwargs or {}
        self.mutation_sigma = mutation_sigma
        self.n_pse = n_pse; self.n_lve_mutation = n_lve_mutation
        self.n_lve_crossover = n_lve_crossover; self.init_fn = init_fn

    def make_agent(self, theta):
        agent = self.agent_class(self.architecture, **self.agent_kwargs)
        agent.set_weights(theta); return agent

    def _init(self):
        if self.init_fn is not None: return self.init_fn()
        return np.random.uniform(-1, 4, self.architecture[0] * self.architecture[1])

    def _select_parent(self, bm):
        bin_ids = list(bm.bins_idx.keys())
        bid = bin_ids[np.random.randint(len(bin_ids))]
        members = bm.bins_idx[bid]
        return members[np.random.randint(len(members))]

    def sample(self, latent_module=None, behavior_matching=None, **kwargs):
        n_total = self.n_pse + self.n_lve_mutation + self.n_lve_crossover
        if latent_module is None or behavior_matching is None or len(behavior_matching.bins) == 0:
            return [self._init() for _ in range(max(n_total, 1))]
        bm = behavior_matching; candidates = []
        for _ in range(self.n_pse):
            idx = self._select_parent(bm)
            parent = bm.dataset[idx]
            candidates.append(parent + self.mutation_sigma * np.random.randn(len(parent)))
        if self.n_lve_mutation == 0 and self.n_lve_crossover == 0: return candidates
        device = next(latent_module.parameters()).device
        latent_module.eval()
        if self.n_lve_mutation > 0:
            indices = [self._select_parent(bm) for _ in range(self.n_lve_mutation)]
            thetas = np.array([bm.dataset[i] for i in indices])
            with torch.no_grad():
                x = torch.tensor(thetas, dtype=torch.float32).to(device)
                mu, logvar = latent_module.encode_dist(x)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                z = z + self.mutation_sigma * torch.randn_like(z)
                decoded = latent_module.decode(z)
            candidates.extend([d.cpu().numpy() for d in decoded])
        if self.n_lve_crossover > 0:
            ia = [self._select_parent(bm) for _ in range(self.n_lve_crossover)]
            ib = [self._select_parent(bm) for _ in range(self.n_lve_crossover)]
            ta = np.array([bm.dataset[i] for i in ia])
            tb = np.array([bm.dataset[i] for i in ib])
            with torch.no_grad():
                xa = torch.tensor(ta, dtype=torch.float32).to(device)
                xb = torch.tensor(tb, dtype=torch.float32).to(device)
                mu_a = latent_module.encode(xa); mu_b = latent_module.encode(xb)
                alpha = torch.rand(self.n_lve_crossover, 1, device=device)
                z = alpha * mu_a + (1 - alpha) * mu_b
                decoded = latent_module.decode(z)
            candidates.extend([d.cpu().numpy() for d in decoded])
        return candidates


# =============================================================================
# SSLVE Orchestrator
# =============================================================================
class SSLVE:
    def __init__(self, search_phase, collector, behavior_matching, latent_module,
                 device='cpu'):
        self.SP = search_phase; self.CO = collector; self.BM = behavior_matching
        self.LM = latent_module; self.device = device
        self.history = {'fitness_min':[],'fitness_mean':[],'fitness_max':[],
                        'coverage':[],'archive_size':[]}

    def step(self, train_kwargs=None):
        if train_kwargs is None: train_kwargs = {}
        thetas = self.SP.sample(latent_module=self.LM, behavior_matching=self.BM)
        infos = []
        for theta in thetas:
            agent = self.SP.make_agent(theta)
            infos.append(self.CO.collect(agent))
        self.BM.update(thetas, infos)
        f_min, f_mean, f_max = self.BM.fitness_stats()
        self.history['fitness_min'].append(f_min)
        self.history['fitness_mean'].append(f_mean)
        self.history['fitness_max'].append(f_max)
        self.history['coverage'].append(self.BM.coverage())
        self.history['archive_size'].append(self.BM.archive_size())
        print(f"Archive: {self.BM.archive_size()}, Bins: {len(self.BM.bins)}, "
              f"Coverage: {self.BM.coverage():.4f}, "
              f"Fitness min/mean/max: {f_min:.2f}/{f_mean:.2f}/{f_max:.2f}")
        lm_history = self.LM.fit(dataset=self.BM.dataset, bin_ids=self.BM.bin_ids,
                                  bins=self.BM.bins_idx, device=self.device, **train_kwargs)
        return lm_history

    def run(self, n_steps, train_kwargs=None):
        histories = []
        for t in range(n_steps):
            print(f"\n--- SSLVE Step {t+1}/{n_steps} ---")
            histories.append(self.step(train_kwargs))
        return histories

    def plot_history(self, save_path=None):
        if not self.history['fitness_min']:
            print("No history."); return
        steps = range(len(self.history['fitness_min']))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax = axes[0]
        ax.plot(steps, self.history['fitness_min'], label='Min')
        ax.plot(steps, self.history['fitness_mean'], label='Mean')
        ax.plot(steps, self.history['fitness_max'], label='Max')
        ax.set_xlabel('Step'); ax.set_ylabel('Fitness')
        ax.set_title('Fitness over Steps'); ax.legend()
        axes[1].plot(steps, self.history['coverage'])
        axes[1].set_xlabel('Step'); axes[1].set_ylabel('Coverage')
        axes[1].set_title('Behavior Coverage')
        axes[2].plot(steps, self.history['archive_size'])
        axes[2].set_xlabel('Step'); axes[2].set_ylabel('Size')
        axes[2].set_title('Archive Size')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        else: plt.show()
        plt.close()


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
    input_dim=N_VOXELS, latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS, beta=BETA, gamma_ssl=GAMMA_SSL,
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
    search_phase=sp, collector=collector,
    behavior_matching=bm, latent_module=lm, device=DEVICE,
)

# =============================================================================
# Run
# =============================================================================
print(f"Grid shape: {GRID_SHAPE} ({N_VOXELS} voxels)")
print(f"Task: {TASK_NAME}")
print(f"Latent dim: {LATENT_DIM}")
print(f"Device: {DEVICE}")
print(f"Bins: {BIN_SIZES} (total={bd.total_bins()})")
print(f"Samples per step: {N_PSE} PSE + {N_LVE_MUTATION} LVE mut + {N_LVE_CROSSOVER} LVE xo = {N_PSE+N_LVE_MUTATION+N_LVE_CROSSOVER}")
print(f"Quick mode: {QUICK_EXPERIMENT}")
print()

train_kwargs = {
    'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
    'lr': LR, 'verbose': True,
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
sslve.plot_history()
