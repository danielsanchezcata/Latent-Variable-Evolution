import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, cma, tools
from IPython import display as ipythondisplay
from matplotlib import animation


class CartPole_CMAES(CMAES_SearchPhase):
    """
    CMA-ES search phase for CartPole with MLP agents.

    Args:
        problem: CartPoleProblem instance
        architecture: list of layer dims, e.g. [4, 32, 2]
        agent_class: agent class (e.g. MLP_Agent)
        sigma: initial CMA-ES step size
        lambda_: CMA-ES population size per generation
        n_gen: number of CMA-ES generations
    """

    # Behavior discretization
    POS_MIN, POS_MAX = -2.4, 2.4
    ANGLE_MIN, ANGLE_MAX = -0.2095, 0.2095
    N_BINS = 30

    def __init__(self, problem, architecture, agent_class, sigma=1.0, lambda_=50, n_gen=100):
        super().__init__(problem)
        self.architecture = architecture
        self.agent_class = agent_class
        self.sigma = sigma
        self.lambda_ = lambda_
        self.n_gen = n_gen

        self.weight_dim = agent_class(architecture).get_weight_dim()

        self.best = None
        self.best_fitness = float('inf')
        self.history = {
            'fitness': {'min': [], 'max': [], 'mean': []},
            'behavior_std': [],
            'behavior_coverage': []
        }
        self.map_count = np.zeros((self.N_BINS, self.N_BINS), dtype=int)

    def _he_centroid(self):
        """He initialization for centroid vector."""
        centroid = np.zeros(self.weight_dim)
        offset = 0
        for i in range(len(self.architecture) - 1):
            fan_in = self.architecture[i]
            fan_out = self.architecture[i + 1]
            w_size = fan_in * fan_out
            std = np.sqrt(2.0 / fan_in)
            centroid[offset:offset + w_size] = np.random.normal(0, std, size=w_size)
            offset += w_size
            centroid[offset:offset + fan_out] = 0.0  # bias = 0
            offset += fan_out
        return centroid

    def _evaluate(self, flat_weights):
        """Evaluate a flat weight vector. Returns (fitness,) tuple for DEAP."""
        agent = self.agent_class(self.architecture)
        agent.set_weights(np.array(flat_weights))
        fit = self.problem.fitness(agent)
        behavior = self.problem.get_behavior(agent)
        return fit, behavior

    def _norm_behavior(self, pos, angle):
        """MinMax normalize behavior to [0, 1]."""
        norm_pos = np.clip((pos - self.POS_MIN) / (self.POS_MAX - self.POS_MIN), 0, 1)
        norm_angle = np.clip((angle - self.ANGLE_MIN) / (self.ANGLE_MAX - self.ANGLE_MIN), 0, 1)
        return norm_pos, norm_angle

    def _update_map(self, norm_pos, norm_angle):
        """Update MAP-count with a single behavior sample."""
        bi = min(int(norm_pos * self.N_BINS), self.N_BINS - 1)
        bj = min(int(norm_angle * self.N_BINS), self.N_BINS - 1)
        self.map_count[bi, bj] = 1

    def _record_generation(self, fitnesses, behaviors):
        """Record per-generation stats."""
        fits = np.array(fitnesses)
        self.history['fitness']['min'].append(float(fits.min()))
        self.history['fitness']['max'].append(float(fits.max()))
        self.history['fitness']['mean'].append(float(fits.mean()))

        norm_behaviors = np.array([self._norm_behavior(p, a) for p, a in behaviors])
        std_pos = norm_behaviors[:, 0].std()
        std_angle = norm_behaviors[:, 1].std()
        self.history['behavior_std'].append(float((std_pos + std_angle) / 2))

        for np_, na_ in norm_behaviors:
            self._update_map(np_, na_)

        coverage = self.map_count.sum() / (self.N_BINS * self.N_BINS)
        self.history['behavior_coverage'].append(float(coverage))

    def sample(self, behavior_matching=None):
        """
        Run CMA-ES and return final generation's population as weight vectors.

        Args:
            behavior_matching: unused for now (for SSLVE compatibility)

        Returns:
            list of numpy arrays (flat weight vectors)
        """
        # Setup DEAP
        if hasattr(creator, "CMAFitness"):
            del creator.CMAFitness
        if hasattr(creator, "CMAIndividual"):
            del creator.CMAIndividual

        creator.create("CMAFitness", base.Fitness, weights=(-1.0,))
        creator.create("CMAIndividual", list, fitness=creator.CMAFitness)

        centroid = self._he_centroid()
        strategy = cma.Strategy(centroid=centroid.tolist(), sigma=self.sigma, lambda_=self.lambda_)

        toolbox = base.Toolbox()
        toolbox.register("generate", strategy.generate, creator.CMAIndividual)
        toolbox.register("update", strategy.update)

        for gen in range(self.n_gen):
            population = toolbox.generate()

            fitnesses = []
            behaviors = []
            for ind in population:
                fit, beh = self._evaluate(ind)
                ind.fitness.values = (fit,)
                fitnesses.append(fit)
                behaviors.append(beh)

                # Track best
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best = np.array(ind, dtype=np.float32).copy()

            toolbox.update(population)
            self._record_generation(fitnesses, behaviors)

            print(f"CMA-ES Gen {gen+1}/{self.n_gen}, "
                  f"Fitness min: {self.history['fitness']['min'][-1]:.1f}, "
                  f"mean: {self.history['fitness']['mean'][-1]:.1f}, "
                  f"coverage: {self.history['behavior_coverage'][-1]:.4f}")

        # Return final generation as list of numpy arrays
        return [np.array(ind, dtype=np.float32) for ind in population]

    def plot_history(self):
        """Plot fitness, behavior std, and behavior coverage over generations."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        gens = range(len(self.history['fitness']['mean']))

        ax = axes[0]
        ax.plot(gens, self.history['fitness']['mean'], label='Mean')
        ax.plot(gens, self.history['fitness']['min'], label='Min')
        ax.plot(gens, self.history['fitness']['max'], label='Max')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (neg steps)')
        ax.set_title('Fitness')
        ax.legend()

        ax = axes[1]
        ax.plot(gens, self.history['behavior_std'])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Behavior Std (avg)')
        ax.set_title('Behavior Diversity (Std)')

        ax = axes[2]
        ax.plot(gens, self.history['behavior_coverage'])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Coverage')
        ax.set_title('Behavior Coverage (MAP)')

        plt.tight_layout()
        plt.show()

    def show_best(self):
        """Render and display the best agent as animation."""
        if self.best is None:
            print("No best agent found yet.")
            return

        agent = self.agent_class(self.architecture)
        agent.set_weights(self.best)
        frames = self.problem.render_agent(agent)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')
        img = ax.imshow(frames[0])

        def animate(i):
            img.set_data(frames[i])
            return [img]

        anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                       interval=50, blit=True)
        plt.close(fig)

        try:
            from IPython.display import HTML
            return HTML(anim.to_html5_video())
        except Exception:
            plt.show()
            return anim


class BipedalWalker_CMAES(CMAES_SearchPhase):
    """
    CMA-ES search phase for BipedalWalker with MLP agents.
    Behavior descriptor: (final_x, leg1_contact_ratio, leg2_contact_ratio).
    MAP-Elites grid: 100 x 10 x 10 = 10,000 cells.

    Args:
        problem: BipedalWalkerProblem instance
        architecture: list of layer dims, e.g. [24, 64, 64, 4]
        agent_class: agent class (e.g. MLP_Agent)
        sigma: initial CMA-ES step size
        lambda_: CMA-ES population size per generation
        n_gen: number of CMA-ES generations
        output_activation: output activation for agent (default 'tanh')
    """

    # Behavior discretization
    X_MIN, X_MAX = 0.0, 100.0
    N_BINS_X = 100
    N_BINS_LEG = 10

    def __init__(self, problem, architecture, agent_class, sigma=1.0,
                 lambda_=50, n_gen=100, output_activation='tanh'):
        super().__init__(problem)
        self.architecture = architecture
        self.agent_class = agent_class
        self.sigma = sigma
        self.lambda_ = lambda_
        self.n_gen = n_gen
        self.output_activation = output_activation

        self.weight_dim = agent_class(architecture, output_activation).get_weight_dim()

        self.best = None
        self.best_fitness = float('inf')
        self.history = {
            'fitness': {'min': [], 'max': [], 'mean': []},
            'behavior_std_x': [],
            'behavior_std_leg1': [],
            'behavior_std_leg2': [],
            'behavior_std_sum': [],
            'behavior_coverage': []
        }
        self.map_count = np.zeros((self.N_BINS_X, self.N_BINS_LEG, self.N_BINS_LEG), dtype=int)

    def _he_centroid(self):
        """He initialization for centroid vector."""
        centroid = np.zeros(self.weight_dim)
        offset = 0
        for i in range(len(self.architecture) - 1):
            fan_in = self.architecture[i]
            fan_out = self.architecture[i + 1]
            w_size = fan_in * fan_out
            std = np.sqrt(2.0 / fan_in)
            centroid[offset:offset + w_size] = np.random.normal(0, std, size=w_size)
            offset += w_size
            centroid[offset:offset + fan_out] = 0.0
            offset += fan_out
        return centroid

    def _evaluate(self, flat_weights):
        """Evaluate a flat weight vector. Returns (fitness, behavior)."""
        agent = self.agent_class(self.architecture, self.output_activation)
        agent.set_weights(np.array(flat_weights))
        fit = self.problem.fitness(agent)
        behavior = self.problem.get_behavior(agent)
        return fit, behavior

    def _update_map(self, final_x, leg1, leg2):
        """Update MAP-count with a single 3D behavior sample."""
        norm_x = np.clip((final_x - self.X_MIN) / (self.X_MAX - self.X_MIN), 0, 1)
        bi = min(int(norm_x * self.N_BINS_X), self.N_BINS_X - 1)
        bj = min(int(np.clip(leg1, 0, 1) * self.N_BINS_LEG), self.N_BINS_LEG - 1)
        bk = min(int(np.clip(leg2, 0, 1) * self.N_BINS_LEG), self.N_BINS_LEG - 1)
        self.map_count[bi, bj, bk] = 1

    def _record_generation(self, fitnesses, behaviors):
        """Record per-generation stats."""
        fits = np.array(fitnesses)
        self.history['fitness']['min'].append(float(fits.min()))
        self.history['fitness']['max'].append(float(fits.max()))
        self.history['fitness']['mean'].append(float(fits.mean()))

        behaviors_arr = np.array(behaviors)  # (N, 3): final_x, leg1, leg2
        # Normalize final_x to [0, 1] for std calculation
        norm_x = np.clip((behaviors_arr[:, 0] - self.X_MIN) / (self.X_MAX - self.X_MIN), 0, 1)
        std_x = float(norm_x.std())
        std_leg1 = float(behaviors_arr[:, 1].std())
        std_leg2 = float(behaviors_arr[:, 2].std())
        self.history['behavior_std_x'].append(std_x)
        self.history['behavior_std_leg1'].append(std_leg1)
        self.history['behavior_std_leg2'].append(std_leg2)
        self.history['behavior_std_sum'].append(std_x + std_leg1 + std_leg2)

        for beh in behaviors:
            self._update_map(beh[0], beh[1], beh[2])

        total_cells = self.N_BINS_X * self.N_BINS_LEG * self.N_BINS_LEG
        coverage = self.map_count.sum() / total_cells
        self.history['behavior_coverage'].append(float(coverage))

    def sample(self, behavior_matching=None):
        """
        Run CMA-ES and return final generation's population as weight vectors.

        Returns:
            list of numpy arrays (flat weight vectors)
        """
        if hasattr(creator, "CMAFitness"):
            del creator.CMAFitness
        if hasattr(creator, "CMAIndividual"):
            del creator.CMAIndividual

        creator.create("CMAFitness", base.Fitness, weights=(-1.0,))
        creator.create("CMAIndividual", list, fitness=creator.CMAFitness)

        centroid = self._he_centroid()
        strategy = cma.Strategy(centroid=centroid.tolist(), sigma=self.sigma, lambda_=self.lambda_)

        toolbox = base.Toolbox()
        toolbox.register("generate", strategy.generate, creator.CMAIndividual)
        toolbox.register("update", strategy.update)

        for gen in range(self.n_gen):
            population = toolbox.generate()

            fitnesses = []
            behaviors = []
            for ind in population:
                fit, beh = self._evaluate(ind)
                ind.fitness.values = (fit,)
                fitnesses.append(fit)
                behaviors.append(beh)

                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best = np.array(ind, dtype=np.float32).copy()

            toolbox.update(population)
            self._record_generation(fitnesses, behaviors)

            print(f"CMA-ES Gen {gen+1}/{self.n_gen}, "
                  f"Fitness min: {self.history['fitness']['min'][-1]:.1f}, "
                  f"mean: {self.history['fitness']['mean'][-1]:.1f}, "
                  f"coverage: {self.history['behavior_coverage'][-1]:.4f}")

        return [np.array(ind, dtype=np.float32) for ind in population]

    def plot_history(self):
        """Plot fitness, behavior std (4 lines), and behavior coverage over generations."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        gens = range(len(self.history['fitness']['mean']))

        ax = axes[0]
        ax.plot(gens, self.history['fitness']['mean'], label='Mean')
        ax.plot(gens, self.history['fitness']['min'], label='Min')
        ax.plot(gens, self.history['fitness']['max'], label='Max')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (neg reward)')
        ax.set_title('Fitness')
        ax.legend()

        ax = axes[1]
        ax.plot(gens, self.history['behavior_std_x'], label='final_x')
        ax.plot(gens, self.history['behavior_std_leg1'], label='leg1_contact')
        ax.plot(gens, self.history['behavior_std_leg2'], label='leg2_contact')
        ax.plot(gens, self.history['behavior_std_sum'], label='sum', linestyle='--', color='black')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Behavior Std')
        ax.set_title('Behavior Diversity (Std)')
        ax.legend()

        ax = axes[2]
        ax.plot(gens, self.history['behavior_coverage'])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Coverage')
        ax.set_title('Behavior Coverage (MAP)')

        plt.tight_layout()
        plt.show()

    def show_best(self):
        """Render and display the best agent as animation."""
        if self.best is None:
            print("No best agent found yet.")
            return

        agent = self.agent_class(self.architecture, self.output_activation)
        agent.set_weights(self.best)
        frames = self.problem.render_agent(agent)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')
        img = ax.imshow(frames[0])

        def animate(i):
            img.set_data(frames[i])
            return [img]

        anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                       interval=50, blit=True)
        plt.close(fig)

        try:
            from IPython.display import HTML
            return HTML(anim.to_html5_video())
        except Exception:
            plt.show()
            return anim
