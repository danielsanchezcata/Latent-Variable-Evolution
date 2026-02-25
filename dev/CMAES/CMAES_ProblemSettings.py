import gymnasium as gym
from abc import ABC, abstractmethod
import numpy as np

class ProblemSetting(ABC):
    """
    Abstract class for problem definition.
    
    Args:
        constraint_handling: 'rejection' | 'penalty'
        penalty_coef: coefficient for penalty (used if constraint_handling='penalty')
    """
    
    def __init__(self, constraint_handling='rejection', penalty_coef=1.0):
        self.constraint_handling = constraint_handling
        self.penalty_coef = penalty_coef
    
    @abstractmethod
    def fitness(self, x):
        """Return fitness value (to be minimized)."""
        pass
    
    @abstractmethod
    def constraint(self, x):
        """Return constraint value. Feasible if <= 0."""
        pass
    
    def has_constraint(self):
        """Whether this problem has constraints. Override to return False if unconstrained."""
        return True
    
    def is_feasible(self, x):
        """Check if solution satisfies constraint."""
        return self.constraint(x) <= 0
    
    def constraint_violation(self, x):
        """Return constraint violation (0 if feasible)."""
        return max(0, self.constraint(x))
    
    def evaluate(self, x):
        """
        Evaluate fitness with constraint handling.
        Returns (fitness_value, is_feasible)
        """
        fit = self.fitness(x)
        feasible = self.is_feasible(x)
        
        if self.constraint_handling == 'penalty' and not feasible:
            fit += self.penalty_coef * self.constraint_violation(x)
        
        return fit, feasible



class CartPoleProblem(ProblemSetting):
    """
    CartPole-v1 problem. Fitness = negative mean steps alive (minimize).
    No constraint.

    Args:
        max_steps: max steps per episode (CartPole-v1 default 500)
        n_episodes: number of episodes to average over
        seed: random seed for env
    """

    def __init__(self, max_steps=500, n_episodes=5, seed=None):
        super().__init__(constraint_handling='rejection', penalty_coef=0.0)
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed = seed

    def fitness(self, agent):
        """
        Run agent in CartPole, return negative mean steps (minimize).
        Also stores state trajectories in self.last_trajectories.

        Args:
            agent: agent with act(obs) method, weights already set

        Returns:
            negative mean steps alive
        """
        total_steps = 0
        self.last_trajectories = []

        for ep in range(self.n_episodes):
            env = gym.make('CartPole-v1')
            obs, _ = env.reset(seed=self.seed + ep if self.seed is not None else None)
            trajectory = [obs.copy()]
            steps = 0

            for _ in range(self.max_steps):
                action = agent.act(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                trajectory.append(obs.copy())
                steps += 1
                if terminated or truncated:
                    break

            total_steps += steps
            self.last_trajectories.append(np.array(trajectory))
            env.close()

        return -total_steps / self.n_episodes

    def constraint(self, agent):
        return -1.0

    def has_constraint(self):
        return False

    def get_behavior(self, agent):
        """
        Run agent and return behavior descriptor: (mean_position, mean_angle).
        Uses last_trajectories if available, otherwise runs fitness first.

        Returns:
            (mean_position, mean_angle)
        """
        if not hasattr(self, 'last_trajectories') or not self.last_trajectories:
            self.fitness(agent)

        all_states = np.concatenate(self.last_trajectories, axis=0)
        mean_pos = all_states[:, 0].mean()
        mean_angle = all_states[:, 2].mean()
        return mean_pos, mean_angle

    def render_agent(self, agent):
        """
        Render one episode of the agent in CartPole.

        Args:
            agent: agent with act(obs) method, weights already set

        Returns:
            list of RGB frames
        """
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        obs, _ = env.reset(seed=self.seed if self.seed is not None else None)
        frames = [env.render()]

        for _ in range(self.max_steps):
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break

        env.close()
        return frames



class BipedalWalkerProblem(ProblemSetting):
    """
    BipedalWalker-v3 problem. Fitness = negative total reward (minimize).
    No constraint. Tracks leg contact flags for behavior descriptor.

    Args:
        max_steps: max steps per episode (default 1600)
        n_episodes: number of episodes to average over
        seed: random seed for env
    """

    def __init__(self, max_steps=1600, n_episodes=3, seed=None):
        super().__init__(constraint_handling='rejection', penalty_coef=0.0)
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed = seed

    def fitness(self, agent):
        """
        Run agent in BipedalWalker, return negative mean total reward (minimize).
        Stores leg contact histories and final hull x positions.

        Args:
            agent: agent with act(obs) method, weights already set

        Returns:
            negative mean total reward
        """
        total_reward = 0
        self.last_leg_contacts = []
        self.last_final_x = []

        for ep in range(self.n_episodes):
            env = gym.make('BipedalWalker-v3')
            obs, _ = env.reset(seed=self.seed + ep if self.seed is not None else None)
            ep_reward = 0
            ep_contacts = []

            for _ in range(self.max_steps):
                action = agent.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_contacts.append((obs[8], obs[13]))
                if terminated or truncated:
                    break

            final_x = env.unwrapped.hull.position.x
            total_reward += ep_reward
            self.last_leg_contacts.append(ep_contacts)
            self.last_final_x.append(final_x)
            env.close()

        return -total_reward / self.n_episodes

    def constraint(self, agent):
        return -1.0

    def has_constraint(self):
        return False

    def get_behavior(self, agent):
        """
        Return behavior descriptor: (final_x, leg1_contact_ratio, leg2_contact_ratio).
        Uses cached data if available, otherwise runs fitness first.

        Returns:
            (mean_final_x, leg1_contact_ratio, leg2_contact_ratio)
        """
        if not hasattr(self, 'last_leg_contacts') or not self.last_leg_contacts:
            self.fitness(agent)

        # Final x: average across episodes
        mean_final_x = np.mean(self.last_final_x)

        # Leg contacts: average across all timesteps of all episodes
        all_contacts = []
        for ep_contacts in self.last_leg_contacts:
            all_contacts.extend(ep_contacts)

        contacts = np.array(all_contacts)
        leg1_ratio = contacts[:, 0].mean()
        leg2_ratio = contacts[:, 1].mean()
        return mean_final_x, leg1_ratio, leg2_ratio

    def render_agent(self, agent):
        """
        Render one episode of the agent.

        Returns:
            list of RGB frames
        """
        env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
        obs, _ = env.reset(seed=self.seed if self.seed is not None else None)
        frames = [env.render()]

        for _ in range(self.max_steps):
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break

        env.close()
        return frames
