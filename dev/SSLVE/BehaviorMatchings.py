import numpy as np
from abc import ABC, abstractmethod



class BehaviorMatching(ABC):
    """
    Abstract class for behavior-matching archive management.

    Maintains:
        dataset   : list of numpy arrays (stored thetas)
        fitnesses : list of floats
        bin_ids   : list of bin IDs, bin_ids[i] is bin of dataset[i]
        bins      : dict {bin_id: [(theta, fitness), ...]}
        bins_idx  : dict {bin_id: [dataset_indices]}

    Args:
        behavior_descriptor: BehaviorDescriptor instance
        fitness_fn: callable(info_dict) -> float (scalar fitness, to minimize)
    """

    def __init__(self, behavior_descriptor, fitness_fn):
        self.behavior_descriptor = behavior_descriptor
        self.fitness_fn = fitness_fn
        self.dataset = []
        self.fitnesses = []
        self.bin_ids = []
        self.bins = {}
        self.bins_idx = {}

    @abstractmethod
    def update(self, thetas, infos):
        """
        Update archive with new candidates.

        Args:
            thetas: list of numpy arrays
            infos: list of info dicts from Collector.collect()

        Updates self.dataset, self.fitnesses, self.bin_ids, self.bins, self.bins_idx in place.
        """
        pass
        

class MAPElitesBM:
    """
    MAP-Elites style behavior matching with top-k per bin.

    self.bins always stores {bin_id: [(theta, fitness), ...]}.
    _rebuild() creates index-based views for LM training:
        self.dataset, self.fitnesses, self.bin_ids, self.bins_idx

    Args:
        behavior_descriptor: object with describe(info) and discretize(descriptor)
        fitness_fn: callable(info) -> float (to minimize)
        top_k: max entries per bin
    """

    def __init__(self, behavior_descriptor, fitness_fn, top_k=10):
        self.behavior_descriptor = behavior_descriptor
        self.fitness_fn = fitness_fn
        self.top_k = top_k
        self.bins = {}  # {bin_id: [(theta, fitness), ...]}
        self.dataset = []
        self.fitnesses = []
        self.bin_ids = []
        self.bins_idx = {}  # {bin_id: [dataset_indices]}

    def update(self, thetas, infos):
        """
        Update archive with new candidates, then rebuild index views.

        Args:
            thetas: list of numpy arrays
            infos: list of info dicts from Collector.collect()
        """
        for theta, info in zip(thetas, infos):
            descriptor = self.behavior_descriptor.describe(info)
            bin_id = self.behavior_descriptor.discretize(descriptor)
            fitness = self.fitness_fn(info)

            if bin_id not in self.bins:
                self.bins[bin_id] = []
            self.bins[bin_id].append((theta, fitness))

            # Keep top-k per bin (lowest fitness = best, since minimizing)
            if len(self.bins[bin_id]) > self.top_k:
                self.bins[bin_id].sort(key=lambda x: x[1])
                self.bins[bin_id] = self.bins[bin_id][:self.top_k]

        self._rebuild()

    def _rebuild(self):
        """Build index-based views from self.bins."""
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
        """Fraction of occupied bins over total possible bins."""
        total = self.behavior_descriptor.total_bins()
        return len(self.bins) / total

    def archive_size(self):
        """Total number of entries in archive."""
        return len(self.dataset)

    def fitness_stats(self):
        """Return min, mean, max fitness across archive."""
        if not self.fitnesses:
            return 0.0, 0.0, 0.0
        f = np.array(self.fitnesses)
        return float(f.min()), float(f.mean()), float(f.max())
