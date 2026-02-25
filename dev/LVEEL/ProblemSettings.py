import numpy as np

class SphereProblem(ProblemSetting):
    """
    Sphere function with C1 constraint.
    fitness: f(x) = sum(x_i^2)
    constraint: sum(45 - x_i) <= 0  (i.e., sum(x_i) >= 45*D)
    """
    
    def __init__(self, dim, constraint_handling='rejection', penalty_coef=1.0):
        super().__init__(constraint_handling, penalty_coef)
        self.dim = dim
    
    def fitness(self, x):
        return sum(xi ** 2 for xi in x)
    
    def constraint(self, x):
        return sum(45 - xi for xi in x)


class NKLandscape(ProblemSetting):
    """
    NK Landscape problem (unconstrained).
    
    Args:
        n: number of loci (binary)
        k: epistatic interactions per locus (0 to n-1)
        adjacent: if True, interactions are adjacent (sliding window); else random
        seed: random seed for generating interaction tables
    """
    
    def __init__(self, n, k, adjacent=True, seed=None):
        super().__init__(constraint_handling='rejection', penalty_coef=0.0)
        self.n = n
        self.k = k
        self.adjacent = adjacent
        
        rng = np.random.RandomState(seed)
        
        # Build interaction neighbors for each locus
        self.neighbors = []
        for i in range(n):
            if adjacent:
                nbrs = [(i + j + 1) % n for j in range(k)]
            else:
                candidates = list(range(n))
                candidates.remove(i)
                nbrs = rng.choice(candidates, size=k, replace=False).tolist()
            self.neighbors.append(nbrs)
        
        # Fitness contribution tables: for each locus, 2^(k+1) entries
        self.tables = []
        for i in range(n):
            table = rng.uniform(0, 1, size=2 ** (k + 1))
            self.tables.append(table)
    
    def _contribution(self, x, i):
        """Fitness contribution of locus i given binary string x."""
        bits = [int(x[i])] + [int(x[j]) for j in self.neighbors[i]]
        index = 0
        for b in bits:
            index = (index << 1) | b
        return self.tables[i][index]
    
    def fitness(self, x):
        """Return negative average contribution (to minimize)."""
        avg = sum(self._contribution(x, i) for i in range(self.n)) / self.n
        return -avg
    
    def constraint(self, x):
        """No constraint. Always feasible."""
        return -1.0
    
    def has_constraint(self):
        return False


class C2NKLandscape(NKLandscape):
    """
    NK Landscape with two-sided ratio constraint on fraction of 1s.
    Constraint: ratio_min <= sum(x)/n <= ratio_max
    
    Args:
        n, k, adjacent, seed: same as NKLandscape
        ratio_min: minimum fraction of 1s
        ratio_max: maximum fraction of 1s
        penalty_coef: penalty coefficient for constraint violation
    """
    
    def __init__(self, n, k, adjacent=True, seed=None, ratio_min=0.1, ratio_max=0.3, penalty_coef=1.0):
        super().__init__(n, k, adjacent, seed)
        self.constraint_handling = 'penalty'
        self.penalty_coef = penalty_coef
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
    
    def constraint(self, x):
        """
        Two-sided ratio constraint.
        Returns max violation (>0 means infeasible, <=0 means feasible).
        """
        ratio = sum(int(xi) for xi in x) / self.n
        lower_violation = self.ratio_min - ratio
        upper_violation = ratio - self.ratio_max
        return max(lower_violation, upper_violation)
    
    def has_constraint(self):
        return True

