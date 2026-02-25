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
