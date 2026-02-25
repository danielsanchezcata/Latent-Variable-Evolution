import numpy as np

class DataGeneration:
    """
    Dataset generation via evolutionary algorithm.
    
    Args:
        problem: ProblemSetting instance
        toolbox: DEAP toolbox (already configured with operators)
        generate_fn: function(toolbox, problem) -> list of individuals
    """
    
    def __init__(self, problem, toolbox, generate_fn):
        self.problem = problem
        self.toolbox = toolbox
        self.generate_fn = generate_fn
    
    def generate(self):
        """
        Run generation function and return dataset.
        Returns: list of individuals (numpy arrays)
        """
        return self.generate_fn(self.toolbox, self.problem)
