from abc import ABC, abstractmethod


class CMAES_SearchPhase(ABC):
    """
    Abstract class for search phase in CMAES.

    At t=0: sample(behavior_matching=None) generates initial dataset Theta_0.
    At t>0: sample(behavior_matching) uses current bins and dataset to generate new samples.

    Args:
        problem: ProblemSetting instance
    """

    def __init__(self, problem):
        self.problem = problem

    @abstractmethod
    def sample(self, behavior_matching=None):
        """
        Generate new samples.

        Args:
            behavior_matching: BehaviorMatching instance (None at t=0)

        Returns:
            list of numpy arrays (new candidate genes)
        """
        pass
