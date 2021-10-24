import numpy as np
from .active_learner import ActiveLearner


class RandomSampling(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)

    def query(self, nQuery, model=None):
        query_indices = np.random.choice(self.unlabeled_indices, nQuery, replace=False).tolist()
        self.labeled_indices += query_indices
        self.unlabeled_indices = list(set(self.unlabeled_indices) - set(query_indices))
