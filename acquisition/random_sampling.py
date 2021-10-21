import numpy as np
from .strategy_features import Strategy
import pdb

class RandomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args, device)

    def query(self, n):
        inds = np.where(self.idxs_lb==0)[0]
        return inds[np.random.permutation(len(inds))][:n]
