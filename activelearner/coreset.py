'''
Reference:
    https://github.com/ej0cl6/deep-active-learning
'''
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from .active_learner import ActiveLearner
from data.dataset import Dataset_idx


class CoreSet(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.dataloaders['unlabeled'] = DataLoader(Dataset_idx(self.dataset['unlabeled']),
                                                   batch_size=self.batch_size, pin_memory=True, shuffle=False)

    def query(self, nQuery, model):
        # get embedding
        model.eval()
        embedding = np.zeros([self.nTrain, model.get_embedding_dim()])
        with torch.no_grad():
            for input, labels, idxs in self.dataloaders['unlabeled']:
                input, labels = Variable(input.to(self.device)), Variable(labels.to(self.device))
                out, e1 = model(input)
                embedding[idxs] = e1.data.cpu().data.numpy()

        # furthest_first
        X = embedding[self.unlabeled_indices, :]
        X_set = embedding[self.labeled_indices, :]

        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        query_indices = []
        for _ in tqdm(range(nQuery), desc='CoreSet'):
            idx = min_dist.argmax()
            query_indices.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        # update query data
        self.update(np.array(self.unlabeled_indices)[query_indices].tolist())
