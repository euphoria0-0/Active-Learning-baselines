'''
Reference:
    https://github.com/JordanAsh/badge
'''
import sys
import pdb
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from scipy import stats
from sklearn.metrics import pairwise_distances

from .active_learner import ActiveLearner


class BADGE(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.nClass = args.nClass

    def query(self, nQuery, model):
        # get grad embedding
        embDim = model.get_embedding_dim()
        model.eval()

        unlabeled_dataset = Dataset_idx_(Subset(self.dataset['unlabeled'], self.unlabeled_indices))
        unlabeled_loader = DataLoader(unlabeled_dataset, **self.loader_args)

        embedding = np.zeros([len(self.unlabeled_indices), embDim * self.nClass])
        with torch.no_grad():
            for input, labels, idxs in unlabeled_loader:
                input, labels = Variable(input.to(self.device)), Variable(input.to(self.device))
                cout, out = model(input)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)

                for j in range(len(labels)):
                    for c in range(self.nClass):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])

        # init centers
        query_indices = init_centers(embedding, nQuery)

        # update query data
        self.update(np.array(self.unlabeled_indices)[query_indices].tolist())


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    print('len X {}'.format(X.shape))
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        sys.stdout.write('\r{}\t{:.10f}'.format(len(mu), sum(D2)))
        sys.stdout.flush()
        if len(mu) % 100 == 0:
            print('{}\t{:.10f}'.format(len(mu), sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    '''gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]'''
    print()
    return indsAll


class Dataset_idx_:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        data, target, _ = self.dataset[index]
        return data, target, index

    def __len__(self):
       return len(self.dataset)