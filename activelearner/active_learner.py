import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class ActiveLearner:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.init_size = args.data_size[0]
        self.batch_size = args.batch_size
        self.device = args.device
        self.nTrain = args.nTrain
        self._init_setting()



    def _init_setting(self):
        total_indices = np.arange(self.nTrain)
        self.labeled_indices = np.random.choice(total_indices, self.init_size, replace=False).tolist()
        self.unlabeled_indices = list(set(total_indices) - set(self.labeled_indices))

        self.dataloaders = {
            'train': DataLoader(self.dataset['train'], batch_size=self.batch_size, pin_memory=True, shuffle=False,
                                sampler=SubsetRandomSampler(self.labeled_indices)),
            'unlabeled': DataLoader(self.dataset['unlabeled'], batch_size=self.batch_size, pin_memory=True, shuffle=False,
                                    sampler=SubsetRandomSampler(self.unlabeled_indices)),
            'test': DataLoader(self.dataset['test'], batch_size=1000, pin_memory=True, shuffle=True)
        }

    def get_current_dataloaders(self):
        return self.dataloaders

    def query(self, n, model):
        pass