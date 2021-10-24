import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data.dataset import Dataset_idx


class ActiveLearner:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.init_size = args.data_size[0]
        self.batch_size = args.batch_size
        self.device = args.device
        self.nTrain = args.nTrain
        self.args = args
        self._init_setting()

    def _init_setting(self):
        total_indices = np.arange(self.nTrain)
        self.labeled_indices = np.random.choice(total_indices, self.init_size, replace=False).tolist()
        self.unlabeled_indices = list(set(total_indices) - set(self.labeled_indices))

        if self.args.al_method == 'vaal' and self.args.use_features:
            self.dataset_ftrs, self.dataset = self.dataset
            self.dataloaders_ftrs = self._get_dataloaders_idx(self.dataset_ftrs)
            self.dataloaders_imgs = self._get_dataloaders_idx(self.dataset)
            self.dataloaders = [self.dataloaders_ftrs, self.dataloaders_imgs]
        else:
            self.dataloaders = self._get_dataloaders_idx(self.dataset)

    def _get_dataloaders_idx(self, dataset):
        dataloaders = {
            'train': DataLoader(Dataset_idx(dataset['train']), batch_size=self.batch_size, pin_memory=True, shuffle=False,
                                sampler=SubsetRandomSampler(self.labeled_indices)),
            'unlabeled': DataLoader(Dataset_idx(dataset['unlabeled']), batch_size=self.batch_size, pin_memory=True, shuffle=False,
                                    sampler=SubsetRandomSampler(self.unlabeled_indices)),
            'test': DataLoader(Dataset_idx(dataset['test']), batch_size=1000, pin_memory=True, shuffle=True)
        }
        return dataloaders


    def update(self, query_indices):
        print(f'selected data: {sorted(query_indices)[:10]}')
        self.labeled_indices += query_indices
        self.unlabeled_indices = list(set(self.unlabeled_indices) - set(query_indices))

        self.dataloaders = self._get_dataloaders_idx(self.dataset)

        if self.args.al_method == 'vaal' and self.args.use_features:
            self.dataloaders_ftrs = self._get_dataloaders_idx(self.dataset_ftrs)


    def get_current_dataloaders(self):
        return self.dataloaders

    def query(self, n, model):
        pass

