import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class ActiveLearner:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.init_size = args.data_size[0]
        self.device = args.device
        self.nTrain = args.nTrain
        self.args = args
        self.loader_args = {'batch_size': args.batch_size, 'pin_memory': True, 'shuffle': False}
        self._init_setting()

    def _init_setting(self):
        total_indices = np.arange(self.nTrain)
        self.labeled_indices = np.random.choice(total_indices, self.init_size, replace=False).tolist()
        self.unlabeled_indices = list(set(total_indices) - set(self.labeled_indices))

        if self.args.al_method == 'vaal' and self.args.use_features:
            self.dataset_ftrs, self.dataset = self.dataset
            self.dataloaders = {
                'ftrs': self._get_dataloaders(self.dataset_ftrs),
                'imgs': self._get_dataloaders(self.dataset)
            }
        else:
            self.dataloaders = self._get_dataloaders(self.dataset)

    def _get_dataloaders(self, dataset):
        dataloaders = {
            'train': DataLoader(dataset['train'], **self.loader_args,
                                sampler=SubsetRandomSampler(self.labeled_indices)),
            'unlabeled': DataLoader(dataset['unlabeled'], **self.loader_args,
                                    sampler=SubsetRandomSampler(self.unlabeled_indices)),
            'test': DataLoader(dataset['test'], **self.loader_args)
        }
        return dataloaders

    def update(self, query_indices):
        print(f'selected data: {sorted(query_indices)[:10]}')
        self.labeled_indices += query_indices
        self.unlabeled_indices = list(set(self.unlabeled_indices) - set(query_indices))

        if self.args.al_method == 'vaal' and self.args.use_features:
            self.dataloaders = {
                'ftrs': self._get_dataloaders(self.dataset_ftrs),
                'imgs': self._get_dataloaders(self.dataset)
            }
        else:
            self.dataloaders = self._get_dataloaders(self.dataset)


    def get_current_dataloaders(self):
        return self.dataloaders

    def query(self, n, model):
        pass