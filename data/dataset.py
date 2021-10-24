import os
from scipy.io import loadmat
import numpy as np
from torch import Tensor
from torch.utils.data import TensorDataset, Subset
import torchvision.transforms as T
import torchvision.datasets as D
import random
from copy import deepcopy

class Dataset:
    def __init__(self, args, pretrained=True):
        self.data = args.dataset
        self.data_dir = args.data_dir
        self.use_features = args.use_features
        self.pretrained = pretrained
        self.al_method = args.al_method

        if self.data == 'CIFAR10':
            self.nClass = 10
            self.nTrain = 50000
            self.nTest = 10000

        elif self.data == 'CIFAR100':
            self.nClass = 100
            self.nTrain = 50000
            self.nTest = 10000

        elif self.data == 'FashionMNIST':
            self.nClass = 10
            self.nTrain = 60000
            self.nTest = 10000

        elif self.data == 'Caltech256':
            self.nClass = 257
            self.nTrain = 22897
            self.nTest = 7710

        self.dataset = {}
        self._getData()
        if self.al_method == 'vaal' and self.use_features:
            self.dataset_ftrs = deepcopy(self.dataset)
            self.use_features = False
            self.pretrained = False
            self._getData()
            self.dataset = [self.dataset_ftrs, self.dataset]


    def _getData(self):
        if self.use_features:
            self._get_features_data()
        else:
            self._data_transform()
            if self.data == 'CIFAR10':
                self.dataset['train'] = D.CIFAR10(self.data_dir+'cifar10', train=True, transform=self.train_transform)
                self.dataset['unlabeled'] = D.CIFAR10(self.data_dir+'cifar10', train=True, transform=self.test_transform)
                self.dataset['test'] = D.CIFAR10(self.data_dir+'cifar10', train=False, transform=self.test_transform)
                self.dataset['label'] = self.dataset['train'].targets

            elif self.data == 'CIFAR100':
                self.dataset['train'] = D.CIFAR100(self.data_dir+'cifar100', train=True, transform=self.train_transform)
                self.dataset['unlabeled'] = D.CIFAR100(self.data_dir+'cifar100', train=True, transform=self.test_transform)
                self.dataset['test'] = D.CIFAR100(self.data_dir+'cifar100', train=False, transform=self.test_transform)
                self.dataset['label'] = self.dataset['train'].targets

            elif self.data == 'FashionMNIST':
                self.dataset['train'] = D.FashionMNIST(self.data_dir+'fashionmnist', train=True, transform=self.train_transform)
                self.dataset['unlabeled'] = D.FashionMNIST(self.data_dir+'fashionmnist', train=True, transform=self.test_transform)
                self.dataset['test'] = D.FashionMNIST(self.data_dir+'fashionmnist', train=False, transform=self.test_transform)
                self.dataset['label'] = self.dataset['train'].targets

            elif self.data == 'Caltech256':
                indices = [*range(self.nTrain + self.nTest)]
                random.seed(0)
                random.shuffle(indices)
                train_indices, test_indices = indices[:self.nTrain], indices[self.nTrain:]

                train_data = D.ImageFolder(root=self.data_dir+'Caltech256', transform=self.train_transform)
                unlabeled_data = D.ImageFolder(root=self.data_dir+'Caltech256', transform=self.test_transform)
                test_data = D.ImageFolder(root=self.data_dir+'Caltech256', transform=self.test_transform)

                self.dataset['train'] = Subset(train_data, train_indices)
                self.dataset['unlabeled'] = Subset(unlabeled_data, train_indices)
                self.dataset['test'] = Subset(test_data, test_indices)
                self.dataset['label'] = [x[1] for idx, x in enumerate(train_data.imgs) if idx in train_indices]

        self.dataset['train'] = Dataset_idx(self.dataset['train'])
        self.dataset['unlabeled'] = Dataset_idx(self.dataset['unlabeled'])
        self.dataset['test'] = Dataset_idx(self.dataset['test'])


    def _get_features_data(self, combine=True):
        # load data
        train = loadmat(os.path.join(self.data_dir, 'ExtractedFeatures/', self.data + 'DataTrn.mat'))
        test = loadmat(os.path.join(self.data_dir, 'ExtractedFeatures/', self.data + 'DataTst.mat'))

        for x in ['train', 'test']:
            data = train['Trn'][0][0] if x == 'train' else test['Tst'][0][0]
            imgs, labels = (data[1], data[0]) if self.data == 'FashionMNIST' else (data[0], data[1])
            labels = np.argmax(labels, 1)
            imgs, labels = Tensor(imgs), Tensor(labels)
            if combine:
                self.dataset[x] = TensorDataset(imgs, labels)
            else:
                self.dataset[x] = (imgs, labels)

        self.dataset['unlabeled'] = self.dataset['train']
        return self.dataset


    def _data_transform(self):
        if self.data == 'CIFAR10':
            mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            if self.al_method == 'learningloss':
                add_transform = [T.RandomHorizontalFlip(),
                                 T.RandomCrop(size=32, padding=4)]
            elif self.al_method == 'vaal':
                add_transform = [T.RandomHorizontalFlip()]
                if self.pretrained:
                    add_transform.insert(0, T.Resize(224))
            elif self.al_method == 'ws':
                add_transform = [T.RandomCrop(size=32, padding=4),
                                 T.RandomHorizontalFlip()]
            else:
                add_transform = []

        elif self.data == 'CIFAR100':
            mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            if self.al_method == 'learningloss':
                add_transform = [T.RandomHorizontalFlip(),
                                 T.RandomCrop(size=32, padding=4)]
            elif self.al_method == 'vaal':
                add_transform = [T.RandomHorizontalFlip()]
                if self.pretrained:
                    add_transform.insert(0, T.Resize(224))
            else:
                add_transform = []

        elif self.data == 'FashionMNIST':
            mean, std = [0.1307], [0.3081]
            if self.al_method == 'learningloss':
                add_transform = [T.RandomHorizontalFlip(),
                                 T.RandomCrop(size=28, padding=4)]
            elif self.al_method == 'vaal':
                add_transform = [T.Pad(2), T.RandomHorizontalFlip()]
                if self.pretrained:
                    add_transform = [T.Resize(224), T.RandomHorizontalFlip()]
            else:
                add_transform = [T.Pad(2)]

        elif self.data == 'Caltech256':
            mean, std = [0.5414, 0.5153, 0.4832], [0.305, 0.3009, 0.3134]
            if self.al_method == 'learningloss':
                add_transform = [T.RandomHorizontalFlip(),
                                 T.RandomCrop(size=224)]
            elif self.al_method == 'vaal':
                add_transform = [T.RandomResizedCrop(224),
                                 T.RandomHorizontalFlip()]
            else:
                add_transform = [T.Resize(224)]

        base_transform = [T.ToTensor(), T.Normalize(mean, std)]
        self.test_transform = T.Compose(base_transform)
        self.train_transform = T.Compose(add_transform + base_transform)



class Dataset_idx:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
       return len(self.dataset)



