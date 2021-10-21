import os
import scipy.io
import random
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, random_split, Subset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as D
# from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder, Caltech256


def get_data_features(data_name='CIFAR10', data_dir='./dataset/', combine=True):
    dataset = {}
    data_name = data_name.upper() if 'cifar' in data_name.lower() else data_name
    # load data
    data_dir = os.path.join(data_dir, 'ExtractedFeatures/')
    train = scipy.io.loadmat(data_dir + data_name + 'DataTrn.mat')
    test = scipy.io.loadmat(data_dir + data_name + 'DataTst.mat')

    for x in ['train', 'test']:
        data = train['Trn'][0][0] if x == 'train' else test['Tst'][0][0]
        imgs, labels = (data[1], data[0]) if 'Fashion' in data_name else (data[0], data[1])
        labels = np.argmax(labels, 1)
        imgs, labels = Tensor(imgs), Tensor(labels)
        if combine:
            dataset[x] = TensorDataset(imgs, labels)
        else:
            dataset[x] = (imgs, labels)

    dataset['unlabeled'] = dataset['train']
    return dataset

def data_transform(data_name, model='others'):
    train_transform = None
    if 'cifar' in data_name:
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if model == 'random':
            train_transform = test_transform
        elif model == 'll':
            if data_name == 'cifar10':
                train_transform = T.Compose([
                    T.RandomHorizontalFlip(),  # origin Learning Loss
                    T.RandomCrop(size=32, padding=4),  # origin Learning Loss
                    T.ToTensor(),
                    T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
                ])
            elif data_name == 'cifar100': # cifar100
                train_transform = T.Compose([
                    T.RandomHorizontalFlip(),  # origin Learning Loss
                    T.RandomCrop(size=32, padding=4),  # origin Learning Loss
                    T.ToTensor(),
                    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
        elif model == 'vaal':
            train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif model == 'badge':
            if data_name == 'cifar10':
                train_transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
                ])
            elif data_name == 'cifar100': # cifar100
                train_transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
    elif 'fashion' in data_name:
        test_transform = T.Compose([
                #T.Pad(padding=2),
                T.ToTensor(),
                T.Normalize([0.5], [0.5])
            ])
        if model == 'random':
            train_transform = test_transform
        elif model == 'll':
            train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=28, padding=4),
                T.ToTensor(),
                T.Normalize([0.1307], [0.3081])
            ])
        elif model == 'vaal':
            train_transform = T.Compose([
                T.Pad(padding=2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5], [0.5])
            ])
        elif model == 'badge':
            train_transform = T.Compose([
                T.Pad(padding=2),
                T.ToTensor(),
                T.Normalize([0.1307], [0.3081])
            ])
    elif 'caltech' in data_name:
        test_transform = T.Compose([
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if model == 'random':
            train_transform = test_transform
        elif model == 'll':
            train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=224),
                T.ToTensor(),
                T.Normalize([0.5353, 0.5085, 0.4769], [0.3065, 0.3022, 0.3139])
            ])
        elif model == 'vaal':
            train_transform = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                #T.Pad(16),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif model == 'badge':
            train_transform = T.Compose([
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.5353, 0.5085, 0.4769], [0.3065, 0.3022, 0.3139])
            ])
    if train_transform is None:
        train_transform = test_transform
    return train_transform, test_transform


def get_data_raw(data_name='cifar10', data_dir='./dataset/', model='others'):
    dataset = {}
    train_transform, test_transform = data_transform(data_name, model)

    if data_name == 'cifar10':
        dataset['train'] = D.CIFAR10(data_dir + 'cifar10', train=True, download=True, transform=train_transform)
        dataset['unlabeled'] = D.CIFAR10(data_dir + 'cifar10', train=True, download=True, transform=test_transform)
        dataset['test'] = D.CIFAR10(data_dir + 'cifar10', train=False, download=True, transform=test_transform)

    elif data_name == 'cifar100':
        dataset['train'] = D.CIFAR100(data_dir + 'cifar100', train=True, download=True, transform=train_transform)
        dataset['unlabeled'] = D.CIFAR100(data_dir + 'cifar100', train=True, download=True, transform=test_transform)
        dataset['test'] = D.CIFAR100(data_dir + 'cifar100', train=False, download=True, transform=test_transform)

    elif 'fashion' in data_name:
        dataset['train'] = D.FashionMNIST(data_dir + 'fashionmnist', train=True, download=True, transform=train_transform)
        dataset['unlabeled'] = D.FashionMNIST(data_dir + 'fashionmnist', train=True, download=True, transform=test_transform)
        dataset['test'] = D.FashionMNIST(data_dir + 'fashionmnist', train=False, download=True, transform=test_transform)

    elif 'caltech' in data_name:
        indices = [*range(30607)]
        #random.shuffle(indices)
        train_indices, test_indices = indices[:22897], indices[22897:] # len 7710 # for normalize

        train_transform, test_transform = data_transform(data_name, model)

        train_data = D.ImageFolder(root=data_dir + 'caltech256', transform=train_transform)
        test_data = D.ImageFolder(root=data_dir + 'caltech256', transform=test_transform)

        dataset['train'] = Subset(train_data, train_indices)
        dataset['unlabeled'] = Subset(test_data, train_indices)
        dataset['test'] = Subset(test_data, test_indices)

    return dataset



def get_data(data_name='CIFAR10', data_dir='./dataset/', use_feature=False, model='others'):
    model = model.lower()
    if use_feature:
        if model in ['random', 'maxentropy', 'coreset', 'badge']:
            dataset = get_data_features(data_name, data_dir, combine=False)
        else:
            dataset = get_data_features(data_name, data_dir)
    else:
        dataset = get_data_raw(data_name.lower(), data_dir, model)

    if model == 'vaal':
        dataset = {'train': Dataset_idx(dataset['train']),
                   'test': Dataset_idx(dataset['test'])}
    return dataset



def get_data_info(data_name):
    dataset_info = {
        'CIFAR10': [10, 50000, 10000],
        'CIFAR100': [100, 50000, 10000],
        'FashionMNIST': [10, 60000, 10000],
        'Caltech256': [257, 22897, 7710]
    }
    return dataset_info[data_name]

### VAAL

class Dataset_idx(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.dataset[index]

        return data, target, index

    def __len__(self):
        return len(self.dataset)

### BADGE

class DataHandler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


### to normalize images
def get_img_mean(Dataset):
    loader = DataLoader(
        Dataset,
        batch_size=1000, num_workers=0, shuffle=False
    )

    mean = torch.zeros(3)
    mean2 = torch.zeros(3)
    total = torch.zeros(1)
    print('--> get mean&stdv of images')
    for data, _ in loader:
        mean += torch.sum(data, dim=(0, 2, 3), keepdim=False)
        mean2 += torch.sum((data ** 2), dim=(0, 2, 3), keepdim=False)
        total += data.size(0)

    total *= (data.size(2) ** 2)
    mean /= total
    std = torch.sqrt((mean2 - total * (mean ** 2)) / (total - 1))

    mean = list(np.around(mean.numpy(), 4))
    std = list(np.around(std.numpy(), 4))
    return mean, std