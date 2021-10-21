import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import numpy as np
import scipy.io


def get_data_features(data_name='CIFAR10', data_dir='../dataset/'):
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
        imgs, labels = torch.Tensor(imgs), torch.Tensor(labels)
        dataset[x] = TensorDataset(imgs, labels)

    return dataset


class indexDataset(Dataset):
    # to index previous training dataset in active_js_1_4.py
    # it is used for distilling previous network
    def __init__(self, data_name, root, train, transform):
        # self.cifar10 = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        self.dataset = get_data_features(data_name, root)
        if train:
            self.dataset = self.dataset['train']
        else:
            self.dataset = self.dataset['test']

    def __getitem__(self, index):
        data, target = self.dataset[index]
        # Your transformations here

        return data, target, index

    def __len__(self):
        return len(self.dataset)


def getCIFAR10(batch_size, data_root='./dataset/', train=True, val=True, **kwargs):
    # data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    #data_root = os.path.expanduser(os.path.join(data_root, '/'))
    num_workers = kwargs.setdefault('num_workers', 0)
    drop_last = kwargs.setdefault('drop_last', False)
    shuffle = kwargs.setdefault('shuffle', True)
    sampler = kwargs.setdefault('sampler', None)
    index = kwargs.pop('index', False)

    if 'transform_train' in kwargs:
        transform_train = kwargs.pop('transform_train')
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if 'transform_test' in kwargs:
        transform_test = kwargs.pop('transform_test')
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        if index:
            dataset = indexDataset('CIFAR10', root=data_root, train=True, transform=transform_train)
        else:
            dataset = get_data_features('CIFAR10', data_dir=data_root)['train']
            # dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        if sampler is not None:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                                       sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                                       shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        ds.append(train_loader)

    if val:
        # test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        test_dataset = get_data_features('CIFAR10', data_dir=data_root)['test']
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                  num_workers=num_workers, drop_last=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR100(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    #data_root = os.path.expanduser(os.path.join(data_root, '/'))
    num_workers = kwargs.setdefault('num_workers', 0)
    drop_last = kwargs.setdefault('drop_last', False)
    shuffle = kwargs.setdefault('shuffle', True)
    sampler = kwargs.setdefault('sampler', None)
    index = kwargs.pop('index', False)

    if 'transform_train' in kwargs:
        transform_train = kwargs.pop('transform_train')
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
    if 'transform_test' in kwargs:
        transform_test = kwargs.pop('transform_test')
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        # print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        if index:
            dataset = indexDataset('CIFAR100', root=data_root, train=True, transform=transform_train)
        else:
            # dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
            dataset = get_data_features('CIFAR100', data_dir=data_root)['train']
        if sampler is not None:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                                       sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                                       shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        ds.append(train_loader)

    if val:
        # test_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
        test_dataset = get_data_features('CIFAR100', data_dir=data_root)['test']
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers, drop_last=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getFashionMNIST(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    # data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    #data_root = os.path.expanduser(os.path.join(data_root, '/'))
    num_workers = kwargs.setdefault('num_workers', 0)
    drop_last = kwargs.setdefault('drop_last', False)
    shuffle = kwargs.setdefault('shuffle', True)
    sampler = kwargs.setdefault('sampler', None)
    index = kwargs.pop('index', False)

    if 'transform_train' in kwargs:
        transform_train = kwargs.pop('transform_train')
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    if 'transform_test' in kwargs:
        transform_test = kwargs.pop('transform_test')
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    ds = []
    if train:
        if index:
            dataset = indexDataset('FashionMNIST', root=data_root, train=True, transform=transform_train)
        else:
            dataset = get_data_features('FashionMNIST', data_dir=data_root)['train']
            # dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        if sampler is not None:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                                       sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                                       shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        ds.append(train_loader)

    if val:
        # test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        test_dataset = get_data_features('FashionMNIST', data_dir=data_root)['test']
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                  num_workers=num_workers, drop_last=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCaltech256(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    # data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    #data_root = os.path.expanduser(os.path.join(data_root, '/'))
    num_workers = kwargs.setdefault('num_workers', 0)
    drop_last = kwargs.setdefault('drop_last', False)
    shuffle = kwargs.setdefault('shuffle', True)
    sampler = kwargs.setdefault('sampler', None)
    index = kwargs.pop('index', False)

    if 'transform_train' in kwargs:
        transform_train = kwargs.pop('transform_train')
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(224), # padding=4
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5353, 0.5085, 0.4769], [0.3065, 0.3022, 0.3139])
        ])
    if 'transform_test' in kwargs:
        transform_test = kwargs.pop('transform_test')
    else:
        transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5353, 0.5085, 0.4769], [0.3065, 0.3022, 0.3139])
        ])
    ds = []
    if train:
        if index:
            dataset = indexDataset('Caltech256', root=data_root, train=True, transform=transform_train)
        else:
            dataset = get_data_features('Caltech256', data_dir=data_root)['train']
        if sampler is not None:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                                       sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                                       shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        ds.append(train_loader)

    if val:
        test_dataset = get_data_features('Caltech256', data_dir=data_root)['test']
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                  num_workers=num_workers, drop_last=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getDataSet(data_type, batch_size, dataroot, **kwargs):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, data_root=dataroot, **kwargs)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, data_root=dataroot, **kwargs)

    elif data_type == 'FashionMNIST':
        train_loader, test_loader = getFashionMNIST(batch_size=batch_size, data_root=dataroot, **kwargs)
    elif data_type == 'Caltech256':
        train_loader, test_loader = getCaltech256(batch_size=batch_size, data_root=dataroot, **kwargs)

    elif data_type == 'indexed_cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, data_root=dataroot, index=True, **kwargs)
    elif data_type == 'indexed_cifar100':
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, data_root=dataroot, index=True, **kwargs)

    elif data_type == 'indexed_FashionMNIST':
        train_loader, test_loader = getFashionMNIST(batch_size=batch_size, data_root=dataroot, index=True, **kwargs)
    elif data_type == 'indexed_Caltech256':
        train_loader, test_loader = getCaltech256(batch_size=batch_size, data_root=dataroot, index=True, **kwargs)


    # for calculating test set entropy
    elif data_type == 'cifar10_eval':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, data_root=dataroot, \
                                               transform_train=transform_train, shuffle=False, drop_last=False,
                                               **kwargs)
    # for calculating test set entropy
    elif data_type == 'cifar100_eval':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, data_root=dataroot, \
                                                transform_train=transform_train, shuffle=False, drop_last=False,
                                                **kwargs)
    # for calculating test set entropy
    elif data_type == 'FashionMNIST_eval':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])
        train_loader, test_loader = getFashionMNIST(batch_size=batch_size, data_root=dataroot, shuffle=False,
                                                    drop_last=False, transform_train=transform_train, **kwargs)
    elif data_type == 'Caltech256_eval':
        transform_train = transforms.Compose([
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.5353, 0.5085, 0.4769], [0.3065, 0.3022, 0.3139])
        ])
        train_loader, test_loader = getCaltech256(batch_size=batch_size, data_root=dataroot, shuffle=False,
                                                  drop_last=False, transform_train=transform_train, **kwargs)

    return train_loader, test_loader