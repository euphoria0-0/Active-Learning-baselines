import numpy as np
import sys
import os
import argparse
import torch
import random
from datetime import datetime

from acquisition.random_sampling import RandomSampling
from acquisition.badge_sampling import BadgeSampling
from acquisition.core_set import CoreSet

from custom_data import get_data
from models import mlp, resnet
from utils import utils


# code based on https://github.com/ej0cl6/deep-active-learning"
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu cuda index')

    parser.add_argument('--model_name', help='query strategy: select in Random, CoreSet, BADGE', type=str, default='Random')

    parser.add_argument('--data_dir', help='data path', type=str, default='../dataset/')
    parser.add_argument('--data', help='dataset (non-openML)', type=str, default='CIFAR10')
    parser.add_argument('--use_features', action='store_true', default=True, help='use feature extracted from ImageNet')

    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix seed for reproducible')
    parser.add_argument('--seed', type=int, default=0, help='seed number')

    parser.add_argument('--num_trial', help='number of trials', type=int, default=5)
    parser.add_argument('--num_cycle', type=int, default=15, help='number of acquisition(round)')
    parser.add_argument('--num_epoch', help='number of epochs', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size used for training only')
    parser.add_argument('--init_size', help='number of points to start', type=int, default=600)
    parser.add_argument('--query_size', help='number of points to query in a batch', type=int, default=1000)

    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)  # Adam lr

    args = parser.parse_args()

    return args


## custom functions
def get_dataset_args(data_name='CIFAR10', num_epoch=100, batch_size=128):
    args_pool = {}
    args_pool['CIFAR10'] = {
        'nClasses': 10,
        'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.3},
    }
    args_pool['FashionMNIST'] = {
        'nClasses': 10,
        'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
        'optimizer_args': {'lr': 0.005, 'momentum': 0.5},
    }
    args_pool['CIFAR100'] = {
        'nClasses': 100,
        'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.3},
    }
    args_pool['Caltech256'] = {
        'nClasses': 257,
        'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.3},
    }
    args_pool[data_name]['n_epoch'] = num_epoch
    args_pool[data_name]['loader_tr_args'] = {'batch_size': batch_size, 'num_workers': 0}
    return args_pool[data_name]


def get_query_strategy(model_name, options):
    (X_tr, Y_tr, idxs_lb, net, handler, args, device) = options
    if model_name.lower() == 'random':
        return RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args, device)
    elif model_name.lower() == 'coreset':
        return CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args, device)
    elif model_name.lower() == 'badge':
        return BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args, device)


if __name__ == '__main__':
    opts = get_args()
    opts.start = datetime.now()

    opts.out_path = './results/{}/{}/'.format(opts.model_name, opts.data)
    if not os.path.exists(opts.out_path):
        os.makedirs('./results/', exist_ok=True)
        os.makedirs('./results/{}/'.format(opts.model_name), exist_ok=True)
        os.makedirs(opts.out_path, exist_ok=True)

    if opts.fix_seed:
        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

    # non-openml data defaults
    args = get_dataset_args(opts.data, opts.num_epoch, opts.batch_size)
    opts.nClasses = args['nClasses']
    args['lr'] = opts.lr

    if not os.path.exists(opts.data_dir):
        os.makedirs(opts.data_dir)

    # load non-openml dataset
    dataset = get_data.get_data_features(opts.data, opts.data_dir, combine=False)
    (X_tr, Y_tr), (X_te, Y_te) = dataset['train'], dataset['test']
    opts.dim = np.shape(X_tr)[1:]
    handler = get_data.DataHandler

    # start experiment
    n_pool = len(Y_tr)
    n_test = len(Y_te)
    print('number of labeled pool: {}'.format(opts.init_size), flush=True)
    print('number of unlabeled pool: {}'.format(n_pool - opts.init_size), flush=True)
    print('number of testing pool: {}'.format(n_test), flush=True)

    device = torch.device(f"cuda:{opts.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())

    # print info
    print(opts.data, flush=True)
    print(opts.model_name, flush=True)

    ## TRIAL
    total = [[] for _ in range(opts.num_trial)]
    QuerySize = opts.query_size

    for trial in range(opts.num_trial):
        print('====== TRIAL {} ======'.format(trial))
        # generate initial labeled pool
        idxs_lb = np.zeros(n_pool, dtype=bool)
        idxs_tmp = np.arange(n_pool)
        np.random.shuffle(idxs_tmp)
        idxs_lb[idxs_tmp[:opts.init_size]] = True

        acc = np.zeros(opts.num_cycle + 2)
        for rd in range(opts.num_cycle + 2):
            opts.query_size = 200 if sum(idxs_lb) < 1000 else QuerySize
            print('== Round {}: Labeled {} Query {}'.format(rd, sum(idxs_lb), opts.query_size), flush=True)

            # load specified network
            if opts.use_features:
                net = mlp.MLP(num_classes=opts.nClasses, model=opts.model_name.lower())
            else:
                net = resnet.ResNet18(num_classes=opts.nClasses)

            if type(X_tr[0]) is not np.ndarray:
                X_tr = X_tr.numpy()

            strategy = get_query_strategy(opts.model_name, (X_tr, Y_tr, idxs_lb, net, handler, args, device))
            strategy.train()

            # round accuracy
            P = strategy.predict(X_te, Y_te)
            acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
            print('Label {} Test Acc: {:.4f}'.format(sum(idxs_lb), acc[rd]))
            total[trial].append([sum(idxs_lb), acc[rd].item()])

            # query
            if rd < opts.num_cycle + 1:
                output = strategy.query(opts.query_size)
                q_idxs = output
                print(len(q_idxs))
                idxs_lb[q_idxs] = True
                print(sum(idxs_lb))

    utils.save_results(opts, total)
    utils.alarm('DONE! {} {}'.format(opts.model_name, opts.data))

