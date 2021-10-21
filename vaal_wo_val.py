'''
Variational Adversarial Active Learning

Source:
    https://github.com/sinhasam/vaal

Reference:
    [Sinha et al. 2019] Variational Adversarial Active Learning (https://arxiv.org/abs/1904.00370)
'''

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import numpy as np
import argparse
import random
import os
from datetime import datetime
import visdom


# Custom
from custom_data import get_data
from utils import utils
from acquisition.vaal_solver import Solver


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='If training is to be done on a GPU')
    parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--visuallize", action="store_true", default=False, help="use visdom")
    parser.add_argument('--vis_port', type=int, default=9000, help='visdom port num')

    parser.add_argument('--data', type=str, default='CIFAR10', help='Name of the dataset used.')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='Path to where the data is')
    parser.add_argument('--use_features', action='store_true', default=False,
                        help='use feature extracted from ImageNet')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size used for training and testing')
    parser.add_argument('--num_trial', type=int, default=1, help='number of trials')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num_cycle', type=int, default=15, help='number of acquisition')
    parser.add_argument('--init_size', type=int, default=600, help='size of initial pool')
    parser.add_argument('--query_size', type=int, default=1000, help='number of querrying data')

    parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=1,
                        help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2,
                        help='Number of VAE steps taken for every task model step')
    parser.add_argument('--adversary_param', type=float, default=1,
                        help='Hyperparameter for training. lambda2 in the paper')

    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix seed for reproducible')
    parser.add_argument('--seed', type=int, default=0, help='seed number')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='train-validation split ratio')

    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log',
                        help='Final performance of the models will be saved with this name')

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    return args


def main(args):

    print(args)
    args.model_name = 'VAAL'

    args.out_path = './results/{}/{}/'.format(args.model_name, args.data)
    if not os.path.exists(args.out_path):
        os.makedirs('./results/', exist_ok=True)
        os.makedirs('./results/{}/'.format(args.model_name), exist_ok=True)
        os.makedirs(args.out_path, exist_ok=True)

    if args.fix_seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

    # python -m visdom.server -port 9000
    if args.visuallize:
        args.vis = visdom.Visdom(server='http://localhost', port=args.vis_port)
        args.plot_data = {'X': [], 'Y': [], 'legend': ['Train Acc', 'Train Loss']}
    else:
        args.vis, args.plot_data = None, None

    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)  # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())  # check

    # load data
    print('======== {} ========'.format(args.data))
    args.num_classes, args.num_images, _ = get_data.get_data_info(args.data)
    args.num_val = args.num_images - int(args.num_images * args.split_ratio)

    dataset = get_data.get_data(args.data, args.data_dir, use_feature=False, model='vaal')
    if args.use_features:
        feature_dataset = get_data.get_data(args.data, args.data_dir, use_feature=True, model='vaal')

    QuerySize = args.query_size
    total = [[] for _ in range(args.num_trial)]

    for trial in range(args.num_trial):

        all_indices = set(np.arange(args.num_images))

        initial_indices = random.sample(list(all_indices), args.init_size)
        sampler = SubsetRandomSampler(initial_indices)

        # dataset with labels available
        querry_dataloader = DataLoader(dataset['train'], sampler=sampler,
                                       batch_size=args.batch_size, drop_last=False)
        test_dataloader = DataLoader(dataset['test'], batch_size=1000, drop_last=False)

        if args.use_features:
            feature_train_dataloader = DataLoader(feature_dataset['train'], batch_size=args.batch_size, pin_memory=True,
                                                  shuffle=False, sampler=sampler)
            feature_test_dataloader = DataLoader(feature_dataset['test'], batch_size=1000, pin_memory=True,
                                                 shuffle=True)
        else:
            feature_train_dataloader, feature_test_dataloader = None, None


        splits = [*range(1000,1000+args.num_cycle*args.query_size,args.query_size)]
        splits = [600,800] + splits
        current_indices = list(initial_indices)

        accuracies = []

        for cycle, split in enumerate(splits):
            # need to retrain all the models on the new images
            # re initialize and retrain the models
            args.query_size = 200 if len(current_indices) < 1000 else QuerySize
            print(len(current_indices), args.query_size)
            solver = Solver(args, test_dataloader, feature_test_dataloader)

            unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
            unlabeled_sampler = SubsetRandomSampler(unlabeled_indices)
            unlabeled_dataloader = DataLoader(dataset['train'], sampler=unlabeled_sampler,
                                              batch_size=args.batch_size, drop_last=False)

            # train the models on the current data
            acc, sampled_indices = solver.train(querry_dataloader,
                                                unlabeled_dataloader,
                                                feature_train_dataloader)

            print('\n====== Labeled {} Acc {:.2f}'.format(split, acc))
            accuracies.append(acc)
            total[trial].append([len(current_indices), acc.item()])

            current_indices = list(current_indices) + list(sampled_indices)
            sampler = SubsetRandomSampler(current_indices)
            querry_dataloader = DataLoader(dataset['train'], sampler=sampler,
                                                batch_size=args.batch_size, drop_last=False)
            if args.use_features:
                feature_train_dataloader = DataLoader(feature_dataset['train'], batch_size=args.batch_size,
                                                      pin_memory=True, shuffle=False, sampler=sampler)

        torch.save(accuracies, os.path.join(args.out_path, args.log_name))

    return total


if __name__ == '__main__':
    start = datetime.now()
    args = get_args()
    args.start = start
    total = main(args)
    utils.save_results(args, total)
    utils.alarm('DONE! {} {}'.format(args.model_name, args.data))
