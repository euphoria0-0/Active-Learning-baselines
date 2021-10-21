
import os
import wandb
import argparse
import torch

from data import reddit, federated_emnist
from model import mlp, resnet



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu cuda index')

    parser.add_argument('--al_method','-AL', type=str, default='random',
                        choices=['random','learningloss','coreset','badge','ws','vaal'])

    parser.add_argument('--dataset', help='dataset', type=str, default='CIFAR10')
    parser.add_argument('--data_dir', help='data path', type=str, default='../dataset/')
    parser.add_argument('--use_features', action='store_true', default=True, help='use feature extracted from ImageNet')

    parser.add_argument('--num_trial', type=int, default=10, help='number of trials')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=30, help='Batch size used for training only')
    parser.add_argument('--data_size', help='number of points at each round', type=list,
                        default=[600,800,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000])

    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix seed for reproducible')
    parser.add_argument('--seed', type=int, default=0, help='seed number')

    args = parser.parse_args()
    return args


def load_data(args):
    if args.use_features:
        if args.al_method == 'VAAL':
            return 1
        else:
            return 2
    else:
        return 0


def create_model(args):
    if args.use_features:
        return mlp.MLP(num_classes=args.num_classes, model=args.al_method)
    else:
        return resnet.ResNet18(num_classes=args.num_classes)


def active_learning_method(args):
    if args.al_method == 'random':
        return RandomSampling
    elif args.al_method == 'learningloss':
        return LearningLoss
    elif args.al_method == 'coreset':
        return CoreSet
    elif args.al_method == 'badge':
        return CoreSet
    elif args.al_method == 'ws':
        return WS
    elif args.al_method == 'vaal':
        return VAAL
    else:
        return 1



if __name__ == '__main__':
    # set up
    args = get_args()
    wandb.init(
        project=f'AL-{args.dataset}-{args.use_features}',
        name=f"{args.al_method}-{args.optimizer}-{args.num_epoch}-{args.lr}-{args.wdecay}",
        config=args
    )
    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)  # change allocation of current GPU
    print('Current cuda device: {}'.format(torch.cuda.current_device()))

    # set data
    data = load_data(args)
    args.num_classes = data.num_classes
    dataset = data.dataset

    # set model
    model = create_model(args)
    AL_method = active_learning_method(args)

    # set federated optim algorithm
