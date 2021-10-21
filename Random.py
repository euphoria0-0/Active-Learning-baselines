'''
Random Query

Reference:
    [Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import random
import sys

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Utils
import visdom
from datetime import datetime
import argparse

# Custom
from custom_data import get_data, sampler
from models import lossnet, mlp, resnet
from utils import utils
import copy


##
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu cuda index')
    parser.add_argument("--parallel", action="store_true", default=False, help="use DataParallel")
    parser.add_argument("--visuallize", action="store_true", default=False, help="use visdom")
    parser.add_argument('--vis_port', type=int, default=9000, help='visdom port num')

    parser.add_argument('--data', type=str, default='CIFAR10', help='Name of the dataset used.')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='Path to where the data is')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size used for training and testing')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num_trial', type=int, default=5, help='number of trials')
    parser.add_argument('--num_cycle', type=int, default=15, help='number of acquisition')
    parser.add_argument('--init_size', type=int, default=600, help='size of initial pool')
    parser.add_argument('--query_size', type=int, default=1000, help='number of querrying data')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='wdecay for SGD')
    parser.add_argument('--milestone', type=int, default=160, help='number of acquisition')

    parser.add_argument('--use_features', action='store_true', default=False, help='use feature extracted from ImageNet')
    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix seed for reproducible')
    parser.add_argument('--seed', type=int, default=0, help='seed number')

    args = parser.parse_args()

    return args


#
def train_epoch(model, criterion, optimizer, dataloaders):
    model.train()

    total, correct = 0, 0
    for (inputs, labels) in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device).long()

        optimizer.zero_grad()

        scores = model(inputs)
        target_loss = criterion(scores, labels)

        _, preds = torch.max(scores.data, 1)
        total += labels.size(0)
        correct += torch.sum(preds == labels.data)

        loss = torch.sum(target_loss) / target_loss.size(0)
        loss.backward()
        optimizer.step()

    return model, correct.double() / total, loss.item()


#
def test(model, dataloaders, mode='test'):
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs, labels = inputs.to(device), labels.to(device).long()

            scores = model(inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)

    return correct.double() / total


#
def train(model, criterion, optimizer, scheduler, dataloaders, args, cycle=0):
    for epoch in range(args.num_epoch):
        model, train_acc, loss = train_epoch(model, criterion, optimizer, dataloaders)
        scheduler.step()

        sys.stdout.write('\rEpoch {}/{} TrainLoss {:.4f} TrainAcc {:.4f}'.format(epoch+1, args.num_epoch,
                                                                                 loss, train_acc))
        sys.stdout.flush()

        if (args.vis != None) and (args.plot_data != None):
            utils.vis_plot(epoch+cycle*args.num_epoch,
                           [loss, train_acc, test_acc],
                           args.vis, args.plot_data, ymax=3)

    return train_acc, loss


def random_acquisition(labeled_indices, unlabeled_indices, query_size):
    unlabeled = copy.deepcopy(unlabeled_indices)
    random.shuffle(unlabeled)
    query_indices = unlabeled[:query_size]
    labeled_indices += query_indices
    unlabeled = list(set(unlabeled) - set(query_indices))
    return labeled_indices, unlabeled



##
# Main
if __name__ == '__main__':
    args = get_args()
    args.start = datetime.now()
    print(args)
    args.model_name = 'Learning-Loss'

    args.out_path = './results/{}/{}/'.format(args.model_name, args.data)
    if not os.path.exists(args.out_path):
        os.makedirs('./results/', exist_ok=True)
        os.makedirs('./results/{}/'.format(args.model_name), exist_ok=True)
        os.makedirs(args.out_path, exist_ok=True)

    if args.fix_seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    # python -m visdom.server -port 9000
    if args.visuallize:
        args.vis = visdom.Visdom(server='http://localhost', port=args.vis_port)
        args.plot_data = {'X': [], 'Y': [], 'legend': ['Total Loss', 'Train Acc', 'Test Acc']}
    else:
        args.vis, args.plot_data = None, None

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)  # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())  # check

    # load data
    print('======== {} ========'.format(args.data))
    dataset = get_data.get_data(args.data, args.data_dir, args.use_features)
    dataset['unlabeled'] = dataset['train']
    num_classes, train_size, test_size = get_data.get_data_info(args.data)
    labeled_size = args.init_size

    total = [[] for _ in range(args.num_trial)]
    for trial in range(args.num_trial):
        print('====== TRIAL {} ======'.format(trial))

        indices = list(range(train_size))
        random.shuffle(indices)
        labeled_indices = indices[:args.init_size]
        unlabeled_indices = indices[args.init_size:]

        dataloaders = {
            'train': DataLoader(dataset['train'], batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                sampler=SubsetRandomSampler(labeled_indices)),
            'test': DataLoader(dataset['test'], batch_size=1000, pin_memory=True, shuffle=True)
        }

        # Active learning cycles
        for cycle in range(args.num_cycle + 2):
            print('== CYCLE {} LabeledSize {}'.format(cycle + 1, len(labeled_indices)))
            # Model
            if args.use_features:
                model = mlp.MLP(num_classes=num_classes, model='others').to(device)
            else:
                model = resnet.ResNet18(num_classes=num_classes).to(device)
            if args.parallel:
                model = torch.nn.DataParallel(model)

            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.wdecay)
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.milestone])

            # Training and test
            train_acc, train_loss = train(model, criterion, optimizer, scheduler,
                                                    dataloaders, args, cycle)

            # test
            test_acc = test(model, dataloaders, mode='test')
            print(' TestAcc: {:.4f}'.format(test_acc))
            total[trial].append([len(labeled_indices), test_acc.item()])

            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_indices)

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(dataset['unlabeled'], batch_size=args.batch_size,
                                          sampler=sampler.SubsetSequentialSampler(unlabeled_indices),
                                          # more convenient if we maintain the order of subset
                                          pin_memory=True, shuffle=False)

            ### Random Query
            QuerySize = 200 if len(labeled_indices) < 1000 else args.query_size
            labeled_indices, unlabeled_indices = random_acquisition(labeled_indices, unlabeled_indices, QuerySize)

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(dataset['train'], batch_size=args.batch_size, pin_memory=True,
                                              sampler=SubsetRandomSampler(labeled_indices))

            # Save a checkpoint
            torch.save({
                'trial': trial + 1,
                'state_dict_backbone': model.state_dict(),
            },
                '{}/weights/{}-{}-trial{}.pth'.format(args.out_path, args.model_name,args.data, trial))


    ## save results
    utils.save_results(args, total)
    utils.alarm('DONE! {} {}'.format(args.model_name, args.data))