'''
Learning Loss for Active Learning

Source:
    https://github.com/Mephisto405/Learning-Loss-for-Active-Learning

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
from utils import utils #, alarm


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
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--num_trial', type=int, default=5, help='number of trials')
    parser.add_argument('--num_cycle', type=int, default=15, help='number of acquisition')
    parser.add_argument('--init_size', type=int, default=600, help='size of initial pool')
    parser.add_argument('--query_size', type=int, default=1000, help='number of querrying data')
    parser.add_argument('--addendum', type=int, default=1000, help='seed number')
    parser.add_argument('--subset', type=int, default=10000, help='seed number')

    parser.add_argument('--margin', type=float, default=1.0, help='MARGIN')
    parser.add_argument('--weights','--w', type=float, default=0.0, help='weight')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='wdecay for SGD')
    parser.add_argument('--milestone', type=int, default=160, help='number of acquisition')
    parser.add_argument('--epochl', type=int, default=120, help='After 120 epochs, stop the gradient from the loss prediction module propagated to the target model')

    parser.add_argument('--use_features', action='store_true', default=False, help='use feature extracted from ImageNet')
    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix seed for reproducible')
    parser.add_argument('--seed', type=int, default=0, help='seed number')

    args = parser.parse_args()

    return args


#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()

    total, correct = 0, 0
    for (inputs, labels) in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device).long()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        _, preds = torch.max(scores.data, 1)
        total += labels.size(0)
        correct += torch.sum(preds == labels.data)

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            if len(features) == 4:
                features[3] = features[3].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = lossnet.LossPredLoss(pred_loss, target_loss, margin=1.0)
        loss = m_backbone_loss + args.weights * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

    return models, (correct.double() / total).cpu().data.numpy(), [m_backbone_loss.item(),m_module_loss.item(),loss.item()]


#
def test(models, dataloaders, mode='test'):
    models['backbone'].eval()
    models['module'].eval()

    total, correct = 0, 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs, labels = inputs.to(device), labels.to(device).long()

            scores, features = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)

    #return (correct.double() / total).cpu().data.numpy()
    return correct.double() / total


#
def train(models, criterion, optimizers, schedulers, dataloaders, args, cycle=0):
    checkpoint_dir = os.path.join('./results/Learning-Loss/', args.data, 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(args.num_epoch):
        models, train_acc, (bbloss, mdloss, loss) = train_epoch(models, criterion, optimizers, dataloaders,
                                                                epoch, args.epochl)

        schedulers['backbone'].step()
        schedulers['module'].step()

        sys.stdout.write('\rEpoch {}/{} TrainLoss {:.4f} TrainAcc {:.4f}'.format(epoch+1, args.num_epoch, loss, train_acc))
        sys.stdout.flush()

        if (args.vis != None) and (args.plot_data != None):
            utils.vis_plot(epoch+cycle*args.num_epoch,
                           [bbloss, mdloss, loss, train_acc, test_acc],
                           args.vis, args.plot_data, ymax=3)

    return train_acc, loss


#
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).to(device)

    with torch.no_grad():
        for (inputs, _) in unlabeled_loader:
            inputs = inputs.to(device)

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()



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
        args.plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss',
                                                  'Train Acc', 'Test Acc']}
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
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(train_size))
        random.shuffle(indices)
        labeled_indices = indices[:args.addendum]
        unlabeled_indices = indices[args.addendum:]

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
                backbone_model = mlp.MLP(num_classes=num_classes, model='ll').to(device)
                loss_module = lossnet.LossNet_MLP().to(device)
            else:
                backbone_model = resnet.ResNet18(num_classes=num_classes, ll=True).to(device)
                loss_module = lossnet.LossNet().to(device)
            if args.parallel:
                backbone_model = torch.nn.DataParallel(backbone_model)
                loss_module = torch.nn.DataParallel(loss_module)

            models = {'backbone': backbone_model, 'module': loss_module}

            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.wdecay)
            optim_module = optim.SGD(models['module'].parameters(), lr=args.lr,
                                     momentum=args.momentum, weight_decay=args.wdecay)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=[args.milestone])
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=[args.milestone])

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            train_acc, train_loss = train(models, criterion, optimizers, schedulers,
                                                    dataloaders, args, cycle)

            # test
            test_acc = test(models, dataloaders, mode='test')
            print(' TestAcc: {:.4f}'.format(test_acc))
            total[trial].append([len(labeled_indices), train_loss, train_acc, test_acc.item()])

            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_indices)
            subset = unlabeled_indices[:args.subset]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(dataset['unlabeled'], batch_size=args.batch_size,
                                          sampler=sampler.SubsetSequentialSampler(subset),
                                          # more convenient if we maintain the order of subset
                                          pin_memory=True, shuffle=False)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)

            # Update the labeled dataset and the unlabeled dataset, respectively
            QuerySize = 200 if len(labeled_indices) < 1000 else args.query_size

            labeled_indices += list(torch.tensor(subset)[arg][-QuerySize:].numpy())
            unlabeled_indices = list(torch.tensor(subset)[arg][:-QuerySize].numpy()) + unlabeled_indices[args.subset:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(dataset['train'], batch_size=args.batch_size, pin_memory=True,
                                              sampler=SubsetRandomSampler(labeled_indices))

            # Save a checkpoint
            torch.save({
                'trial': trial + 1,
                'state_dict_backbone': models['backbone'].state_dict(),
                'state_dict_module': models['module'].state_dict()
            },
                '{}/weights/{}-{}-trial{}.pth'.format(args.out_path, args.model_name,args.data, trial))


    ## save results
    utils.save_results(args, total)
    utils.alarm('DONE! {} {}'.format(args.model_name, args.data))