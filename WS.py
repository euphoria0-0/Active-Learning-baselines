from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse
import logging
import time
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime

import custom_data.WS_data_loader as data_loader
from models import mlp
from utils.tensorboard_logger.tensorboard_logger import configure, unconfigure, log_value
from utils import active_util
from utils import utils


# Training
def train(logger, train_loader, model, criterion, optimizer, epoch, episode_i):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets.long())

        # measure accuracy and record loss
        acc1 = accuracy(outputs.data, targets, topk=(1,))[0]
        losses.update(loss.data, inputs.size(0))
        top1.update(acc1, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % args.print_freq == 0:
    logger.info('Epoch: [{0}][{1}/{2}] '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                'acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        epoch, i, len(train_loader), batch_time=batch_time,
        loss=losses, top1=top1))

    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, episode_i * args.epochs + epoch)
        log_value('train_acc', top1.avg, episode_i * args.epochs + epoch)


def test(logger, test_loader, model, criterion, epoch, episode_i, state='test'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    entropy_total = []
    confusion_matrix = torch.zeros(args.num_classes, args.num_classes)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets.long())

            # measure accuracy and record loss
            acc1 = accuracy(outputs.data, targets, topk=(1,))[0]
            losses.update(loss.data, inputs.size(0))
            top1.update(acc1, inputs.size(0))

            # compute class separate accuracy
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # compute entropy for active learning
            if state == 'entropy':
                entropy = F.softmax(outputs.data, dim=1) * F.log_softmax(outputs.data, dim=1)
                entropy_total.extend(-1.0 * entropy.sum(dim=1))

                # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('{0}: [{1}/{2}] '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                            'acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    state, i, len(test_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

        accuracy_each_class = confusion_matrix.diag() / confusion_matrix.sum(1)
        logger.info(' * acc@1 {top1.avg:.3f}'.format(top1=top1))
        logger.info(' * accuracy of each classes: \n  {}'.format([round(x,4) for x in accuracy_each_class.tolist()]))

    # log to TensorBoard
    if args.tensorboard and state == 'test':
        log_value('val_loss', losses.avg, episode_i * args.epochs + epoch)
        log_value('val_acc', top1.avg, episode_i * args.epochs + epoch)

    if state == 'entropy':
        return torch.stack(entropy_total).cpu()
    else:
        return top1.avg


def save_checkpoint(state, episode_i, save_dir, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    filename = '{}/checkpoint_episode_{}.pth.tar'.format(save_dir, episode_i)
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, episode_i):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 250 epochs"""
    if epoch in args.epoch_step:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay
            args.lr = param_group['lr']

    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', args.lr, episode_i * args.epochs + epoch)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def episode(logger, save_dir, episode_i, norms, labeled_idx, sampler, train_eval_loader, test_loader, model):
    best_acc = 0

    # Data
    train_loader, _ = data_loader.getDataSet(
        args.dataset, args.batch_size, args.dataroot, sampler=sampler, drop_last=args.drop_last,
        num_workers=args.num_workers)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.decay_type == 'rational':
        weight_decay = args.weight_decay / (episode_i + 1)
    elif args.decay_type == 'fix':
        weight_decay = args.weight_decay
    else:
        raise NotImplementedError()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=weight_decay)

    if episode_i == 0:
        norms = active_util.save_weight_distribution(args, save_dir, model, -1, norms, save=args.save_fig,
                                                     xlim=args.xlim, ylim=args.ylim)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, episode_i)

        train(logger, train_loader, model, criterion, optimizer, epoch, episode_i)

        if epoch % args.test_freq == 0 or epoch == args.epochs - 1:
            acc = test(logger, test_loader, model, criterion, epoch, episode_i)

    logger.info('Best accuracy: %f' % best_acc)
    logger.info('Fianal accuracy: %f' % acc)
    if args.tensorboard:
        log_value('episode accuracy ', acc, episode_i)

    entropy = test(logger, train_eval_loader, model, criterion, epoch, episode_i, state='entropy')

    # save weight distribution
    norms = active_util.save_weight_distribution(args, save_dir, model, episode_i, norms, save=args.save_fig,
                                                 xlim=args.xlim, ylim=args.ylim)

    return entropy, acc.item(), norms



def main(args, seed):
    args.start = datetime.now()
    save_dir = args.save_dir + '/' + str(seed)
    os.makedirs(save_dir, exist_ok=True)
    os.chmod(save_dir, 0o777)

    if args.tensorboard: configure(save_dir)

    logger = logging.getLogger("js_logger")
    fileHandler = logging.FileHandler(save_dir + '/train.log')
    streamHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.INFO)
    logger.info(args)

    lr = args.lr
    if args.dataset == 'cifar10' or args.dataset == 'mnist':
        args.num_classes = 10
        len_dataset = 50000
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        len_dataset = 50000
    elif args.dataset == 'FashionMNIST':
        args.num_classes = 10
        len_dataset = 60000
    elif args.dataset == 'Caltech256':
        args.num_classes = 257
        len_dataset = 22897

    train_eval_loader, test_loader = data_loader.getDataSet(args.dataset + '_eval', 1000, args.dataroot,
                                                            num_workers=args.num_workers)
    QuerySize = args.increment

    labeled_idx = np.random.choice(len_dataset, args.init_size, replace=False)
    num_episode = int(args.max_data / args.increment) + 2
    num_each_class = active_util.class_count(train_eval_loader, labeled_idx, args.num_classes)
    logger.info('sample of each class:\n\t{}'.format(num_each_class))

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and args.cudnn:
        cudnn.benchmark = True

    if args.load_dir:
        if os.path.isdir(args.load_dir):
            # Load checkpoint.
            print("=> loading checkpoint '{}'".format(args.load_dir))
            checkpoint = torch.load(args.load_dir + '/' + args.checkpoint)
            args.start_episode = checkpoint['episode_i']
            labeled_idx = checkpoint['labeled_idx']
        else:
            print("=> no checkpoint found at '{}'".format(args.load_dir))

    num_labeled_lst = []
    episode_accs = []
    norms = None
    for episode_i in range(args.start_episode, num_episode):

        args.increment = 200 if len(labeled_idx) < 1000 else QuerySize
        print('======= LEN LABELED SET {} ======='.format(len(labeled_idx)))
        num_labeled_lst.append(len(labeled_idx))

        # model_to_call = getattr(models, args.net_type)
        # model = model_to_call(num_classes=args.num_classes)
        model = mlp.MLP(num_classes=args.num_classes)
        model = model.to(device)
        model = torch.nn.DataParallel(model)

        sampler = torch.utils.data.sampler.SubsetRandomSampler(labeled_idx)
        entropy, episode_acc, norms = episode(
            logger, save_dir, episode_i, norms, labeled_idx, sampler, train_eval_loader, test_loader, model)
        episode_accs.append(episode_acc)

        if len(labeled_idx) <= len_dataset - args.increment and episode_i != num_episode - 1:
            labeled_idx = active_util.entropy_sampler(labeled_idx, len_dataset, entropy, args.increment,
                                                      type=args.sampling)
            num_each_class = active_util.class_count(train_eval_loader, labeled_idx, args.num_classes)
            logger.info('sample of each class:\n\t{}'.format(num_each_class))
        args.lr = lr

        # save checkpoint
        if args.save_model:
            save_checkpoint({
                'episode_i': episode_i + 1,
                'labeled_idx': labeled_idx,
                'norms': norms,
                'model': model.state_dict(),
            }, episode_i, save_dir)

    logger.info('episode accuracy:\n\t{}'.format(episode_accs))

    splits = [*range(args.increment, args.max_data+args.increment, args.increment)]
    if args.init_size < 1000:
        splits = [0] + [args.init_size+200*i for i in range(10) if args.init_size + 200*i < 1000] + splits
    log_file = pd.DataFrame(norms, index=splits)
    log_file.transpose().to_excel(save_dir + '/log_file.xlsx')

    logger.removeHandler(fileHandler)
    logger.removeHandler(streamHandler)
    logging.shutdown()
    del logger, fileHandler, streamHandler
    if args.tensorboard: unconfigure()

    return num_labeled_lst, np.array(episode_accs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Weight Decay Scheduling')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='number_workers in data_loader')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--start_episode', default=0, type=int,
                        help='manual episode number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--epoch-step', type=int, nargs='+', default=[150, 225],
                        help='Learning Rate Decay Steps')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--lr_decay', default=0.1, type=float,
                        help='learning rate decay ratio')
    parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                        help='weight decay')
    parser.add_argument('--decay_type', default='rational', type=str,
                        help='weight decay decay method: rational, fix')
    parser.add_argument('--net_type', default='mlp',
                        help='resnet18, densnetBC100, mlp')
    parser.add_argument('--dataset', default='cifar10',
                        help='cifar10 | cifar100')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='number of classes (automatically setted in main function)')
    parser.add_argument('--dataroot', default='../dataset/',
                        help='path to dataset')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--test_freq', default=20, type=int,
                        help='test frequency (default: 10)')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='whether to use standard augmentation (default: True)')
    parser.add_argument('--save_dir', type=str, default='./results/WS/',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--save_model', default=False, action='store_true',
                        help='If true, save model every episode')
    parser.add_argument('--checkpoint', default='checkpoint.pth.tar', type=str,
                        help='checkpoint file name')
    parser.add_argument('-l', '--load_dir', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no_tensorboard', dest='tensorboard', action='store_false',
                        help='Whether to log progress to TensorBoard')
    parser.add_argument('--drop_last', default=False, action='store_true',
                        help='If true, drop last batch in data loader')
    parser.add_argument('--increment', type=int, default=1000,
                        help='data increment size')
    parser.add_argument('--max_data', type=int, default=15000,
                        help='data increment size. should be divided by increment')
    parser.add_argument('--sampling', type=str, default='descending',
                        help='descending | random')
    parser.add_argument('--seed', type=int, nargs='+', default=[0,1,2,3,4],
                        help='random seed')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha for distillation loss')
    parser.add_argument('--T', type=float, default=6,
                        help='Temperature for distillation loss')
    parser.add_argument('--xlim', type=float, default=None,
                        help='xlim for plotting weight distribution')
    parser.add_argument('--ylim', type=int, default=None,
                        help='ylim for plotting weight distribution')
    parser.add_argument('--save_fig', default=True, action='store_true',
                        help='If true, save weight distribution pdf and L2 norm of each layer')
    parser.add_argument('--cudnn', default=True, action='store_false',
                        help='use cudnn benchmark')
    parser.add_argument('--init_size', type=int, default=600,
                        help='initial data size')
    parser.set_defaults(tensorboard=True)
    args = parser.parse_args()

    if torch.cuda.is_available() and args.cudnn:
        print('Current cuda device ', torch.cuda.current_device())

    os.makedirs(args.save_dir + args.dataset + '/' + args.net_type + '/inc_' + str(args.increment) + '_batch_' + str(
        args.batch_size), exist_ok=True)
    os.chmod(args.save_dir, 0o777)
    os.chmod(args.save_dir + args.dataset, 0o777)
    os.chmod(args.save_dir + args.dataset + '/' + args.net_type, 0o777)
    os.chmod(args.save_dir + args.dataset + '/' + args.net_type + '/inc_' + str(args.increment) + '_batch_' + str(
        args.batch_size), 0o777)

    args.out_path = '{}{}/'.format(args.save_dir, args.dataset)

    args.save_dir = '{}{}/{}/inc_{}_batch_{}/WD_{}_decay_{}_{}_no_init_lr{}_epoch{}_imax{}_seed{}_'.format(
        args.save_dir, args.dataset, args.net_type, args.increment, args.batch_size, \
        args.weight_decay, args.decay_type, args.sampling, args.lr, args.epochs, args.max_data, \
        str(args.seed).replace("[", "").replace("]", "").replace(",", "_"))

    i = 0
    while os.path.isdir(args.save_dir + str(i)):
        i += 1
    args.save_dir = args.save_dir + str(i)
    if args.load_dir:
        args.save_dir = args.load_dir
    os.makedirs(args.save_dir, exist_ok=True)
    os.chmod(args.save_dir, 0o777)

    splits = [*range(args.increment, args.max_data + args.increment, args.increment)]
    if args.init_size < 1000:
        splits = [args.init_size + 200 * i for i in range(10) if args.init_size + 200 * i < 1000] + splits
    args.num_trial = len(args.seed)

    if args.seed is not None:
        total = [[] for _ in range(len(args.seed))]
        seeds_accs = []
        for trial, seed in enumerate(args.seed):
            print('==== TRIAL {} seed {} ===='.format(trial, seed))
            np.random.seed(seed)
            #train
            labeled, episode_accs = main(args, seed)
            seeds_accs.append(episode_accs)
            total[trial] = [[int(x),round(y*0.01, 4)] for x,y in zip(labeled,episode_accs)]

        average_acc = np.average(seeds_accs, 0)
        std_acc = np.std(seeds_accs, 0)
        seeds_accs.append(average_acc)
        seeds_accs.append(std_acc)
        args.seed.append('average')
        args.seed.append('std')
        avg_acc_file = pd.DataFrame(seeds_accs, columns=splits, index=args.seed)
        avg_acc_file.to_excel(args.save_dir + '/avg_acc_file.xlsx')

        os.makedirs(args.save_dir + '/' + str(average_acc).replace("[", "").replace("]", "").replace(",", " "))

    args.model_name = 'WS'
    args.data = args.dataset
    utils.alarm('DONE! WS {}'.format(args.dataset))
    utils.save_results(args, total)