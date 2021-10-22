'''
Active Learning baselines

References:
    https://github.com/ej0cl6/deep-active-learning
    https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
'''
import os
import wandb
import argparse
import torch

from model import mlp, resnet, lossnet
from AL_Trainer import Trainer
from data.dataset import Dataset
import activelearner as al



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu cuda index')

    parser.add_argument('--al_method','-AL', type=str, default='random',
                        choices=['random','learningloss','coreset','badge','ws','vaal'])

    parser.add_argument('--dataset', help='dataset', type=str, default='CIFAR10')
    parser.add_argument('--data_dir', help='data path', type=str, default='D:/data/img_clf/')
    parser.add_argument('--use_features', action='store_true', default=True, help='use feature extracted from ImageNet')
    parser.add_argument('--pretrained', action='store_true', default=True, help='use feature extracted from ImageNet')

    parser.add_argument('--num_trial', type=int, default=10, help='number of trials')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=30, help='Batch size used for training only')
    parser.add_argument('--data_size', help='number of points at each round', type=list,
                        default=[600,800,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000])

    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='multistep', choices=['multistep', 'step'],
                        help='optimizer')
    parser.add_argument('--milestone', type=int, default=160, help='number of acquisition')

    parser.add_argument('--epochl', type=int, default=120,
                        help='After 120 epochs, stop the gradient from the loss prediction module propagated to the target model')
    parser.add_argument('--margin', type=float, default=1.0, help='MARGIN')
    parser.add_argument('--weights', '--w', type=float, default=0.0, help='weight')

    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix seed for reproducible')
    parser.add_argument('--seed', type=int, default=0, help='seed number')

    args = parser.parse_args()
    return args


def create_model(args):
    mlp_model = mlp.MLP(num_classes=args.nClass, al_method=args.al_method).to(args.device)
    resnet_model = resnet.ResNet18(num_classes=args.nClass).to(args.device)

    if args.al_method == 'learningloss':
        return {'backbone': mlp_model if args.use_features else resnet_model,
                'module': lossnet.LossNet().to(args.device)}
    elif args.use_features:
        return mlp_model
    else:
        return resnet_model

def active_learning_method(al_method):
    if al_method == 'random':
        from activelearner.random import RandomSampling
        return RandomSampling
    elif al_method == 'learningloss':
        return al.learningloss.LearningLoss
    elif al_method == 'coreset':
        return CoreSet
    elif al_method == 'badge':
        return BADGE
    elif al_method == 'ws':
        return WS
    elif al_method == 'vaal':
        return VAAL
    elif al_method == 'Proxy':
        return SelectionProxy
    elif al_method == 'SeqGCN':
        return SequentialGCN
    elif al_method == 'TAVAAL':
        return TAVAAL
    else:
        return 1


dataset_name = {'CIFAR10': 'CIFAR10', 'CIFAR100': 'CIFAR100',
                'FASHIONMNIST': 'FashionMNIST', 'CALTECH256': 'Caltech256'}


if __name__ == '__main__':
    # set up
    args = get_args()

    args.dataset = dataset_name[args.dataset.upper()]
    tmp = '-ftrs' if args.use_features else ''
    run = wandb.init(
        project=f'AL-{args.dataset}{tmp}',
        name=f"{args.al_method}-{args.optimizer},{args.num_epoch},{args.lr}",
        config=args
    )
    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)  # change allocation of current GPU
    print(f'Current cuda device: {torch.cuda.current_device()}')

    # set data
    dataset = Dataset(args)
    args.nClass = dataset.nClass
    args.nTrain, args.nTest = dataset.nTrain, dataset.nTest
    dataset = dataset.dataset
    print(f'Dataset: {args.dataset} Train {args.nTrain}')

    # save result
    args.results = []
    for trial in range(args.num_trial):
        print(f'>> TRIAL {trial}')

        # set active learner
        AL_method = active_learning_method(args.al_method)(dataset, args)
        # set dataloader for creating few labeled and lots of unlabeled data

        results = []
        for round in range(len(args.data_size)-1):
            nLabeled = args.data_size[round]
            nQuery = args.data_size[round+1] - args.data_size[round]
            print(f'>> round {round+1} Labeled {nLabeled} Query {nQuery}')

            # set model
            model = create_model(args)

            # Active Learning
            ## train
            trainer = Trainer(model, AL_method.dataloaders, args)
            train_acc = trainer.train()
            ## test
            test_acc = trainer.test()

            ## query
            labeled_indices, unlabeled_indices = AL_method.query(nQuery, trainer.model)

            ## save results
            results.append([nLabeled, test_acc])

        args.results.append(results)

    table = wandb.Table(data=torch.mean(torch.tensor(args.results), dim=1),
                        columns=['nLabeled', 'TestAcc'])
    wandb.log({'acc plot': wandb.plot.line(table, 'nLabeled', 'TestAcc',
                                           title=f'mean TestAcc on {args.dataset}')})





