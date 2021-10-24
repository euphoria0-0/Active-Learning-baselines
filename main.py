'''
Active Learning baselines

References:
    https://github.com/ej0cl6/deep-active-learning
    https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
'''
import sys

import wandb
import argparse
import torch

from model import mlp, resnet
from data.dataset import Dataset



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

    parser.add_argument('--batch_size','-b', type=int, default=30, help='Batch size used for training only')
    parser.add_argument('--data_size', help='number of points at each round', type=list,
                        default=[600,800,1000,2000])#,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000])

    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='multistep', choices=['multistep', 'step', None],
                        help='optimizer')
    parser.add_argument('--milestone', type=int, default=160, help='number of acquisition')

    parser.add_argument('--epoch_loss', type=int, default=120,
                        help='After 120 epochs, stop the gradient from the loss prediction module propagated to the target model')
    parser.add_argument('--margin', type=float, default=1.0, help='MARGIN')
    parser.add_argument('--weights','-w', type=float, default=0.0, help='weight')

    parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=1,
                        help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2,
                        help='Number of VAE steps taken for every task model step')
    parser.add_argument('--adversary_param', type=float, default=1,
                        help='Hyperparameter for training. lambda2 in the paper')

    parser.add_argument('--subset', type=int, default=10000, help='subset for learning loss')

    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix seed for reproducible')
    parser.add_argument('--seed', type=int, default=0, help='seed number')

    parser.add_argument('--comment', type=str, default='', help='comment')

    args = parser.parse_args()
    return args


def active_learning_method(al_method):
    if al_method == 'random':
        from activelearner.random import RandomSampling
        return RandomSampling
    elif al_method == 'learningloss':
        from activelearner.learningloss import LearningLoss
        return LearningLoss
    elif al_method == 'coreset':
        from activelearner.coreset import CoreSet
        return CoreSet
    elif al_method == 'badge':
        from activelearner.badge import BADGE
        return BADGE
    elif al_method == 'vaal':
        from activelearner.vaal import VAAL
        return VAAL
    elif al_method == 'ws':
        return WS
    elif al_method == 'Proxy':
        return SelectionProxy
    elif al_method == 'SeqGCN':
        return SequentialGCN
    elif al_method == 'TAVAAL':
        return TAVAAL
    else:
        return 1

def create_model(args):
    if args.use_features:
        return mlp.MLP(num_classes=args.nClass, al_method=args.al_method).to(args.device)
    else:
        return resnet.ResNet18(num_classes=args.nClass).to(args.device)

def trainer_method(al_method):
    if al_method == 'learningloss':
        from trainer.trainer_ll import Trainer
    elif al_method == 'vaal':
        from trainer.trainer_vaal import Trainer
    else:
        from trainer.trainer import Trainer
    return Trainer



dataset_name = {'CIFAR10': 'CIFAR10', 'CIFAR100': 'CIFAR100',
                'FASHIONMNIST': 'FashionMNIST', 'CALTECH256': 'Caltech256'}


if __name__ == '__main__':
    # set up
    args = get_args()
    args.dataset = dataset_name[args.dataset.upper()]

    ftrs = '-ftrs' if args.use_features else ''
    comm = '-'+args.comment if args.comment != '' else ''
    wandb.init(
        project=f'AL-{args.dataset}{ftrs}',
        name=f"{args.al_method}{comm}",#-{args.optimizer},{args.num_epoch},{args.lr}",
        config=args
    )

    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)  # change allocation of current GPU
    print(f'Current cuda device: {torch.cuda.current_device()}')

    # save results
    table = wandb.Table(columns=['Trial', 'nLabeled', 'TestAcc'])
    results = []

    # set data
    dataset = Dataset(args)
    args.nClass, args.nTrain, args.nTest = dataset.nClass, dataset.nTrain, dataset.nTest
    dataset = dataset.dataset
    print(f'Dataset: {args.dataset}')
    print(f'Train {args.nTrain} = Labeled {args.data_size[0]} + Unlabeled {args.nTrain - args.data_size[0]}')

    for trial in range(args.num_trial):
        print(f'>> TRIAL {trial+1}/{args.num_trial}')

        # set active learner and dataloader
        AL_method = active_learning_method(args.al_method)(dataset, args)

        results_trial = []
        for round in range(len(args.data_size)):
            nLabeled = args.data_size[round]
            if round < len(args.data_size) - 1:
                nQuery = args.data_size[round+1] - args.data_size[round]
                print(f'> round {round+1}/{len(args.data_size)} Labeled {nLabeled} Query {nQuery}')
            else:
                print(f'> round {round+1}/{len(args.data_size)} Labeled {nLabeled}')

            # set model
            model = create_model(args)

            # Active Learning
            ## train
            trainer = trainer_method(args.al_method)(model, AL_method.dataloaders, args)
            train_acc = trainer.train()
            ## test
            test_acc = trainer.test()

            ## query
            if round < len(args.data_size) - 1:
                AL_method.query(nQuery, trainer.model)

            ## save results
            results_trial.append(test_acc)
            table.add_data(trial, nLabeled, test_acc)

            wandb.log({
                f'Test/Acc-{trial+1}': test_acc
            })

        results.append(results_trial)

    for acc in torch.mean(torch.tensor(results), dim=0).tolist():
        wandb.log({
            'Result/meanTestAcc': acc
        })

    artifacts = wandb.Artifact(f'{args.dataset}_{args.al_method}_results', type='predictions')
    artifacts.add(table, 'test_results')
    wandb.log_artifact(artifacts)
