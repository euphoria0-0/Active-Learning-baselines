import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#from tqdm import tqdm
from copy import deepcopy


class Trainer:
    def __init__(self, model, dataloaders, args):
        self.model = model.to(args.device)
        self.dataloaders = dataloaders
        self.device = args.device
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.al_method = args.al_method
        self.args = args

        # loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # optimizer
        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

        # learning rate scheduler
        if args.lr_scheduler == 'multistep':
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.milestone])
        else:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)


    def train(self):
        if self.al_method == 'learningloss':
            self.optimizer = {'backbone': self.optimizer, 'module': deepcopy(self.optimizer)}
            self.lr_scheduler = {'backbone': self.lr_scheduler, 'module': deepcopy(self.lr_scheduler)}

            self.model['backbone'].train()
            self.model['module'].train()

            for epoch in range(self.num_epoch): #tqdm(range(self.num_epoch), leave=True):
                train_acc = self.train_epoch_learningloss(epoch)
        else:
            self.model.train()

            for epoch in range(self.num_epoch):
                train_acc = self.train_epoch(epoch)
        
        return train_acc

    def train_epoch(self, epoch):
        train_loss, correct, total = 0., 0, 0

        for input, labels in self.dataloaders['train']:
            input, labels = input.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.criterion(output, labels.long())
            _, preds = torch.max(output.data, 1)

            train_loss += loss.item() * input.size(0)
            correct += torch.sum(preds == labels.data).cpu().data.numpy()
            total += input.size(0)

            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()

        self.lr_scheduler.step()

        train_acc = correct / total
        train_loss = train_loss / total

        sys.stdout.write('\rEpoch {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(epoch + 1, self.num_epoch,
                                                                                 train_loss, train_acc))
        sys.stdout.flush()
        torch.cuda.empty_cache()
        return train_acc


    def train_epoch_learningloss(self, epoch):
        train_loss, correct, total = 0., 0, 0

        for input, labels in self.dataloaders['train']:
            input, labels = input.to(self.device), labels.to(self.device)
            self.optimizer['backbone'].zero_grad()
            self.optimizer['module'].zero_grad()

            output, features = self.model['backbone'](input)

            target_loss = self.criterion(output, labels.long())
            _, preds = torch.max(output.data, 1)

            #train_loss += loss.item() * input.size(0)
            correct += torch.sum(preds == labels.data).cpu().data.numpy()
            total += input.size(0)

            if epoch > self.args.epoch_loss:
                # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                if len(features) == 4:
                    features[3] = features[3].detach()

            pred_loss = self.model['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss = self.model['module'](pred_loss, target_loss, margin=1.0)
            loss = m_backbone_loss + self.args.weights * m_module_loss

            loss.backward()
            self.optimizer['backbone'].step()
            self.optimizer['module'].step()

            torch.cuda.empty_cache()

        self.lr_scheduler['backbone'].step()
        self.lr_scheduler['module'].step()

        train_loss = loss.item()
        train_acc = correct / total

        sys.stdout.write('\rEpoch {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(epoch + 1, self.num_epoch,
                                                                                 train_loss, train_acc))
        sys.stdout.flush()
        torch.cuda.empty_cache()
        return train_acc


    def test(self, phase='test'):
        self.model.eval()

        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0
            for input, labels in self.dataloaders[phase]:
                input, labels = input.to(self.device), labels.to(self.device)
                output = self.model(input)
                #loss = self.criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                #test_loss += loss.item() * input.size(0)
                correct += preds.eq(labels).sum().cpu().data.numpy()
                total += input.size(0)

                torch.cuda.empty_cache()

        test_acc = correct / total
        print(' TestAcc: {:.4f}'.format(test_acc))
        torch.cuda.empty_cache()
        return test_acc

    def get_model(self):
        return self.model
