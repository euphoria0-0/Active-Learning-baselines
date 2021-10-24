import sys
import torch
import torch.nn as nn
import torch.optim as optim
#from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataloaders, args):
        self.model = model
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
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wdecay)

        # learning rate scheduler
        if args.lr_scheduler == 'multistep':
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.milestone])
        elif args.lr_scheduler == 'step':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        else: # equal to None
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.num_epoch])


    def train(self):
        self.model.train()
        for epoch in range(self.num_epoch):
            train_acc = self.train_epoch(epoch)
        return train_acc

    def train_epoch(self, epoch):
        train_loss, correct, total = 0., 0, 0

        for input, labels in self.dataloaders['train']:
            input, labels = input.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output, _ = self.model(input)
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

    def test(self, phase='test'):
        self.model.eval()

        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0
            for input, labels in self.dataloaders[phase]:
                input, labels = input.to(self.device), labels.to(self.device)
                output, _ = self.model(input)
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

