import sys
import torch
import torch.nn as nn
import torch.optim as optim
#from tqdm import tqdm
from model import lossnet


class Trainer:
    def __init__(self, model, dataloaders, args):
        self.device = args.device
        self.model = {'backbone': model}
        if args.use_features:
            self.model['module'] = lossnet.LossNet_MLP().to(args.device)
        else:
            self.model['module'] = lossnet.LossNet.to(args.device)
        self.dataloaders = dataloaders
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.al_method = args.al_method
        self.args = args

        # loss function
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        self.optimizer = {
            'backbone': optim.SGD(self.model['backbone'].parameters(), lr=args.lr, weight_decay=args.wdecay),
            'module': optim.SGD(self.model['module'].parameters(), lr=args.lr, weight_decay=args.wdecay)}

        self.lr_scheduler = {
            'backbone': optim.lr_scheduler.StepLR(self.optimizer['backbone'], step_size=10, gamma=0.9),
            'module': optim.lr_scheduler.StepLR(self.optimizer['module'], step_size=10, gamma=0.9)}


    def train(self):
        self.model['backbone'].train()
        self.model['module'].train()

        for epoch in range(self.num_epoch): #tqdm(range(self.num_epoch), leave=True):
            train_acc = self.train_epoch(epoch)

        return train_acc

    def train_epoch(self, epoch):
        train_loss, correct, total = 0., 0, 0

        for input, labels, _ in self.dataloaders['train']:
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
            m_module_loss = lossnet.LossPredLoss(pred_loss, target_loss, margin=1.0)
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
        self.model['backbone'].eval()
        self.model['module'].eval()

        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0
            for input, labels, _ in self.dataloaders[phase]:
                input, labels = input.to(self.device), labels.to(self.device)
                output, _ = self.model['backbone'](input)
                _, preds = torch.max(output.data, 1)

                correct += preds.eq(labels).sum().cpu().data.numpy()
                total += input.size(0)

                torch.cuda.empty_cache()

        test_acc = correct / total
        print(' TestAcc: {:.4f}'.format(test_acc))
        torch.cuda.empty_cache()
        return test_acc

    def get_model(self):
        return self.model