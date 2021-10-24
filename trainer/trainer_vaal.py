import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#from tqdm import tqdm
from model import VAE
from copy import deepcopy



def read_data(dataloader):
    while True:
        for img, label, _ in dataloader:
            if img is None:
                sys.exit(0)
            yield img, label

class Trainer:
    def __init__(self, model, dataloaders, args):
        self.dataloaders_ftrs, self.dataloaders = dataloaders
        self.device = args.device
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.al_method = args.al_method
        self.args = args

        # create model
        self.task_model = model
        if args.dataset == 'FashionMNIST':
            self.vae = VAE.VAE(args.latent_dim, nc=1).to(self.device)
        elif args.dataset == 'Caltech256':
            self.vae = VAE.VAE_Caltech(args.latent_dim).to(self.device)
        else:
            self.vae = VAE.VAE(args.latent_dim).to(self.device)
        self.discriminator = VAE.Discriminator(args.latent_dim).to(self.device)

        # loss function
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # optimizer
        if args.optimizer == 'sgd':
            self.optim_task_model = optim.SGD(self.task_model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        else:
            self.optim_task_model = optim.Adam(self.task_model.parameters(), lr=self.args.lr, weight_decay=args.wdecay)
        self.optim_vae = optim.Adam(self.vae.parameters(), lr=5e-4)
        self.optim_discriminator = optim.Adam(self.discriminator.parameters(), lr=5e-4)

        self.args.train_iterations = (self.args.nTrain * self.args.num_epoch) // self.args.batch_size
        self.lr_change = self.args.train_iterations // 4

        self.labeled_data = read_data(self.dataloaders['train'])
        self.unlabeled_data = read_data(self.dataloaders['unlabeled'])

        if self.args.use_features:
            self.ftrs_labeled_data = read_data(self.dataloaders_ftrs['train'])


    def train(self):
        self.vae.train()
        self.discriminator.train()
        self.task_model.train()

        best_acc = 0
        for iter_count in range(self.args.train_iterations):
            if iter_count > 0 and iter_count % self.lr_change == 0:
                for param in self.optim_task_model.param_groups:
                    param['lr'] = param['lr'] / 10

            labeled_imgs, labels, unlabeled_imgs = self._next_imgs()

            # task step
            if self.args.use_features:
                labeled_ftrs, labels_ftrs = next(self.ftrs_labeled_data)
                labeled_ftrs, labels_ftrs = labeled_ftrs.to(self.device), labels_ftrs.to(self.device)
                train_acc = self._task_model_train_step(labeled_ftrs, labels_ftrs)
            else:
                train_acc = self._task_model_train_step(labeled_imgs, labels)

            # VAE step
            for count in range(self.args.num_vae_steps):
                self._vae_model_train_step(labeled_imgs, unlabeled_imgs)
                if count < (self.args.num_vae_steps - 1):
                    labeled_imgs, _, unlabeled_imgs = self._next_imgs()

            # Discriminator step
            for count in range(self.args.num_adv_steps):
                self._discriminator_model_train_step(labeled_imgs, unlabeled_imgs)
                if count < (self.args.num_adv_steps - 1):
                    labeled_imgs, _, unlabeled_imgs = self._next_imgs()

            sys.stdout.write(
                '\rtrain iter {}/{} | task loss {:.4f} vae loss {:.4f} discriminator loss {:.4f}'.format(
                    iter_count + 1, self.args.train_iterations, self.task_loss.item(), self.total_vae_loss.item(),
                    self.dsc_loss.item()))
            sys.stdout.flush()

            if (iter_count % 300 == 0 and iter_count > 0) or iter_count == self.args.train_iterations - 1:

                if train_acc > best_acc:
                    best_acc = train_acc
                    best_task_model = deepcopy(self.task_model)
                    best_vae = deepcopy(self.vae)
                    best_discriminator = deepcopy(self.discriminator)

                if iter_count % 30000 == 0 or iter_count == self.args.train_iterations - 1:
                    print('\ncurrent step: {} acc: {:.4f} best acc: {:.4f}'.format(iter_count, train_acc, best_acc))

        self.task_model = best_task_model
        self.model = [best_vae, best_discriminator]

        torch.cuda.empty_cache()

        return best_acc


    def _next_imgs(self):
        # sample new batch if needed to train the adversarial network
        labeled_imgs, labels = next(self.labeled_data)
        unlabeled_imgs, _ = next(self.unlabeled_data)

        labeled_imgs, labels = labeled_imgs.to(self.device), labels.to(self.args.device)
        unlabeled_imgs = unlabeled_imgs.to(self.args.device)
        return labeled_imgs, labels, unlabeled_imgs

    def test(self, phase='test'):
        test_loader = self.dataloaders_ftrs if self.args.use_features else self.dataloaders
        self.task_model = self.task_model.to(self.device)
        self.task_model.eval()

        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0
            for input, labels, _ in test_loader[phase]:
                input, labels = input.to(self.device), labels.to(self.device)
                output, _ = self.task_model(input)
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

    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD


    def _task_model_train_step(self, input, labels):
        output, _ = self.task_model(input)
        self.task_loss = self.ce_loss(output, labels.long())
        _, preds = torch.max(output.data, 1)
        correct = preds.eq(labels).sum().cpu().data.numpy()
        total = input.size(0)

        self.optim_task_model.zero_grad()
        self.task_loss.backward()
        self.optim_task_model.step()
        return correct / total

    def _vae_model_train_step(self, input, unlabeled_input):
        recon, z, mu, logvar = self.vae(input)
        unsup_loss = self.vae_loss(input, recon, mu, logvar, self.args.beta)
        unlab_recon, unlab_z, unlab_mu, unlab_logvar = self.vae(unlabeled_input)
        transductive_loss = self.vae_loss(unlabeled_input,
                                          unlab_recon, unlab_mu, unlab_logvar, self.args.beta)

        labeled_preds = self.discriminator(mu)
        unlabeled_preds = self.discriminator(unlab_mu)

        lab_real_preds = torch.ones(input.size(0)).to(self.device)
        unlab_real_preds = torch.ones(unlabeled_input.size(0)).to(self.device)

        dsc_loss = self.bce_loss(labeled_preds.view(-1), lab_real_preds) + \
                   self.bce_loss(unlabeled_preds.view(-1), unlab_real_preds)
        self.total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
        del unsup_loss
        del transductive_loss
        del dsc_loss

        self.optim_vae.zero_grad()
        self.total_vae_loss.backward()
        self.optim_vae.step()

    def _discriminator_model_train_step(self, input, unlabeled_input):
        with torch.no_grad():
            _, _, mu, _ = self.vae(input)
            _, _, unlab_mu, _ = self.vae(unlabeled_input)

        labeled_preds = self.discriminator(mu)
        unlabeled_preds = self.discriminator(unlab_mu)

        lab_real_preds = torch.ones(input.size(0)).to(self.args.device)
        unlab_fake_preds = torch.zeros(unlabeled_input.size(0)).to(self.args.device)

        self.dsc_loss = self.bce_loss(labeled_preds.view(-1), lab_real_preds) + \
                   self.bce_loss(unlabeled_preds.view(-1), unlab_fake_preds)

        self.optim_discriminator.zero_grad()
        self.dsc_loss.backward()
        self.optim_discriminator.step()









