import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

import copy
#from utils import utils
import utils
from utils import utils
from models import mlp, resnet, VAE


### sampler
class AdversarySampler:
    def __init__(self, query_size, device):
        self.query_size = query_size
        self.device = device

    def sample(self, vae, discriminator, data):
        all_preds = []
        all_indices = []

        for images, _, indices in data:
            images = images.to(self.device)

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.query_size))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices


### solver
class Solver:
    def __init__(self, args, test_dataloader, feature_test_dataloader=None):
        self.args = args
        self.test_dataloader = test_dataloader
        if self.args.use_features:
            self.feature_test_dataloader = feature_test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = AdversarySampler(self.args.query_size, self.args.device)

    def read_data(self, dataloader):
        while True:
            for img, label, _ in dataloader:
                if img is None:
                    sys.exit(0)
                yield img, label


    def train(self, querry_dataloader, unlabeled_dataloader,
              feature_train_dataloader=None):

        self.args.train_iterations = (self.args.num_images * self.args.num_epoch) // self.args.batch_size
        lr_change = self.args.train_iterations // 4
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader)

        if self.args.use_features:
            feature_data = self.read_data(feature_train_dataloader)
            task_model = mlp.MLP(num_classes=self.args.num_classes, model='vaal').to(self.args.device)
        else:
            task_model = resnet.ResNet18(num_classes=self.args.num_classes).to(self.args.device)

        if self.args.data == 'FashionMNIST':
            vae = VAE.VAE(self.args.latent_dim, nc=1).to(self.args.device)
        elif self.args.data == 'Caltech256':
            vae = VAE.VAE_Caltech(self.args.latent_dim).to(self.args.device)
        else:
            vae = VAE.VAE(self.args.latent_dim).to(self.args.device)

        discriminator = VAE.Discriminator(self.args.latent_dim).to(self.args.device)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_task_model = optim.SGD(task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        vae.train()
        discriminator.train()
        task_model.train()

        best_acc = 0
        for iter_count in range(self.args.train_iterations):

            if iter_count is not 0 and iter_count % lr_change == 0:
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] / 10
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs, _ = next(unlabeled_data)

            labeled_imgs = labeled_imgs.to(self.args.device)
            unlabeled_imgs = unlabeled_imgs.to(self.args.device)
            labels = labels.to(self.args.device)

            if self.args.use_features:
                labeled_features, labels_features = next(feature_data)
                labeled_features = labeled_features.to(self.args.device)
                labels_features = labels_features.to(self.args.device)

                # task_model step
                preds = task_model(labeled_features)
                task_loss = self.ce_loss(preds, labels_features.long())
                correct = accuracy_score(labels_features.cpu().numpy(), torch.argmax(preds, dim=1).cpu().numpy(), normalize=False)
            else:
                preds = task_model(labeled_imgs)
                task_loss = self.ce_loss(preds, labels)
                correct = accuracy_score(labels.cpu(), torch.argmax(preds, dim=1).cpu().numpy(), normalize=False)
            train_acc = correct / preds.size(0)

            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            # VAE step
            for count in range(self.args.num_vae_steps):
                recon, z, mu, logvar = vae(labeled_imgs)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs,
                                                  unlab_recon, unlab_mu, unlab_logvar, self.args.beta)

                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0)).to(self.args.device)
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0)).to(self.args.device)

                dsc_loss = self.bce_loss(labeled_preds.view(-1), lab_real_preds) + \
                           self.bce_loss(unlabeled_preds.view(-1), unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
                del unsup_loss
                del transductive_loss
                del dsc_loss

                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs, _ = next(unlabeled_data)

                    labeled_imgs = labeled_imgs.to(self.args.device)
                    unlabeled_imgs = unlabeled_imgs.to(self.args.device)

            # Discriminator step
            for count in range(self.args.num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs)

                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0)).to(self.args.device)
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0)).to(self.args.device)

                dsc_loss = self.bce_loss(labeled_preds.view(-1), lab_real_preds) + \
                           self.bce_loss(unlabeled_preds.view(-1), unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs, _ = next(unlabeled_data)

                    labeled_imgs = labeled_imgs.to(self.args.device)
                    unlabeled_imgs = unlabeled_imgs.to(self.args.device)

            sys.stdout.write(
                '\rtrain iter {}/{} | task loss {:.4f} vae loss {:.4f} discriminator loss {:.4f}'.format(
                    iter_count+1, self.args.train_iterations, task_loss.item(), total_vae_loss.item(),dsc_loss.item()))
            sys.stdout.flush()

            if (iter_count % 300 == 0 and iter_count > 0) or iter_count == self.args.train_iterations - 1:

                if train_acc > best_acc:
                    best_acc = train_acc
                    best_task_model = copy.deepcopy(task_model)
                    best_vae = copy.deepcopy(vae)
                    best_discriminator = copy.deepcopy(discriminator)

                if iter_count % 30000 == 0:
                    print('\ncurrent step: {} acc: {:.4f} best acc: {:.4f}'.format(iter_count, train_acc, best_acc))

        best_task_model = best_task_model.to(self.args.device)
        best_vae = best_vae.to(self.args.device)
        best_discriminator = best_discriminator.to(self.args.device)

        final_accuracy = self.test(best_task_model)
        querry_indices = self.sampler.sample(best_vae, best_discriminator, unlabeled_dataloader)

        torch.cuda.empty_cache()

        return final_accuracy, querry_indices

    def validate(self, task_model, loader, print_loss=False):
        task_model.eval()
        total, correct = 0, 0
        task_loss = 0
        for imgs, labels, _ in loader:
            imgs = imgs.to(self.args.device)

            with torch.no_grad():
                preds = task_model(imgs)
                if print_loss:
                    task_loss += self.ce_loss(preds.cpu(), labels.long())

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)

        if print_loss:
            return correct / total * 100, task_loss
        else:
            return correct / total * 100

    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        if self.args.use_features:
            test_loader = self.feature_test_dataloader
        else:
            test_loader = self.test_dataloader

        for imgs, labels, _ in test_loader:
            imgs = imgs.to(self.args.device)

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total

    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD