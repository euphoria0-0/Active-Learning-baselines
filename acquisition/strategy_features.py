import numpy as np
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
import pdb
#import resnet

class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args, device):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.device = device

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.
        total = 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            #print(out.size(), y.size())
            accFinal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            #tmp = (torch.max(out, 1)[1] == y).float()
            #print(tmp.size(), torch.sum(tmp), tmp)
            total += x.size(0)
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        #print(total, len(loader_tr.dataset.X))
        return accFinal / total # len(loader_tr.dataset.X)

    def train(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        n_epoch = self.args['n_epoch']
        self.clf = self.net.apply(weight_reset).to(self.device)
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long()),
                               shuffle=True, **self.args['loader_tr_args'])  # , transform=self.args['transform']

        for epoch in range(1,n_epoch+1):
            accCurrent = self._train(epoch, loader_tr, optimizer)
            sys.stdout.write('\rEpoch {} Train Acc: {:.4f}'.format(epoch, accCurrent))
            #sys.stdout.write('\r' + str(epoch) + ' training accuracy: ' + str(accCurrent))
            sys.stdout.flush()
            if (epoch % 50 == 0) and (accCurrent < 0.2):  # reset if not converging
                self.clf = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)
                #print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
                print('\rEpoch {} Train Acc: {:.4f}'.format(epoch, accCurrent), flush=True)
        '''epoch = 1
        while accCurrent < 0.995:
            accCurrent = self._train(epoch, loader_tr, optimizer)
            epoch += 1
            '''
        print()

    def predict(self, X, Y):
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y),  # ,transform=self.args['transformTest']
                                   shuffle=False, **self.args['loader_te_args'])
        else:
            loader_te = DataLoader(self.handler(X.numpy(), Y),  # , transform=self.args['transformTest']
                                   shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y), #, transform=self.args['transformTest']
                               shuffle=False,
                               **self.args['loader_te_args'])
        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data

        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y), #, transform=self.args['transformTest']
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i + 1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop

        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y), #, transform=self.args['transformTest']
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i + 1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y), #, transform=self.args['transformTest']
                               shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()

        return embedding

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y):
        model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y),  # , transform=self.args['transformTest']
                               shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                                        -1 * batchProbs[j][c])
            return torch.Tensor(embedding)
