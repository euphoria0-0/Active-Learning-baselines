import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from activelearner.active_learner import ActiveLearner


class LearningLoss(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.subset = args.subset
        self.batch_size = args.batch_size


    def query(self, nQuery, model):
        subset = np.random.choice(self.unlabeled_indices, self.subset, replace=False)
        self.dataloaders['unlabeled'] = DataLoader(self.dataset['unlabeled'], batch_size=self.batch_size,
                                                   sampler=SubsetRandomSampler(subset),
                                                   pin_memory=True, shuffle=False)
        model['backbone'].eval()
        model['module'].eval()

        uncertainty = torch.tensor([])
        with torch.no_grad():
            for (inputs, _) in self.dataloaders['unlabeled']:
                inputs = inputs.to(self.device)

                scores, features = model['backbone'](inputs)
                pred_loss = model['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                uncertainty = torch.cat((uncertainty, pred_loss.cpu().data), 0)

                torch.cuda.empty_cache()

        arg = np.argsort(uncertainty.numpy())[-nQuery:]

        query_indices = subset[arg].tolist()
        self.update(query_indices)
        return self.labeled_indices, self.unlabeled_indices
