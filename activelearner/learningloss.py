'''
Reference:
    https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
'''
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from .active_learner import ActiveLearner


class LearningLoss(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.subset = args.subset

    def query(self, nQuery, model):
        subset = np.random.choice(self.unlabeled_indices, self.subset, replace=False)
        unlabeled_loader = DataLoader(self.dataset['unlabeled'], **self.loader_args,
                                      sampler=SubsetRandomSampler(subset))
        model['backbone'].eval()
        model['module'].eval()

        uncertainty = torch.tensor([])
        with torch.no_grad():
            for (inputs, _, _) in unlabeled_loader:
                inputs = inputs.to(self.device)

                scores, features = model['backbone'](inputs)
                pred_loss = model['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                uncertainty = torch.cat((uncertainty, pred_loss.cpu().data), 0)

                torch.cuda.empty_cache()

        arg = np.argsort(uncertainty.numpy())[-nQuery:]

        query_indices = subset[arg].tolist()
        self.update(query_indices)
