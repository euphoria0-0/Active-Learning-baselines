'''
Reference:
    https://github.com/sinhasam/vaal
'''
import torch
from .active_learner import ActiveLearner


class VAAL(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)


    def query(self, nQuery, model):
        vae, discriminator = model
        vae, discriminator = vae.to(self.device), discriminator.to(self.device)
        vae.eval()
        discriminator.eval()

        all_preds, all_indices = [], []
        for images, _, indices in self.dataloaders_imgs['unlabeled']:
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
        _, querry_indices = torch.topk(all_preds, nQuery)
        query_indices = torch.tensor(all_indices)[querry_indices].tolist()

        self.update(query_indices)

