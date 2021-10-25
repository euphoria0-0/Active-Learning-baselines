'''
Reference:
    Weight Decay Scheduling and Knowledge Distillation for Active Learning
    https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710426.pdf
'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .active_learner import ActiveLearner


class WS(ActiveLearner):
	def __init__(self, dataset, args):
		super().__init__(dataset, args)
		self.norms = None
		self.nClass = args.nClass

	def query(self, nQuery, model):
		self.dataloaders['unlabeled'] = DataLoader(self.dataset['unlabeled'], **self.loader_args)

		if self.norms is None:
			self.norms = self.save_weight_distribution(model)
		model.eval()

		total_classes, entropy_total = [], []
		confusion_matrix = torch.zeros(self.nClass, self.nClass)
		with torch.no_grad():
			for input, labels, _ in self.dataloaders['unlabeled']:
				input, labels = input.to(self.device), labels.to(self.device)
				output, _ = model(input)

				_, preds = torch.max(output.data, 1)

				# compute class separate accuracy
				for t, p in zip(labels.view(-1), preds.view(-1)):
					confusion_matrix[t.long(), p.long()] += 1

				# compute entropy for active learning
				entropy = F.softmax(output.data, dim=1) * F.log_softmax(output.data, dim=1)
				entropy_total.extend(-1.0 * entropy.sum(dim=1))

				# class count
				total_classes.extend(labels.cpu().data.numpy())

				torch.cuda.empty_cache()

		accuracy_each_class = confusion_matrix.diag() / confusion_matrix.sum(1)

		entropy = torch.stack(entropy_total).cpu()

		self.norms = self.save_weight_distribution(model)

		query_indices = self.entropy_sampler(entropy, nQuery, type=self.args.sampling)
		num_each_class = self.class_count(self.dataloaders['unlabeled'], query_indices, self.nClass)

		for i in range(self.nClass):
			print('ClassIdx {} Num {:4d} Acc {:.4f}'.format(i, num_each_class[i], accuracy_each_class[i]))

		torch.cuda.empty_cache()

		self.update(query_indices)


	def save_weight_distribution(self, model):
		weight_l2_norm = 0
		for name, param in model.named_parameters():
			weight_l2_norm += param.data.norm(2).item()**2
		weight_l2_norm = weight_l2_norm**(1./2)

		if self.norms is None:
			self.norms = {'weight_l2_norm': []}
		self.norms['weight_l2_norm'].append(weight_l2_norm)
		return self.norms


	def entropy_sampler(self, entropy, nQuery, type='descending'):
		print('==> Sampling data..')

		if type == 'descending':
			new_data_idx = np.argsort(-entropy[self.unlabeled_indices])[:nQuery]
			#new_data_idx = self.unlabeled_indices[np.argsort(-entropy[self.unlabeled_indices])][:nQuery]
		elif type == 'ascending':
			new_data_idx = np.argsort(entropy[self.unlabeled_indices])[:nQuery]
			#new_data_idx = self.unlabeled_indices[np.argsort(entropy[self.unlabeled_indices])][:nQuery]
		elif type == 'random':
			new_data_idx = np.random.choice(self.unlabeled_indices, nQuery, replace=False)
		elif type == 'sampled_descending':
			np.random.shuffle(self.unlabeled_indices)
			unlabeled_indices = self.unlabeled_indices[:10000]
			new_data_idx = np.argsort(-entropy[unlabeled_indices])[:nQuery]
			#new_data_idx = self.unlabeled_indices[np.argsort(-entropy[self.unlabeled_indices])][:nQuery]
		else:
			raise NotImplementedError()

		return new_data_idx.tolist()

	def class_count(self, data_loader, labeled_idx, num_classes):
		total_classes = []
		for i, (inputs, targets, _) in enumerate(data_loader):
			total_classes.extend(targets.numpy())
		labeled_classes = np.take(total_classes, labeled_idx)

		num_each_class = np.zeros([num_classes, ], dtype=int)
		for i in range(num_classes):
			num_each_class[i] = (labeled_classes == i).sum()

		return num_each_class

