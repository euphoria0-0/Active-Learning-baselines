import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
from scipy.spatial import distance_matrix
# from sklearn.metrics import pairwise_distances
import seaborn as sns
from matplotlib import pyplot as plt
import copy
import time

def save_weight_distribution(args, save_dir, model, episode_i, norms=None, save=True, xlim=None, ylim=None):
	# weight = torch.cat([param.data.cpu().view(-1) for param in model.parameters()], 0).numpy()
	# weight_l2_norm = np.sqrt(np.sum(np.square(weight)))
	weight_l2_norm = 0
	for name, param in model.named_parameters():
		weight_l2_norm += param.data.norm(2).item()**2
	weight_l2_norm = weight_l2_norm**(1./2)
	# print(weight_l2_norm)

	if 'resnet' in args.net_type:
		# bn_parameters = ['bn', 'shortcut.1']	# shortcut.1.weight, shortcut.1.bias : bn at shortcut
		bn_parameters = ['layer1.0.bn1.weight']
	elif 'densenet' in args.net_type:
		bn_parameters = ['bn']
	elif 'conv' in args.net_type:
		bn_parameters = ['bn']
	else:
		bn_parameters = ['bn']

	if norms==None:
		norms = {'weight_l2_norm': []}
		if save:
			for (name, param) in model.named_parameters():
				norms[name] = []
	norms['weight_l2_norm'].append(weight_l2_norm)

	if save:
		conv_weight = []
		for name, param in model.named_parameters():
			norm = param.data.norm(2).item()
			norms[name].append(norm)

			# for bn parameter
			if any(bn_parameter in name for bn_parameter in bn_parameters):
				# param = param.data.cpu().view(param.numel()).numpy()
				# abs_param = np.absolute(param)
				abs_param = torch.abs(param.data).cpu().view(param.numel()).numpy()
				if 'bias' in name:
					graph = sns.displot(abs_param, kde=False, bins=np.arange(0, 2, 0.001))
					graph.set_xlim(0,max(abs_param)+0.001)
				else:
					a=plt.figure()
					ax = a.add_subplot(111)

					graph = sns.displot(abs_param, kde=False, bins=np.arange(0, 2, 0.01))
					graph.set_xlim(0,max(abs_param)+0.05)
					mean_param = np.average(abs_param)
					sigma = mean_param*0.1
					no_capacy_factor = np.sum(abs_param < sigma)
					plt.axvline(x=sigma, color='r')
					plt.text(0.8, 0.8, str(no_capacy_factor), horizontalalignment='center', verticalalignment='center',transform=ax.transAxes, fontsize=30)
					plt.savefig('{}/episode_{}_{}.pdf'.format(save_dir, episode_i, name), dpi=300)
					plt.close()
			else:
				param = param.data.cpu().view(param.numel()).numpy()
				# graph = sns.distplot(param, kde=False, bins=np.arange(0, 1, 0.001))
				conv_weight.extend(param)

		a=plt.figure()
		ax = a.add_subplot(111)
		graph = sns.displot(conv_weight, kde=False, bins=np.arange(-0.02, 0.02, 0.0001))
		conv_l2_norm = np.sqrt(np.sum(np.square(conv_weight)))
		plt.text(0.8, 0.8, str(round(conv_l2_norm,2)), horizontalalignment='center', verticalalignment='center',transform=ax.transAxes, fontsize=30)
		if xlim is not None:
			graph.set_xlim(-xlim,xlim)
		if ylim is not None:
			graph.set_ylim(0,ylim)
		plt.savefig('{}/episode_{}_conv_weight_distribution_CR.pdf'.format(save_dir, episode_i), dpi=300)
		plt.close()

	return norms

def entropy_sampler(labeled_idx, len_dataset, entropy, increment, type='descending'):
	print('==> Sampling data..')
		
	total = np.arange(len_dataset)
	unlabeled_idx = np.array(list(set(total)-set(labeled_idx)))

	if type == 'descending':
		new_data_idx = unlabeled_idx[np.argsort(-entropy[unlabeled_idx])][:increment]
	elif type == 'ascending':
		new_data_idx = unlabeled_idx[np.argsort(entropy[unlabeled_idx])][:increment]
	elif type == 'random':
		new_data_idx = np.random.choice(unlabeled_idx, increment, replace=False)
	elif type == 'sampled_descending':
		np.random.shuffle(unlabeled_idx)
		unlabeled_idx = unlabeled_idx[:10000]
		new_data_idx = unlabeled_idx[np.argsort(-entropy[unlabeled_idx])][:increment]
	else:
		raise NotImplementedError()

	# print(new_data_idx)
	labeled_idx = np.concatenate((labeled_idx, new_data_idx))
	return labeled_idx

def learning_loss_sampler(labeled_idx, len_dataset, loss_prediction, increment, type='descending'):
	print('==> Sampling data..')
	
	total = np.arange(len_dataset)
	unlabeled_idx = np.array(list(set(total)-set(labeled_idx)))

	np.random.shuffle(unlabeled_idx)
	unlabeled_idx = unlabeled_idx[:10000]	# select 10000 samples from unlabeled data
	unlabeled_loss_prediction = loss_prediction[unlabeled_idx]

	if type == 'descending':
		new_data_idx = unlabeled_idx[np.argsort(-unlabeled_loss_prediction)][:increment]
	elif type == 'ascending':
		new_data_idx = unlabeled_idx[np.argsort(unlabeled_loss_prediction)][:increment]
	elif type == 'random':	
		new_data_idx = np.random.choice(unlabeled_idx, increment, replace=False)
	else:
		raise NotImplementedError()
	labeled_idx = np.concatenate((labeled_idx, new_data_idx))
	return labeled_idx

def class_count(data_loader, labeled_idx, num_classes):
	total_classes = []
	for i, (inputs, targets) in enumerate(data_loader):
		total_classes.extend(targets.numpy())
	labeled_classes = np.take(total_classes, labeled_idx)

	num_each_class = np.zeros([num_classes,],dtype=int)
	for i in range(num_classes):
		num_each_class[i] = (labeled_classes==i).sum()

	return num_each_class

def test():
	dataset = np.array(range(100))

	# idx = [1,2,3,4]
	idx = torch.randperm(4)[:2]
	sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)
	loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler)

	print('\t1st')
	for i, batch in enumerate(loader):
		print(batch)
		
	idx = torch.cat((idx, torch.tensor([5,6,7,8,9,10])), dim = 0)
	print(idx)
	sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)
	loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler)

	print('\t2nd')
	for i, batch in enumerate(loader):
		print(batch)

if __name__ == '__main__':
	# test()

	# a = np.random.choice(10, 9, replace=False)
	# print(a)

	# a = torch.randperm(5)[:5]
	# print(a)

	####### test entropy_sampler
	labeled_idx = np.array([1,3,5,7,9])
	len_dataset = 15
	entropy = np.arange(15)/10.
	np.random.shuffle(entropy)
	print(entropy)
	increment = 3
	labeled_idx = entropy_sampler(labeled_idx, len_dataset, entropy, increment, type='descending')
	print(labeled_idx)
	labeled_idx = entropy_sampler(labeled_idx, len_dataset, entropy, increment, type='descending')
	print(labeled_idx)
	print(entropy[labeled_idx])
