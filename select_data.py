from __future__ import print_function
from cifar10_models.vgg import vgg16_bn, vgg19_bn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random

torch.manual_seed(1)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

def select_data():

	hash_map = {}

	transform_ = T.Compose(
			[
				T.ToTensor(),
				T.Normalize(mean, std),
			]
		)

	#retrieve test data
	test_data_ = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform_)

	#convert to numpy array
	test_data = test_data_.data

	#retrieve test labels
	test_label = test_data_.targets

	visited = []
	data = []
	label = []
	hash_map = {}
	total = 0

	while total != 100:
		ind = random.randint(0, 999)

		#check if ind is already visited
		if ind in visited:
			continue

		#check if label is full
		if test_label[ind] in hash_map and hash_map[test_label[ind]] == 10:
			continue

		#add the example
		if test_label[ind] not in hash_map:
			hash_map[test_label[ind]] = 1
		else:
			hash_map[test_label[ind]] += 1

		visited.append(ind)
		total += 1
		data.append(list(test_data[ind]))
		label.append(test_label[ind])

	data = np.array(data)
	label = np.array(label)

	norm_data = []
	data = data/255

	for data_ in data:
		data_ = transform_(data_)
		norm_data.append(data_.numpy())

	norm_data = np.array(norm_data)

	np.save('./data', norm_data)
	np.save('./label', label)

def main():

	select_data()

if __name__ == "__main__":
	main()