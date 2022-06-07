from __future__ import print_function
from cifar10_models.vgg import vgg16_bn, vgg19_bn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random

torch.manual_seed(0)

epsilons = [0, 0.01, 0.02, 0.03]

def select_data():

	sub_data = []
	hash_map = {}

	#retrieve test data
	test_data_ = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transforms.ToTensor())

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
		ind = random.randint(0, 10000)

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

	np.save('./data', data)
	np.save('./label', label)

def main():

	select_data()

if __name__ == "__main__":
	main()