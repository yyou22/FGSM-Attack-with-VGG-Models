from __future__ import print_function
from cifar10_models.vgg import vgg16_bn, vgg19_bn
from cifar10_models.resnet import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

#device = torch.device("cuda")
start_idx = 0

def natural(model, X_data, Y_data):

	model.eval()

	wrong = 0
	adv_examples = []

	for idx in range(start_idx, len(Y_data)):

		# load original image
		image = np.array(np.expand_dims(X_data[idx], axis=0), dtype=np.float32)

		# load label
		label = np.array([Y_data[idx]], dtype=np.int64)

		# transform to torch.tensor
		data = torch.from_numpy(image)
		target = torch.from_numpy(label)

		X, y = Variable(data, requires_grad = True), Variable(target)

		# output of model
		out = model(X)
		init_pred = out.data.max(1)[1]
		
		if init_pred != target:
			wrong += 1

	print(wrong)


def main():

	model = resnet18(pretrained=True)

	X_data = np.load("./data_3.npy")
	Y_data = np.load("./label_3.npy")

	natural(model, X_data, Y_data)

if __name__ == "__main__":
	main()