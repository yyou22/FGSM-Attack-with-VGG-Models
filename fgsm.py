from __future__ import print_function
from cifar10_models.vgg import vgg16_bn, vgg19_bn
from cifar10_models.resnet import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt

#device = torch.device("cuda")
start_idx = 0

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

def natural(model, X_data, Y_data):

	model.eval()

	transform_ = T.Compose(
			[
				T.ToTensor(),
				T.Normalize(mean, std),
			]
		)

	wrong = 0
	adv_examples = []

	for idx in range(start_idx, len(Y_data)):

		x_data = transform_(X_data[idx])
		x_data = x_data.numpy()

		# load original image
		image = np.array(np.expand_dims(x_data, axis=0), dtype=np.float32)

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

	model = vgg16_bn(pretrained=True)

	X_data = np.load("./data.npy")
	X_data = X_data/255.0

	Y_data = np.load("./label.npy")

	natural(model, X_data, Y_data)

if __name__ == "__main__":
	main()