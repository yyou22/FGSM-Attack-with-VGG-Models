from __future__ import print_function
from cifar10_models.vgg import vgg16_bn, vgg19_bn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='FGSM Attack on CIFAR-10 with VGG Models')
parser.add_argument('--natural', action='store_true', help='natural prediction on the unperturbed dataset')
parser.add_argument('--epsilon', default=0.03, type=float, help='epsilon, the maximum amount of perturbation that can be applied')

args = parser.parse_args()

#device = torch.device("cuda")
start_idx = 0

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

inv_mean = [-0.4914/0.2471, -0.4822/0.2435, -0.4465/0.2616]
inv_std = [1/0.2471, 1/0.2435, 1/0.2616]

epsilon = args.epsilon

def natural(model, X_data, Y_data):

	model.eval()

	transform_ = T.Compose(
			[
				T.ToTensor(),
				T.Normalize(mean, std),
			]
		)

	wrong = 0

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

def margin_loss(logits,y):
	logit_org = logits.gather(1,y.view(-1,1))
	logit_target = logits.gather(1,(logits - torch.eye(10)[y] * 9999).argmax(1, keepdim=True))
	loss = -logit_org + logit_target
	loss = torch.sum(loss)
	return loss

def fgsm(image, epsilon, target, model):

	X, y = Variable(image, requires_grad = True), Variable(target)

	# output of model
	out = model(X)

	loss = margin_loss(out, target)

	model.zero_grad()

	loss.backward()

	data_grad = X.grad.data

	sign_data_grad = torch.sign(data_grad)

	perturbed_image = image + epsilon*sign_data_grad

	perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0)

	return perturbed_image

def attack(model, X_data, Y_data):

	model.eval()

	tensor_ = T.ToTensor()
	normalize = T.Normalize(mean, std)

	transform_ = T.Compose(
			[
				T.ToTensor(),
				T.Normalize(mean, std),
			]
		)

	inv_normalize = T.Normalize(inv_mean, inv_std)

	wrong = 0
	adv_examples = []

	#for idx in range(start_idx, len(Y_data)):
	for idx in range(start_idx, 10):

		#display image
		plt.imshow(X_data[idx])
		plt.show()

		un_x_data = tensor_(X_data[idx])
		un_x_data = un_x_data.numpy()

		# load unnormalized original image
		un_image = np.array(np.expand_dims(un_x_data, axis=0), dtype=np.float32)

		x_data = transform_(X_data[idx])
		x_data = x_data.numpy()

		# load original image
		image = np.array(np.expand_dims(x_data, axis=0), dtype=np.float32)

		# load label
		label = np.array([Y_data[idx]], dtype=np.int64)

		# transform to torch.tensor
		data = torch.from_numpy(un_image)
		target = torch.from_numpy(label)

		perturbed_data = fgsm(data, epsilon, target, model)

		perturbed_data = normalize(perturbed_data)

		X_ = Variable(perturbed_data)
		out = model(X_)

		pred = out.data.max(1)[1]
		
		if pred != target:
			wrong += 1

		#undo transformation
		perturbed_data = inv_normalize(perturbed_data)

		og_image = tensor_(X_data[idx])
		og_image = og_image.numpy()

		og_image = np.array(np.expand_dims(og_image, axis=0), dtype=np.float32)
		diff = og_image - perturbed_data.numpy()

		#display image
		plt.imshow(transforms.ToPILImage()(perturbed_data[0]))
		plt.show()

	print(wrong)

def main():

	model = vgg16_bn(pretrained=True)

	X_data = np.load("./data.npy")
	Y_data = np.load("./label.npy")

	if args.natural:
		natural(model, X_data, Y_data)
	else:
		attack(model, X_data, Y_data)

if __name__ == "__main__":
	main()