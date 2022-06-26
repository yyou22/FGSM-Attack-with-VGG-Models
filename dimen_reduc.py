from __future__ import print_function
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import argparse
import pickle
import math
import csv
import os

parser = argparse.ArgumentParser(description='dimesionality reduction on FGSM-perturbed CIFAR10 datasets')
parser.add_argument('--natural', action='store_true', help='natural prediction on the unperturbed dataset')
parser.add_argument('--epsilon', default=0.03, type=float, help='[0.01|0.02|0.03], epsilon, the maximum amount of perturbation that can be applied')
parser.add_argument('--model', default='vgg16', help='[vgg16|vgg19], model that is being attacked')

args = parser.parse_args()

def TSNE_(data):

	tsne = TSNE(n_components=2)
	data = tsne.fit_transform(data)

	return data

def PCA_(data):

	pca = PCA(n_components=2)
	data = pca.fit_transform(data)

	return data

def main():

	if args.natural:
		path = '../data/' + args.model + '/000'
		#X_data = np.load("../data/X.npy")
		X_data = np.load(path + '/features.npy')
	else:
		path = '../data/' + args.model + '/' + ''.join(str(args.epsilon).split('.'))
		#X_data = np.load(path + '/adv_X.npy')
		X_data = np.load(path + '/features.npy')

	Y_data = np.load("../data/Y.npy")
	Y_data = Y_data.reshape((Y_data.shape[0], 1))

	confid_level = np.load(path + '/confid_level.npy')

	Y_hat = np.load(path + '/Y_hat.npy')
	Y_hat = Y_hat.reshape((Y_hat.shape[0], 1))

	with open(path + '/error.pckl', 'rb') as file:
		error = pickle.load(file)

	#X_data = X_data.reshape(X_data.shape[0], 3072)
	X_data = X_data.reshape(X_data.shape[0], 512)

	X_data = PCA_(X_data)
	#X_data = TSNE_(X_data)

	tx, ty = X_data[:, 0].reshape(100, 1), X_data[:, 1].reshape(100, 1)
	tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
	ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

	type_ = ['%.5f'] * 12 + ['%d'] * 2

	result = np.concatenate((tx, ty, confid_level, Y_hat, Y_data), axis=1)
	np.savetxt(path + "/data.csv", result, header="xpos,ypos,0,1,2,3,4,5,6,7,8,9,pred,target", comments='', delimiter=',', fmt=type_)

if __name__ == "__main__":
	main()

