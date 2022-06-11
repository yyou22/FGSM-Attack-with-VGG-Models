from sklearn.manifold import TSNE
import numpy as np
import argparse
import pickle
import math
import csv
import os

parser = argparse.ArgumentParser(description='dimesionality reduction on FGSM-perturbed CIFAR10 datasets')
parser.add_argument('--epsilon', default=0.03, type=float, help='[0.00|0.01|0.02|0.03], epsilon, the maximum amount of perturbation that can be applied')
parser.add_argument('--model', default='vgg16', help='[vgg16|vgg19], model that is being attacked')

args = parser.parse_args()

def TSNE_(data):

	tsne = TSNE(n_components=2)
	data = tsne.fit_transform(data)

	return data

def main():

	path = '../data/' + args.model + '/' + ''.join(str(args.epsilon).split('.'))

	if math.isclose(args.epsilon, 0.00, abs_tol=1e-8):
		X_data = np.load("../data/X.npy")
	else:
		X_data = np.load(path + '/adv_X.npy')

	Y_data = np.load("../data/Y.npy")
	Y_data = Y_data.reshape((Y_data.shape[0], 1))

	confid_level = np.load(path + '/confid_level.npy')

	Y_hat = np.load(path + '/Y_hat.npy')
	Y_hat = Y_hat.reshape((Y_hat.shape[0], 1))

	with open(path + '/error.pckl', 'rb') as file:
		error = pickle.load(file)

	X_data = X_data.reshape(X_data.shape[0], 3072)
	X_data = TSNE_(X_data)

	type_ = ['%.5f'] * 12 + ['%d'] * 2

	result = np.concatenate((X_data, confid_level, Y_hat, Y_data), axis=1)
	np.savetxt(path + "/data.csv", result, header="xpos,ypos,0,1,2,3,4,5,6,7,8,9,pred,target", comments='', delimiter=',', fmt=type_)

if __name__ == "__main__":
	main()