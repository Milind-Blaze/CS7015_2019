"""Assignment 4

"""

import argparse
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 
import os
from os.path import join
import pandas as pd
import pickle 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys


plt.rcParams.update({'font.size': 22})


parser = argparse.ArgumentParser()
parser.add_argument("--n", default = 100, help = "number of hidden units", type = int)
parser.add_argument("--k", default = 1, help = "number of steps of cd per example", type = int)
parser.add_argument("--eta", default = 0.001, help = "learning rate for cd", type = float)
parser.add_argument("--num_epochs", default = 1, help = "number of epochs to run for", type = int)
parser.add_argument("--path_train", help = "path to the training data", default = "./train.csv")
parser.add_argument("--path_test", help = "path to the test data", default = "./test.csv")
args = parser.parse_args()



np.random.seed(0)







########################################################### functions #######################################################

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def sample_vector(n, weight, vector, bias):
	if vector.ndim == 1:
		vector = vector[:, np.newaxis]
	if bias.ndim == 1:
		bias = bias[:, np.newaxis]


	z_linear = np.dot(weight, vector) + bias
	probs = sigmoid(z_linear)
	if probs.ndim == 1:
		probs = probs[:, np.newaxis]

	# print(np.shape(z_linear), np.shape(probs), "shape of z_linear")
	assert np.shape(probs) == (n,1) 

	random = np.random.random((n,1))

	one = random < probs

	zerovec = np.zeros((n,1))

	zerovec[one] = 1

	assert zerovec.ndim == 2

	return zerovec


def show_image():
	image_temp = X_train_thresh[image_id,:]
	h_temp =  sample_vector(n, W, image_temp, c)
	image_recon_temp = sample_vector(num_visible, W.T, h, b)
	image_recon_temp = np.reshape(image_recon_temp, (28,28))
	plt.imshow(image_recon_temp, cmap = "gray")
	plt.show()
	plt.close()
########################################################### set values #######################################################

threshold = 127
k = args.k
n = args.n
eta = args.eta
num_epochs = args.num_epochs
m = 6400
l = 64
path_train = args.path_train
path_test = args.path_test


path_folder = "./k" + str(k) + " n" + str(n) + " eta" + str(eta) + " epochs" + str(num_epochs)
try:
	os.mkdir(path_folder)
except FileExistsError:
    print("folder already exists. Name: " + path_folder)


############################################################ Prepare the data #################################################


############# selecting the dataset ################

data_train_df = pd.read_csv(path_train)
data_train = data_train_df.to_numpy()
X_train = data_train[:,1:-1]
labels_train = data_train[:,-1]

data_test_df = pd.read_csv(path_test)
data_test = data_test_df.to_numpy()
X_test = data_test[:,1:-1]
labels_test = data_test[:,-1]


# # for mnist
# (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
# X_train = np.reshape(xtrain, (np.shape(xtrain)[0], -1))
# print(np.shape(X_train), "shape for mnist")
# X_test =  np.reshape(xtest, (np.shape(xtest)[0], -1))
# labels_test = ytest




############## thresholding #####################

greater = X_train >= threshold
lesser = X_train < threshold

X_train_thresh = X_train.copy()
X_train_thresh[greater] = 1
X_train_thresh[lesser] = 0

# for mnist
# X_test = np.reshape(xtest, (np.shape(xtest)[0], -1))
# print(np.shape(X_test), "shape for mnist")

greater = X_test >= threshold
lesser = X_test < threshold

X_test_thresh = X_test.copy()
X_test_thresh[greater] = 1
X_test_thresh[lesser] = 0

print("max", np.max(X_test_thresh), np.max(X_train_thresh))




############################################################ Plotting the data #################################################

# image_id = 1032

# image = np.reshape(X_train[image_id,:], (28,28))
# image_thresh = np.reshape(X_train_thresh[image_id, :], (28,28))


# ######## plotting the image ############

# plt.figure(figsize = (10,8))
# plt.imshow(image, cmap = "gray")
# plt.show()
# plt.close()

# plt.figure(figsize = (10,8))
# plt.imshow(image_thresh, cmap = "gray")
# plt.show()
# plt.close()


############################################################ Creating the RBM #################################################

num_visible = np.shape(X_train_thresh)[1]
print(num_visible)


################ parameters ##################

W = 0.01*np.random.randn(n, num_visible)
# print(np.shape(W))
b = np.zeros((num_visible,1))
c = np.zeros((n,1))



########################################## Training the RBM ####################################################################




# print(sample_vector(n,W,b,c))


num_examples = np.shape(X_train_thresh)[0]
print("number of examples: ", num_examples)

h = np.zeros((n,1))


image_id = 10


image_thresh = np.reshape(X_train_thresh[image_id, :], (28,28))


# ######## plotting the image ############

plt.figure(figsize = (10,8))
plt.imshow(image_thresh, cmap = "gray")
plt.savefig(join(path_folder, "original.png"))
plt.close()







image_temp = X_train_thresh[image_id,:]

for epoch in range(num_epochs):
	print("Epoch: ", epoch)
	plt.figure(figsize = (20,16))
	subplot_no = 1
	for i in range(num_examples):
		# print(i)

		###### creating subplots- the 8 x 8 grid ######
		if ((i%(936) == 0)) and (subplot_no <= 64):
			print("i = ", i)
			# show_image()
			
			h_temp =  sample_vector(n, W, image_temp, c)
			image_recon_temp1 = sample_vector(num_visible, W.T, h_temp, b)
			image_recon_temp = np.reshape(image_recon_temp1, (28,28))
			plt.subplot(8,8,subplot_no)
			plt.imshow(image_recon_temp, cmap = "gray")
			plt.axis("off")

			# print("loss: ", np.sum((image_temp-image_recon_temp1)**2))

			subplot_no += 1

		v = X_train_thresh[i,:]
		if v.ndim == 1:
			v = v[:, np.newaxis]
		vtemp = X_train_thresh[i,:]
		if vtemp.ndim == 1:
			vtemp = vtemp[:, np.newaxis]

		for t in range(k):
			h = sample_vector(n, W, vtemp, c)
			vtemp = sample_vector(num_visible, W.T, h, b)


		z = sigmoid(np.dot(W,v) + c)
		if z.ndim == 1:
			z = z[:, np.newaxis]
		ztemp = sigmoid(np.dot(W, vtemp) + c)
		if ztemp.ndim == 1:
			ztemp = ztemp[:, np.newaxis]
		# print("shapes", np.shape(v.T), np.shape(vtemp.T))
		W = W + eta*(np.dot(z, v.T) - np.dot(ztemp, vtemp.T))
		b = b + eta*(v - vtemp)
		c = c + eta*(z - ztemp)

	plt.savefig(join(path_folder,"changing_image"+ str(epoch) + ".png"))
	# plt.show()
	plt.close()


############ plotting a few images ###############


image = X_train_thresh[image_id,:]
h = sample_vector(n, W, image, c)
vtemp = sample_vector(num_visible, W.T, h, b)
print(np.shape(vtemp))

image = np.reshape(image, (28,28))
vtemp = np.reshape(vtemp, (28,28))

plt.figure()
plt.imshow(image, cmap = "gray")
plt.savefig(join(path_folder, "origafter.png"))
plt.close()

plt.figure()
plt.imshow(vtemp, cmap = "gray")
plt.savefig(join(path_folder, "origarecon.png"))
plt.close()




################## Computing the hidden representations for the test images #######################




print("Computing the hidden representations for the training examples")
print("Shape of test data: ", np.shape(X_test_thresh))

num_test_examples = np.shape(X_test_thresh)[0]
hidden_reps = []
for i in range(num_test_examples):
	image = X_test_thresh[i,:]
	h = sample_vector(n, W, image, c)
	hidden_reps.append(np.array(h[:,0]))

print(np.shape(hidden_reps))

hidden_reps = np.array(hidden_reps)

embedded = TSNE(n_components = 2, early_exaggeration = 7).fit_transform(hidden_reps)
print(np.shape(embedded))

# path_pickle = join(path_folder, "embedded.pickle")

vis_x = embedded[:,0]
vis_y = embedded[:,1]

print(np.shape(labels_test))

plt.figure(figsize = (20,16))
plt.title("Embeddings for testing examples, k = " + str(k))
plt.xlabel("x")
plt.ylabel("Y")
plt.scatter(vis_x[:10000], vis_y[:10000], c = labels_test[:10000], cmap = plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks = range(max(labels_test)))
plt.savefig(join(path_folder, "clusters_test.png"))
plt.close()