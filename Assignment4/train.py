"""Convolutional neural networks
"""

import argparse
import matplotlib.pyplot as plt 
import numpy as np 
import os
from os.path import join
import pandas as pd
import pickle 
from sklearn.decomposition import PCA
import skimage
import sys
import tensorflow as tf





def convert_to_onehot(indices, num_classes):
	"""Converts labels to onehot vectors 

	Parameters:
		indices (array): true labels 
		num_classes (int): number of classes

	Returns:
		result (array): one hot labels of shape (num_examples, num_classes)

	"""
	# borrowed from stack overflow
	output = np.eye(num_classes)[np.array(indices).reshape(-1)]
	# the reshape just seems to verify the shape of the matrix
	# each target vector is converted to a row vector
	result = output.reshape(list(np.shape(indices))+[num_classes])
	return result



def read_data(path_train, path_val, path_test, shape):
	"""Reads data for training, validation and testing

	Read data and return normalized (divide by 255) data volumes for training, validation 
	and testing

	Parameters:
		path_train (string): path to the training data
		path_val (string): path to the validation data
		path_test (string): path to the testing data
		shape (tuple): a tuple of dimensions to reshape the data (height, width, num_channels)

	Returns:
		X_train (numpy array): training data of shape (number of examples, 64, 64, 3) normalized by 255
		Y_train (numpy array): onehot training labels of shape (number of examples, number of classes)
		X_val (numpy array): training data of shape (number of examples, 64, 64, 3) normalized by 255
		Y_val (numpy array): onehot validation labels of shape (number of examples, number of classes)
		X_test (numpy array): testing data of shape (number of examples, 64, 64, 3) normalized by 255
	"""

	h = shape[0]
	w = shape[1]
	c = shape[2]

	data_train = pd.read_csv(path_train)
	data_train = data_train.to_numpy()
	# TODO: (0) Check this normalization and look for a better one
	X_train = (data_train[:, 1:-1])/255
	# TODO: (0) Check this reshaping 
	X_train = X_train.reshape(-1,h,w,c)
	y_train = data_train[:,-1]
	Y_train = convert_to_onehot(y_train, num_classes)
	print("Shape of training data: ", np.shape(X_train), "Shape of train labels: ", np.shape(Y_train))

	data_val = pd.read_csv(path_val)
	data_val = data_val.to_numpy()
	# TODO(0): Check this normalization and look for a better one
	X_val = (data_val[:, 1:-1])/255
	# TODO: (0) Check this reshaping 
	X_val = X_val.reshape(-1,h,w,c)
	y_val = data_val[:,-1]
	Y_val = convert_to_onehot(y_val, num_classes)
	print("Shape of validation data: ", np.shape(X_val), "Shape of validation labels: ", np.shape(Y_val))


	data_test = pd.read_csv(path_test)
	data_test = data_test.to_numpy()
	# TODO(0): Check this normalization and look for a better one
	X_test = (data_test[:, 1:])/255
	# TODO: (0) Check this reshaping 
	X_test = X_test.reshape(-1,h,w,c)
	print("Shape of test data: ", np.shape(X_test))

	return X_train, Y_train, X_val, Y_val, X_test

def create_placeholders(shape, num_classes):
	""" Create and return placeholders for data supply 

		Parameters:
			shape (tuple): a tuple of dimensions to reshape the data (height, width, num_channels)
			num_classes (int): number of classes into which data must be classified

		Returns:
			x (placeholder): a placeholder for input images
			y (placeholder): a placeholder for the labels
	"""
	h = shape[0]
	w = shape[1]
	c = shape[2]
	x = tf.placeholder(tf.float32, shape = (None, h, w, c), name = "x")
	y = tf.placeholder(tf.float32, shape = (None, num_classes), name = "y")

	return x,y



def convnet_actual(x):
	"""The convnet defined in the PA2 pdf

		Parameters:
			x (placeholder): input data

		Returns:
			softmax_linear (numpy array): output of the operations of the conv net with the final layer being linear
	"""

	conv1 = tf.contrib.layers.conv2d(x, 32, [5,5], stride = 1, padding = "SAME", activation_fn = tf.nn.relu)#, scope = "pa2")
	conv2 = tf.contrib.layers.conv2d(conv1, 32, [5,5], stride = 1, padding = "same", activation_fn = tf.nn.relu)#, scope = "pa2")
	pool1 = tf.contrib.layers.max_pool2d(conv2, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	conv3 = tf.contrib.layers.conv2d(pool1, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.relu)#, scope = "pa2")
	conv4 = tf.contrib.layers.conv2d(conv3, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.relu)#, scope = "pa2")
	pool2 = tf.contrib.layers.max_pool2d(conv4, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	conv5 = tf.contrib.layers.conv2d(pool2, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.relu)#, scope = "pa2")
	conv6 = tf.contrib.layers.conv2d(conv5, 128, [3,3], stride = 1, padding = "VALID", activation_fn = tf.nn.relu)#, scope = "pa2")
	pool3 = tf.contrib.layers.max_pool2d(conv6, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	flat = tf.contrib.layers.flatten(pool3)
	fc1 = tf.contrib.layers.fully_connected(flat, 256, activation_fn = tf.nn.relu)
	batch = tf.contrib.layers.batch_norm(fc1, scale = True)
	softmax_linear = tf.contrib.layers.fully_connected(batch, 20, activation_fn = None)

	return softmax_linear

def convnet_actual_he(x):
	"""The convnet defined in the PA2 pdf

		Parameters:
			x (placeholder): input data

		Returns:
			softmax_linear (numpy array): output of the operations of the conv net with the final layer being linear
	"""

	conv1 = tf.contrib.layers.conv2d(x, 32, [5,5], stride = 1, padding = "SAME", activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	conv2 = tf.contrib.layers.conv2d(conv1, 32, [5,5], stride = 1, padding = "same", activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	pool1 = tf.contrib.layers.max_pool2d(conv2, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	conv3 = tf.contrib.layers.conv2d(pool1, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	conv4 = tf.contrib.layers.conv2d(conv3, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	pool2 = tf.contrib.layers.max_pool2d(conv4, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	conv5 = tf.contrib.layers.conv2d(pool2, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	conv6 = tf.contrib.layers.conv2d(conv5, 128, [3,3], stride = 1, padding = "VALID", activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	pool3 = tf.contrib.layers.max_pool2d(conv6, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	flat = tf.contrib.layers.flatten(pool3)
	fc1 = tf.contrib.layers.fully_connected(flat, 256, activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))
	batch = tf.contrib.layers.batch_norm(fc1, scale = True)
	softmax_linear = tf.contrib.layers.fully_connected(batch, 20, activation_fn = None, weights_initializer = tf.initializers.he_normal(seed))

	return softmax_linear


#regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)



def convnet_given(x):
	"""The convnet defined in the PA2 pdf

		Parameters:
			x (placeholder): input data

		Returns:
			softmax_linear (numpy array): output of the operations of the conv net with the final layer being linear
	"""

	conv1 = tf.contrib.layers.conv2d(x, 32, [5,5], stride = 1, padding = "SAME", activation_fn = tf.nn.leaky_relu)#, scope = "pa2")
	batch1 = tf.contrib.layers.batch_norm(conv1, scale =True)
	conv2 = tf.contrib.layers.conv2d(batch1, 32, [5,5], stride = 1, padding = "same", activation_fn = tf.nn.leaky_relu)#, scope = "pa2")
	batch2 = tf.contrib.layers.batch_norm(conv2, scale =True)
	pool1 = tf.contrib.layers.max_pool2d(batch2, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	conv3 = tf.contrib.layers.conv2d(pool1, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.leaky_relu,)#, scope = "pa2")
	batch3 = tf.contrib.layers.batch_norm(conv3, scale =True)
	conv4 = tf.contrib.layers.conv2d(batch3, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.leaky_relu)#, scope = "pa2")
	batch4 = tf.contrib.layers.batch_norm(conv4, scale =True)
	pool2 = tf.contrib.layers.max_pool2d(batch4, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	conv5 = tf.contrib.layers.conv2d(pool2, 128, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.leaky_relu)#, scope = "pa2")
	drop5 = tf.contrib.layers.dropout(conv5, keep_prob = 0.55)
	batch5 = tf.contrib.layers.batch_norm(drop5, scale =True)
	conv6 = tf.contrib.layers.conv2d(batch5, 256, [3,3], stride = 1, padding = "VALID", activation_fn = tf.nn.leaky_relu)#, scope = "pa2")
	drop6 = tf.contrib.layers.dropout(conv6, keep_prob = 0.55)
	batch6 = tf.contrib.layers.batch_norm(drop6, scale =True)
	pool3 = tf.contrib.layers.max_pool2d(batch6, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	flat = tf.contrib.layers.flatten(pool3)
	regularizer = tf.contrib.layers.l2_regularizer(scale=0.0009)
	fc1 = tf.contrib.layers.fully_connected(flat, 512, activation_fn = tf.nn.leaky_relu,weights_regularizer= regularizer)#, scope = "pa2")
	dropf1 = tf.contrib.layers.dropout(fc1, keep_prob = 0.55)
	batch7 = tf.contrib.layers.batch_norm(dropf1, scale =True)
	fc2 = tf.contrib.layers.fully_connected(batch7, 512, activation_fn = tf.nn.leaky_relu,weights_regularizer= regularizer)#, scope = "pa2")
	dropf2 = tf.contrib.layers.dropout(fc2, keep_prob = 0.55)
	batch8 = tf.contrib.layers.batch_norm(dropf2, scale =True)
	softmax_linear = tf.contrib.layers.fully_connected(batch8, 20, activation_fn = None)#, scope = "pa2")

	return softmax_linear

def convnet_given_he(x):
	"""The convnet defined in the PA2 pdf

		Parameters:
			x (placeholder): input data

		Returns:
			softmax_linear (numpy array): output of the operations of the conv net with the final layer being linear
	"""
	seed = 1234
	conv1 = tf.contrib.layers.conv2d(x, 32, [5,5], stride = 1, padding = "SAME", activation_fn = tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	batch1 = tf.contrib.layers.batch_norm(conv1, scale =True)
	conv2 = tf.contrib.layers.conv2d(batch1, 32, [5,5], stride = 1, padding = "same", activation_fn = tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	batch2 = tf.contrib.layers.batch_norm(conv2, scale =True)
	pool1 = tf.contrib.layers.max_pool2d(batch2, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	conv3 = tf.contrib.layers.conv2d(pool1, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	batch3 = tf.contrib.layers.batch_norm(conv3, scale =True)
	conv4 = tf.contrib.layers.conv2d(batch3, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	batch4 = tf.contrib.layers.batch_norm(conv4, scale =True)
	pool2 = tf.contrib.layers.max_pool2d(batch4, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	conv5 = tf.contrib.layers.conv2d(pool2, 64, [3,3], stride = 1, padding = "SAME", activation_fn = tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")
	batch5 = tf.contrib.layers.batch_norm(conv5, scale =True)
	conv6 = tf.contrib.layers.conv2d(batch5, 128, [3,3], stride = 1, padding = "VALID", activation_fn = tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal(seed))#, s, weights_initializer = tf.initializers.he_normal(seed)cope = "pa2")
	batch6 = tf.contrib.layers.batch_norm(conv6, scale =True)
	pool3 = tf.contrib.layers.max_pool2d(batch6, kernel_size = [2,2], stride = [2,2], padding = "SAME")#, scope = "pa2")

	flat = tf.contrib.layers.flatten(pool3)
	fc1 = tf.contrib.layers.fully_connected(flat, 256, activation_fn = tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal(seed),weights_regularizer= regularizer)#, scope = "pa2")
	drop1 = tf.contrib.layers.dropout(fc1, keep_prob = 0.6)
	batch7 = tf.contrib.layers.batch_norm(drop1, scale =True)
	fc2 = tf.contrib.layers.fully_connected(batch7, 256, activation_fn = tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal(seed),weights_regularizer= regularizer)#, scope = "pa2")
	drop2 = tf.contrib.layers.dropout(fc2, keep_prob = 0.6)
	batch8 = tf.contrib.layers.batch_norm(drop2, scale =True)
	softmax_linear = tf.contrib.layers.fully_connected(batch8, 20, activation_fn = None, weights_initializer = tf.initializers.he_normal(seed))#, scope = "pa2")

	return softmax_linear



def convnet(x):
	conv1 = tf.contrib.layers.conv2d(x, 32, [3,3], stride = 1, padding = "same", activation_fn = tf.nn.relu)
	max1 = tf.contrib.layers.max_pool2d(inputs = conv1, kernel_size = [2,2], stride = [2,2], padding = "SAME")
	conv2 = tf.contrib.layers.conv2d(max1, 64, [3,3], stride = 1, padding = "same", activation_fn = tf.nn.relu)
	max2 = tf.contrib.layers.max_pool2d(inputs = conv2, kernel_size = [2,2], stride = [2,2], padding = "SAME")
	conv3 = tf.contrib.layers.conv2d(max2, 128, [3,3], stride = 1, padding = "same", activation_fn = tf.nn.relu)
	max3 = tf.contrib.layers.max_pool2d(inputs = conv3, kernel_size = [2,2], stride = [2,2], padding = "SAME")
	conv3_flat = tf.contrib.layers.flatten(inputs = max3) 
	fc1 = tf.contrib.layers.fully_connected(conv3_flat, 128, activation_fn = tf.nn.relu)
	# drop = tf.contrib.layers.dropout(fc1, keep_prob = 0.5)
	output = tf.contrib.layers.fully_connected( fc1, 20, activation_fn = None)
	return output


def convnet_he(x):
	seed = 0
	conv1 = tf.contrib.layers.conv2d(x, 32, [3,3], stride = 1, padding = "same", activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))
	max1 = tf.contrib.layers.max_pool2d(inputs = conv1, kernel_size = [2,2], stride = [2,2], padding = "SAME")
	conv2 = tf.contrib.layers.conv2d(max1, 64, [3,3], stride = 1, padding = "same", activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))
	max2 = tf.contrib.layers.max_pool2d(inputs = conv2, kernel_size = [2,2], stride = [2,2], padding = "SAME")
	conv3 = tf.contrib.layers.conv2d(max2, 128, [3,3], stride = 1, padding = "same", activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))
	max3 = tf.contrib.layers.max_pool2d(inputs = conv3, kernel_size = [2,2], stride = [2,2], padding = "SAME")
	conv3_flat = tf.contrib.layers.flatten(inputs = max3) 
	fc1 = tf.contrib.layers.fully_connected(conv3_flat, 128, activation_fn = tf.nn.relu, weights_initializer = tf.initializers.he_normal(seed))
	# drop = tf.contrib.layers.dropout(fc1, keep_prob = 0.5)
	output = tf.contrib.layers.fully_connected( fc1, 20, activation_fn = None, weights_initializer = tf.initializers.he_normal(seed))
	return output


def flip_images_lr(X):
	"""Returns horizontally flipped images from X

	Parameters:
		X (numpy array): input set of images of shape (num_images, height, width, channels)

	Returns:
		X_flip (numpy array): output set of images of shape (num_images, height, width, channels)
			where the ith image in X_flip corresponds to ith image in X
	"""
	num_images = np.shape(X)[0]
	print(num_images)
	X_flip = np.array([np.fliplr(X[i,:]) for i in range(num_images)])
	return X_flip

def flip_images_ud(X):
	"""Returns vertically flipped images from X

	Parameters:
		X (numpy array): input set of images of shape (num_images, height, width, channels)

	Returns:
		X_flip (numpy array): output set of images of shape (num_images, height, width, channels)
			where the ith image in X_flip corresponds to ith image in X
	"""
	num_images = np.shape(X)[0]
	print(num_images)
	X_flip = np.array([np.flipud(X[i,:]) for i in range(num_images)])
	return X_flip


def rotate_images(X, angle):
	"""Rotate the input images for data augmentation

	Parameters:
		X (numpy array): input set of images of shape (num_images, height, width, channels)
		angle (float): angle of counter clockwise rotation in degrees

	Returns:
		X_rotate (numpy array): rotated images 
	"""
	num_images = np.shape(X)[0]
	X_rotate = np.array([skimage.transform.rotate(X[i,:], angle  = angle, mode = "reflect") for i in range(num_images)])
	return X_rotate


def color_change(X):
	"""Returns differently colored images

	Parameters:
		X (numpy array): input set of images of shape (num_images, height, width, channels)

	Retursn:
		X_color (numpy array): output set of images of shape (num_images, height, width, channels)
			where the ith image in X_color corresponds to the color changed ith image in X
	"""	

	X_color = np.flip(X_train, axis = 3)
	return X_color

def data_aug(X,Y):
	"""Perform data augmentation on the input data

	Parameters:
		X (numpy array): input set of images of shape (num_images, height, width, channels)
		Y (numpy array): input set of labels of shape (num_images, num_classes)

	Returns:
		X_aug (numpy array): output set of images of shape (augmented_num_images, height, width, channels)
		Y_aug (numpy array): output set of labesl of shape (augmented_labels, num_classes)
	"""
	# create data by flipping
	X_flip = flip_images_lr(X)
	Y_flip = Y
	X_aug = np.append(X, X_flip, axis = 0)
	Y_aug = np.append(Y, Y_flip, axis = 0)
	

	# create data by rotation 
	X_rotate = rotate_images(X, 30)
	Y_rotate = Y 
	X_aug = np.append(X_aug, X_rotate, axis = 0)
	Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# # create data by rotation 
	# X_rotate = rotate_images(X, 10)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# # create data by rotation 
	# X_rotate = rotate_images(X, 20)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# # create data by rotation 
	# X_rotate = rotate_images(X, -30)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# # create data by rotation 
	# X_rotate = rotate_images(X, -10)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# # create data by rotation 
	# X_rotate = rotate_images(X, -20)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# X_rotate = rotate_images(X, -15)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# # create data by rotation 
	# X_rotate = rotate_images(X, 15)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# X_color = color_change(X)
	# Y_color = Y 
	# X_aug = np.append(X_color, X_rotate, axis = 0)
	# Y_aug = np.append(Y_color, Y_rotate, axis = 0)

	# X_rotate = rotate_images(X_flip, 30)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# # create data by rotation 
	# X_rotate = rotate_images(X_flip, -20)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# # create data by rotation 
	# X_rotate = rotate_images(X_flip, 20)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)

	# # create data by rotation 
	# X_rotate = rotate_images(X_flip, -30)
	# Y_rotate = Y 
	# X_aug = np.append(X_aug, X_rotate, axis = 0)
	# Y_aug = np.append(Y_aug, Y_rotate, axis = 0)
	

	

	print(np.shape(X_aug), np.shape(Y_aug), "shape data aug")

	return X_aug,Y_aug

############################################## Implementing the parser #############################################

parser = argparse.ArgumentParser()
# TODO: (0) set the parameter defaults to the best value
parser.add_argument("--lr", default = 0.001, help = "learning rate, defaults to 0.01", type = float)
parser.add_argument("--batch_size", default = 256, help = "size of each minibatch, defaults to 256", type = int)
parser.add_argument("--init", default = 1, help = "initialization to be used; 1: Xavier; 2: He; defaults to 1", type = int)
# TODO: (1) find the save_dir path from the directory structure
parser.add_argument("--save_dir", help = "location for the storage of the final model")
parser.add_argument("--epochs", default = 10, help = "number of epochs", type = int)
parser.add_argument("--dataAugment", default = 0, help = "1: use data augmentation, 0: do not use data augmentation", type = int)
parser.add_argument("--train", default = "train.csv", help = "path to the training data")
parser.add_argument("--val", default = "valid.csv", help = "path to the validation data")
parser.add_argument("--test", default = "test.csv", help = "path to the test data")
args = parser.parse_args()


eta = args.lr
batch_size = args.batch_size
# find optimiser name for xavier and he in tf
if args.init == 1:
	initializer = "xavier"
elif args.init == 2:
	initializer = "he"

path_save_dir = args.save_dir
num_epochs = args.epochs

if args.dataAugment == 1:
	dataAugment = True
elif args.dataAugment == 0:
	dataAugment = False

path_train = args.train
path_val = args.val
path_test = args.test


num_classes = 20
shape = (64,64,3)


############################################## Read data #############################################



X_train, Y_train, X_val, Y_val, X_test = read_data(path_train, path_val, path_test, shape)
if dataAugment:
	X_train, Y_train = data_aug(X_train, Y_train)
	print("Data Augmentation used!")
print(np.shape(X_train), np.shape(Y_train), "shape of training data")

shape = (48,48,3)
X_train = X_train[:,8:56,8:56,:]
X_val = X_val[:,8:56,8:56,:]
X_test = X_test[:,8:56,8:56,:]
############################################## Play area #############################################

print("eta", eta)
print("batch_size", batch_size)
# Create placeholders
x, y = create_placeholders(shape, num_classes)

# output of the convolutional network
#if initializer == "xavier":
output = convnet_given(x)
#elif initializer == "he":
#	output = convnet_given_he(x)

pred = tf.nn.softmax(output, name = "predicitons")


# cost and accuracy
cost = tf.losses.softmax_cross_entropy(onehot_labels = y, logits = output, scope = "cost") + tf.losses.get_regularization_loss()
correctpred = tf.equal(tf.argmax(output, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correctpred,tf.float32), name = "accuracy")

# optimizer node
# TODO: (0) try different optimizers

optimizer = tf.train.AdamOptimizer(eta).minimize(cost)

# initialization
init = tf.global_variables_initializer()

tf.summary.scalar("Loss",cost)
tf.summary.scalar("Accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()


# with tf.Session() as sess:
# 	sess.run(init)
# 	for epoch in range(num_epochs):
# 		summary_writer = tf.summary.FileWriter("../Solution/", sess.graph)
# 		_, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict = {x: X_train[:256,:], y: Y_train[:256, :]})
# 		summary_writer.add_summary(summary, epoch)

saver = tf.train.Saver()
path_save_model = join(path_save_dir, "model.ckpt")

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    summary_writer = tf.summary.FileWriter('./output', sess.graph)
    i = 0
    while i < num_epochs:
        for batch in range(len(X_train)//batch_size):
            batch_x = X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
            batch_y = Y_train[batch*batch_size:min((batch+1)*batch_size,len(Y_train))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})

        # loss on the final minibatch alone
        # loss, acc = sess.run([cost,accuracy], feed_dict = {x: X_train, y: Y_train})
        print("Epoch: " + str(i) + "\n" + ", Loss: " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        
        # Calculate accuracy for all validation images
        val_acc, val_loss = sess.run([accuracy,cost], feed_dict={x: X_val,y : Y_val})
        train_loss.append(loss)
        valid_loss.append(val_loss)
        train_accuracy.append(acc)
        valid_accuracy.append(val_acc)
        print("Validation Accuracy:","{:.5f}".format(val_acc))
        # summary_writer.add_summary(summary, epoch)
        print("\n")


        #if i >= 5 and (valid_loss[i] >= valid_loss[i-5]):
        #	i = num_epochs
        i = i + 1
        print("\n")

    save_path = saver.save(sess, path_save_model)
    predictions = sess.run(pred, feed_dict={x: X_test})
    #print(predictions, type(predictions))
    layers = {v.op.name: v for v in tf.trainable_variables()}
    #print(layers)
    weights = [layer for layer in tf.trainable_variables() if layer.name == "Conv/weights:0"][0]
    weights_list = weights.eval()



    summary_writer.close()




print(np.shape(weights_list))
weights_list = np.array(weights_list)
weights_list = weights_list/(np.max(weights_list))
print(np.max(weights_list))
plt.figure(figsize = (10,8))
for i in range(32):
	plt.subplot(4,8,i+1)	
	plt.imshow(weights_list[:,:,:,i])
	plt.axis("off")
plt.savefig("visualisation.pdf", format = "pdf")
plt.show()
plt.close()

num_examples = ((np.shape(X_test)[0]))
Yhat_test_classes = np.argmax(predictions, axis = 1)
output = np.array([range(num_examples), Yhat_test_classes])
output = output.T
sub = pd.DataFrame({"id": output[:,0], "label": output[:,1]})
_ = sub.to_csv("submission_2lay.csv", index = False)
print("Created submission")

# plotting the train and the validation losses

plt.figure(figsize = (10,8))
plt.title("Loss vs Number of epochs")
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.plot(train_loss, label = "train loss")
plt.plot(valid_loss, label = "validation loss")
plt.legend()
plt.savefig("loss_vs_number_of_epochs.pdf", format = "pdf")
plt.show()
plt.close()





