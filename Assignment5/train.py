"""RNNs
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


######################################## Functions ########################################

def softmax(z):
	z = z-np.max(z)
	numer = np.exp(z)
	denom = np.sum(numer, axis = 0) # softmax over each example seprately
	return numer/denom

def find_predictions(rnn_output):
	num_examples = np.shape(rnn_output)[0]
	num_characters = np.shape(rnn_output)[1]
	predictions = []
	for example_num in range(num_examples):
		example = rnn_output[example_num]
		example = np.array(example)
		# print('example', np.shape(example))
		word = []
		for char in range(num_characters):
			character = example[char]
			char_probabs = softmax(character)
			word.append(np.argmax(character))
		predictions.append(word)

	return predictions


def find_accuracy_train(rnn_output, labels):
	predictions = find_predictions(rnn_output)
	# print(predictions[0])
	# print(labels[0])
	max_indices = []
	matches = []
	for ex in range(np.shape(predictions)[0]): # iterating over batch size
		end = labels[ex].index(83) # TODO: assuming eos is 83 in hindi vocab
		max_index = min(end, len(predictions[ex]))
		max_indices.append(max_index) 
		match = np.sum(np.array(predictions[ex][:max_index]) == np.array(labels[ex][:max_index]))
		matches.append(match)


	accuracy = np.sum(matches)/np.sum(max_indices)
	return accuracy

def find_accuracy_true(rnn_output, labels):
	predictions = rnn_output



	# print(predictions[0])
	# print(labels[0])
	max_indices = []
	matches = []
	for ex in range(np.shape(predictions)[0]): # iterating over batch size
		end = labels[ex].index(83) # TODO: assuming eos is 83 in hindi vocab
		max_index = min(end, len(predictions[ex]))
		max_indices.append(max_index) 
		match = np.sum(np.array(predictions[ex][:max_index]) == np.array(labels[ex][:max_index]))
		matches.append(match)


	accuracy = np.sum(matches)/np.sum(max_indices)
	return accuracy



######################################### Parser ##########################################

parser = argparse.ArgumentParser()
# TODO: (0) set the parameter defaults to the best value
parser.add_argument("--lr", default = 0.001, help = "learning rate, defaults to 0.001", type = float)
parser.add_argument("--batch_size", default = 256, help = "size of each minibatch, defaults to 20", type = int)
parser.add_argument("--init", default = 1, help = "initialization to be used; 1: Xavier; 2: uniform random; defaults to 1", type = int)
parser.add_argument("--dropout_prob", help = "the probability of dropping a neuron", type = float, default = 0.8)
parser.add_argument("--save_dir", default = "./save_dir/", help = "lthe directory in which the checkpoints will be saved")
parser.add_argument("--epochs", default = 10, help = "the number of epochs for which model is trained. Epoch: one iteration over training data", type = int)
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


dropout_prob = args.dropout_prob
path_save_dir = args.save_dir

num_epochs = args.epochs

path_train = args.train
path_val = args.val
path_test = args.test


###################################### Messing with data #####################################

# Creating a vocabulary from training data

train_data = pd.read_csv(path_train, encoding = "utf8")
print("Training, Number of English lines: ", len(train_data["ENG"]))
print("Training, Number of Hindi lines: ", len(train_data["HIN"]))


# creating vocabularires. In NMT vocabulary has words. Here, it has letters.
# all letters are  converted to lower case while buliding the vocabulary

vocab_hindi = []
vocab_english = []

for i in range(len(train_data["ENG"])):
	input_string = train_data["ENG"][i].lower()
	characters = input_string.strip().split(" ")
	for character in characters: 
		if not(character in vocab_english):
			vocab_english.append(character)

vocab_english.extend(["<eos>", "<pad>", "<go>", "<unk>"])


for i in range(len(train_data["HIN"])):
	input_string = train_data["HIN"][i]
	characters = input_string.strip().split(" ")
	for character in characters:
		if not (character in vocab_hindi):
			vocab_hindi.append(character)
vocab_hindi.extend(["<eos>", "<pad>", "<go>", "<unk>"])

print("vocab hindi", vocab_hindi)
print("vocab english", vocab_english)
print("size of hindi vocab: ", len(vocab_hindi))
print("size of english vocab", len(vocab_english))


# Creating indices based vectors 

# training data

X_train = []
for i in range(len(train_data["ENG"])):
	input_string = train_data["ENG"][i].lower()
	characters = input_string.strip().split(" ")
	indices = []
	for character in characters:
		if character in vocab_english:
			index = vocab_english.index(character)
			indices.append(index)
		else: 
			indices.append(vocab_english.index("<unk>"))
	X_train.append(indices)



Y_train_loss = []
for i in range(len(train_data["HIN"])):
	input_string = train_data["HIN"][i]
	characters = input_string.strip().split(" ")
	indices = []
	for character in characters:
		if character in vocab_hindi:
			index = vocab_hindi.index(character)
			indices.append(index)
		else: 
			indices.append(vocab_hindi.index("<unk>"))
	indices.append(vocab_hindi.index("<eos>"))
	Y_train_loss.append(indices)

Y_train_decoder = []
for i in range(len(train_data["HIN"])):
	input_string = train_data["HIN"][i]
	characters = input_string.strip().split(" ")
	indices = []
	indices.append(vocab_hindi.index("<go>"))
	for character in characters:
		if character in vocab_hindi:
			index = vocab_hindi.index(character)
			indices.append(index)
		else: 
			indices.append(vocab_hindi.index("<unk>"))
	Y_train_decoder.append(indices)

# validation data

val_data = pd.read_csv(path_val, encoding = "utf8")

print("Validation, Number of English lines: ", len(val_data["ENG"]))
print("Validation, Number of Hindi lines: ", len(val_data["HIN"]))



X_val = []
for i in range(len(val_data["ENG"])):
	input_string = val_data["ENG"][i].lower()
	characters = input_string.strip().split(" ")
	indices = []
	for character in characters:
		if character in vocab_english:
			index = vocab_english.index(character)
			indices.append(index)
		else: 
			indices.append(vocab_english.index("<unk>"))
	X_val.append(indices)

# print(val_data["ENG"][4])
# print(np.shape(X_val[4]))
# print(X_val[4])

Y_val = []
for i in range(len(val_data["HIN"])):
	input_string = val_data["HIN"][i]
	characters = input_string.strip().split(" ")
	indices = []
	for character in characters:
		if character in vocab_hindi:
			index = vocab_hindi.index(character)
			indices.append(index)
		else: 
			indices.append(vocab_hindi.index("<unk>"))
	indices.append(vocab_hindi.index("<eos>"))
	Y_val.append(indices)


Y_val_decoder = []
for i in range(len(val_data["HIN"])):
	input_string = val_data["HIN"][i]
	characters = input_string.strip().split(" ")
	indices = []
	indices.append(vocab_hindi.index("<go>"))
	for character in characters:
		if character in vocab_hindi:
			index = vocab_hindi.index(character)
			indices.append(index)
		else: 
			indices.append(vocab_hindi.index("<unk>"))
	Y_val_decoder.append(indices)


pad_length = 100


########## padding the data ###############3

for i in range(np.shape(X_train)[0]):
	if len(X_train[i]) < pad_length:
		pad_array = [vocab_english.index("<pad>")]*(pad_length - len(X_train[i]))
		X_train[i].extend(pad_array)
# print(np.shape(X_train))
# print(X_train[567])
# print(train_data["ENG"][567])
for i in range(np.shape(X_val)[0]):
	if len(X_val[i]) < pad_length:
		pad_array = [vocab_english.index("<pad>")]*(pad_length - len(X_val[i]))
		X_val[i].extend(pad_array)
# print(np.shape(X_val))
# print(X_val[567])
# print(val_data["ENG"][567])
#for i in range(np.shape(X_test)[0]):
#	if len(X_test[i]) < pad_length:
#		pad_array = [vocab_english.index("<pad>")]*(pad_length - len(X_test[i]))
#		X_test[i].extend(pad_array)


for i in range(np.shape(Y_train_decoder)[0]):
	if len(Y_train_decoder[i]) < pad_length:
		pad_array = [vocab_english.index("<pad>")]*(pad_length - len(Y_train_decoder[i]))
		Y_train_decoder[i].extend(pad_array)


for i in range(np.shape(Y_train_loss)[0]):
	if len(Y_train_loss[i]) < pad_length:
		pad_array = [vocab_english.index("<pad>")]*(pad_length - len(Y_train_loss[i]))
		Y_train_loss[i].extend(pad_array)
# print(np.shape(Y_train))
# print(Y_train[567])
# print(train_data["ENG"][567])
# print(train_data["HIN"][567])

for i in range(np.shape(Y_val)[0]):
	if len(Y_val[i]) < pad_length:
		pad_array = [vocab_english.index("<pad>")]*(pad_length - len(Y_val[i]))
		Y_val[i].extend(pad_array)


for i in range(np.shape(Y_val_decoder)[0]):
	if len(Y_val_decoder[i]) < pad_length:
		pad_array = [vocab_english.index("<pad>")]*(pad_length - len(Y_val_decoder[i]))
		Y_val_decoder[i].extend(pad_array)

# print(np.shape(Y_val))
# print(Y_val[567])
# print(val_data["ENG"][567])
# print(val_data["HIN"][567])

#print(np.shape(X_train), np.shape(X_val), np.shape(X_test))
print(np.shape(Y_train_decoder), np.shape(Y_train_loss), np.shape(Y_val))

###################################### setting the local variables ######################################

vocabsize_english = len(vocab_english)
vocabsize_hindi = len(vocab_hindi)
inembsize = 256 # given
encsize = 256 # given
outembsize = 256

keep_probability = 1 - dropout_prob
seed = 1234

print("Vocabsize_english: ", vocabsize_english)
print("Vocabsize_hindi: ", vocabsize_hindi)
print("keep_probability: ", keep_probability)







###################################### Building the model ######################################

#TODO: (0) padding the data embedding word lookup doesn't accept ids that aren't of the same length
tf.reset_default_graph()

# creating the placeholders 

# first none is batchsize, second none is size of vector
inputs_enc = tf.placeholder(tf.int32, [batch_size, pad_length], name = "input")
inputs_dec = tf.placeholder(tf.int32, [batch_size, None], name = "output_has_go")
target_dec = tf.placeholder(tf.int32, [batch_size, None], name = "targets_for_loss_have_eos")

target_sequence_length = tf.placeholder(tf.int32, [batch_size], name='target_sequence_length')
max_target_len = tf.reduce_max(target_sequence_length) 

keep_prob = tf.placeholder(tf.float32, name = "keep_prob")


	
# Embedding

# embedding at the input (using contrib over tf.nn.lookup)
# input = batchsize x doc_length
# returns batchsize x doc_length x embeddim

embed = tf.contrib.layers.embed_sequence(inputs_enc, 
										vocab_size = vocabsize_english,
										embed_dim = inembsize)





# Encoder

# TODO: (0) Add dropout

########## bidirectional ###########

# applying dropout as specified in the instructions	
forward_cell = tf.contrib.rnn.LSTMCell(encsize)
forward_cell = tf.contrib.rnn.DropoutWrapper(forward_cell, output_keep_prob = keep_prob, seed = seed)
backward_cell = tf.contrib.rnn.LSTMCell(encsize)
backward_cell = tf.contrib.rnn.DropoutWrapper(backward_cell, output_keep_prob = keep_prob, seed = seed)

outputs, states = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, inputs = embed, dtype = tf.float32)
encoder_outputs = tf.concat(outputs, -1)

encoder_state_h = tf.concat([states[0].h, states[1].h], -1)
encoder_state_c = tf.concat([states[0].c, states[1].c], -1)
encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c = encoder_state_c, h = encoder_state_h)




########### unidirectional ############

# forward_cell = tf.contrib.rnn.LSTMCell(512)
# outputs, states = tf.nn.dynamic_rnn(forward_cell, embed, dtype = tf.float32)
# encoder_outputs = outputs
# print(states)





######### decoder training ############## 



decoder_embeddings = tf.Variable(tf.random_uniform([vocabsize_hindi, outembsize]))
decoder_input_embedding = tf.nn.embedding_lookup(decoder_embeddings, inputs_dec)


decoder_cell1 = tf.contrib.rnn.LSTMCell(512)
attention_mechanism = tf.contrib.seq2seq.LuongAttention(512, encoder_outputs)
decoder_cell11 = tf.contrib.seq2seq.AttentionWrapper(decoder_cell1, attention_mechanism)
initial_state = decoder_cell11.zero_state(batch_size, tf.float32).clone(cell_state = encoder_state) 
decoder_cell11 = tf.contrib.rnn.DropoutWrapper(decoder_cell11, output_keep_prob = keep_prob, seed = seed)

decoder_cell2 = tf.contrib.rnn.LSTMCell(512)
decoder_cell2 = tf.contrib.rnn.DropoutWrapper(decoder_cell2, output_keep_prob = keep_prob, seed = seed)

decoder_cell = 	tf.contrib.rnn.MultiRNNCell([decoder_cell11, decoder_cell2])

helper = tf.contrib.seq2seq.TrainingHelper(decoder_input_embedding, target_sequence_length)
output_layer = tf.layers.Dense(vocabsize_hindi)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, (initial_state, encoder_state), output_layer)
decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

# print(decoder_outputs)

######### decoder inference ##############

inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, tf.fill([batch_size], 
															vocab_hindi.index("<go>")), vocab_hindi.index("<eos>"))

inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, inference_helper, (initial_state, encoder_state), output_layer)

inferred_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, impute_finished = True, maximum_iterations = pad_length)


logits_inferred = inferred_output.sample_id

# masks = tf.sequence_mask(target_sequence_length, pad_length, dtype=tf.float32, name='masks')

# loss_inferred = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target_dec, logits = logits_inferred))








######### training ops ##########

logits = decoder_outputs.rnn_output

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target_dec, logits = logits))
optimizer = tf.train.AdamOptimizer(eta)
# global_step = tf.Variable(0, name='global_step', trainable=False)
# params = tf.trainable_variables()
gradients = optimizer.compute_gradients(loss)
capped_gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients if grad is not None]
train_op = optimizer.apply_gradients(capped_gradients)
# 
# train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

init_op = tf.global_variables_initializer()



train_loss = []
val_loss = []
train_acc = []
val_acc = []
val_acc_true = []

saver = tf.train.Saver()
path_save_model = join(path_save_dir, "model.ckpt")

# X_train = X_train[:batch_size]


with tf.Session() as sess:
	writer = tf.summary.FileWriter("output", sess.graph)
	# print(sess.run(decoder))
	sess.run(init_op)

	########## training ########### 
	epoch = 0
	while epoch < num_epochs:
		print("epoch: ", epoch)
		for batch in range(np.shape(X_train)[0]//batch_size):
			print(batch)
			X_train_batch = X_train[batch*batch_size:min((batch+1)*batch_size,np.shape(X_train)[0])]
			Y_train_batch = Y_train_decoder[batch*batch_size:min((batch+1)*batch_size,np.shape(Y_train_decoder)[0])]
			target_batch = Y_train_loss[batch*batch_size:min((batch+1)*batch_size,np.shape(Y_train_decoder)[0])]

			batch_lengths = [len(Y_train_batch[i]) for i in range(np.shape(Y_train_batch)[0])]

			feed_dict = {inputs_enc: X_train_batch, inputs_dec: Y_train_batch, target_dec: target_batch,
						 target_sequence_length: batch_lengths, keep_prob: keep_probability}
			_, loss_epoch = sess.run([train_op, loss], feed_dict = feed_dict)
		

		####### train loss and accuracy at the end of an epoch ######
		print("Training loss: ", loss_epoch)
		train_loss.append(loss_epoch)

		X_train_batch = X_train[:batch_size]
		Y_train_batch = Y_train_decoder[:batch_size]
		target_batch = Y_train_loss[:batch_size]

		batch_lengths = [len(Y_train_batch[i]) for i in range(np.shape(Y_train_batch)[0])]
		feed_dict = {inputs_enc: X_train_batch, inputs_dec: Y_train_batch, target_dec: target_batch, 
					target_sequence_length: batch_lengths, keep_prob: 1.0}
		predictions = sess.run(decoder_outputs, feed_dict = feed_dict)
		predictions_actual = predictions.rnn_output
		# print(np.shape(predictions_actual))
		train_accuracy = find_accuracy_train(predictions_actual, target_batch)
		print("Train Accuracy: ", train_accuracy)
		train_acc.append(train_accuracy)
		


		####### valid loss and accuracy at teh end of an epoch #######

		num_val_batches = np.shape(X_val)[0]//batch_size
		loss_epoch_val = 0

		for batch in range(num_val_batches):
			X_val_batch = X_val[batch*batch_size:min((batch+1)*batch_size,np.shape(X_val)[0])]
			Y_val_batch = Y_val_decoder[batch*batch_size:min((batch+1)*batch_size,np.shape(Y_val_decoder)[0])]
			target_batch = Y_val[batch*batch_size:min((batch+1)*batch_size,np.shape(Y_val_decoder)[0])]

			batch_lengths = [len(Y_val_batch[i]) for i in range(np.shape(Y_val_batch)[0])]
			feed_dict = {inputs_enc: X_val_batch, inputs_dec: Y_val_batch, target_dec: target_batch, 
						target_sequence_length: batch_lengths, keep_prob: 1.0}
			predictions, loss_epoch_temp = sess.run([decoder_outputs, loss], feed_dict = feed_dict)
			predictions_actual = predictions.rnn_output
			loss_epoch_val = loss_epoch_val + loss_epoch_temp

		loss_epoch_val = loss_epoch_val/num_val_batches
		# print(np.shape(predictions_actual))
		print("Validation loss: ", loss_epoch_val)
		
		val_accuracy = find_accuracy_train(predictions_actual, target_batch) # just for the last validation batch
		print("Validation Accuracy: ", val_accuracy)
		

		val_acc.append(val_accuracy)
		val_loss.append(loss_epoch_val)


		num_val_batches = np.shape(X_val)[0]//batch_size
		predictions_all = []
		for batch in range(num_val_batches):
			X_val_batch = X_val[batch*batch_size:min((batch+1)*batch_size,np.shape(X_val)[0])]
			target_batch = Y_val[batch*batch_size:min((batch+1)*batch_size,np.shape(Y_val_decoder)[0])]

			feed_dict = {inputs_enc: X_val_batch, keep_prob: 1.0}
			predictions= sess.run(logits_inferred, feed_dict = feed_dict)
			predictions_all.append(predictions[0])

		val_accuracy_true = find_accuracy_true(predictions_all, Y_val)
		print("true")
		print("True validation accuracy: ", val_accuracy_true)
		val_acc_true.append(val_accuracy_true)


		epoch = epoch + 1


	_ = saver.save(sess, path_save_model)





	writer.close()




######## Plotting training loss ############

plt.figure(figsize = (10,8))
plt.title("Loss")
plt.xlabel("Number of epochs")
plt.ylabel("loss")
plt.plot(train_loss, label = "train loss")
plt.plot(val_loss, label = "validation loss")
plt.legend()
plt.savefig("loss.pdf", format = "pdf")
plt.close()

plt.figure(figsize = (10,8))
plt.title("Accuracy")
plt.xlabel("Number of epochs")
plt.ylabel("accuracy")
plt.plot(train_acc, label = "train accuracy")
plt.plot(val_acc, label = "validation accuracy")
plt.savefig("accuracy.pdf", format = "pdf")
plt.close()
