""" File to train a feedforward neural network of desired architecture with one of Adam, NAG, GD, SGD

Author: Milind Kumar V
"""

import argparse
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd
import pickle 
from sklearn.decomposition import PCA
import sys

# TODO: (5) add docstrings 

# TODO: (6) make plotting function dynamic


def plotfigure(xlabel, ylabel, title, x, y=[], figsize = (10,8), style="k-", graph="plot"):
    plt.figure(figsize=figsize)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(y) == 0:
        plt.plot(x,style)
    else:
        if graph=="plot":
            plt.plot(x,y,style)
        if graph== "semilogx":
            plt.semilogx(x,y,style)
        if graph== "semilogy":
            plt.semilogy(x,y,style)
        if graph== "loglog":
            plt.loglog(x,y,style)
    plt.tight_layout()
    plt.show()
    plt.close()


# Function seems to be working okay, cross verified 
def createnetwork(num_hidden, activation_func, sizes, inputsize = 784, outputsize = 10):
    sizes = [inputsize] + sizes
    sizes = sizes + [outputsize]
    np.random.seed(1234)
    parameters = {}
    if activation_func == "relu":
        for i in range(1, num_hidden+2):
            parameters["W" + str(i)] = 0.01*np.random.randn(sizes[i], sizes[i-1])*(np.sqrt(2/(sizes[i] + sizes[i-1])))
            parameters["b" + str(i)] = np.zeros((sizes[i],1))
    else:
        for i in range(1, num_hidden+2):
            parameters["W" + str(i)] = np.random.randn(sizes[i], sizes[i-1])
            parameters["b" + str(i)] = np.random.randn(sizes[i],1)
        # TODO: (5) scale these by 0.01 like in andrew ng's course?

    return parameters


# Takes in vector arguments and reads out vector arguments
def sigmoid(z):
	return 1/(1 + np.exp(-z))


def tanh(z):
	return np.tanh(z)
    # (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
    # my defintion produced nans

# TODO: (0) bug fix in case of matrices
def softmax(z):
    z = z-np.max(z)
    numer = np.exp(z)
    denom = np.sum(numer, axis = 0) # softmax over each example seprately
    return numer/denom


# TODO: (1) One hot conversion: row vs column, important to assess
# TODO: (2) Verify that the labels lie in [0,9] and not [1,10]
# Number of classes included in arguments to avoid confusion due to skewed data
def convert_to_onehot(indices, num_classes):
    # borrowed from stack overflow
    output = np.eye(num_classes)[np.array(indices).reshape(-1)]
    # the reshape just seems to verify the shape of the matrix
    # each target vector is converted to a row vector
    return output.reshape(list(np.shape(indices))+[num_classes])

# TODO: (0) verify that the shapes of X and Y are the same!!!
# TODO: (2) verify the definition of the loss function
# TODO: (4) verify that all the loss functions are working fine
def squared_loss(X, Y):
    x = np.array(X)
    y = np.array(Y)
    # to hold off broadcasting behaviour in case of 1D arrays
    if x.ndim == 1:
        x = x[:,np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    loss = 0.5*np.sum((y-x)**2)
    return loss


def cross_entropy_loss(X, Y):
    x = np.array(X)
    y = np.array(Y)
    # to hold off broadcasting behaviour in case of 1D arrays
    if x.ndim == 1:
        x = x[:,np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    loss_vec = -1*y*np.log(x)
    return np.sum(loss_vec) 
    # alternate implementation
    # x = np.array(X).reshape(-1)
    # y = np.array(Y).reshape(-1)
    # logx = np.log(x)
    # loss_vec = (-1)*(y*logx)
    # loss = np.sum(loss_vec)
    # return loss    


def read_data(path_to_csv):
    df = pd.read_csv(path_to_csv)
    data = df.to_numpy()
    X = data[:,1:-1]
    y = data[:,-1]
    Y = [int(i) for i in y] # TODO: (3) Check if this conversion to int is always valid!
    return data, X.T, Y # TODO: (0) convert this to TRANSPOSE ASAP!!! returns a matrix of size (no. of feat., num of eg.)

def activate(z, activation):
    if activation == "sigmoid":
        return sigmoid(z)
    elif activation == "tanh":
        return tanh(z)
    elif activation == "relu":
        return (z>0)*(z) + 0.01*((z<0)*z)


# TODO: (0) Fix bugs in case of single example. Keemdims = true.

def forward_pass(X, parameters, activation, num_hidden):
    A = {}
    # To prevent broadcasting when a single input vector is given
    if X.ndim == 1:
        X = X[:, np.newaxis] 
    H = {"h0":X}
    for l in range(1, num_hidden + 2):
        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]
        
        hprev = H["h" + str(l-1)]
        al = np.dot(Wl,hprev) + bl
        A["a" + str(l)] = al
        if l != num_hidden + 1: 
            hl = activate(al, activation)
        elif l == num_hidden + 1:
            hl = softmax(al)
        H["h" + str(l)] = hl
        # TODO: (10) remove this print statement
    #    print(l, "Wl", np.shape(Wl), "bl", np.shape(bl), "hprev", np.shape(hprev), "al", np.shape(al), "hl", np.shape(hl))
    yhat = H["h" + str(num_hidden + 1)]
    return yhat, A, H 

def creategrads(num_hidden, sizes, inputsize = 784, outputsize = 10):
    sizes = [inputsize] + sizes
    sizes = sizes + [outputsize]
    grads = {"dh0":np.zeros((inputsize,1)),
            "da0":np.zeros((inputsize,1))}
    for i in range(1, num_hidden+2):
        grads["dW" + str(i)] = np.zeros((sizes[i], sizes[i-1]))
        grads["db" + str(i)] = np.zeros((sizes[i],1))
        grads["da" + str(i)] = np.zeros((sizes[i],1))
        grads["dh" + str(i)] = np.zeros((sizes[i],1))
        #print("shape of gradients", np.shape(grads["dW" + str(i)]), np.shape(grads["db" + str(i)]),np.shape(grads["da" + str(i)]),np.shape(grads["dh" + str(i)]))
        # TODO: (5) scale these by 0.01 like in andrew ng's course?

    return grads


def grad_sigmoid(z):
    return (sigmoid(z))*(1 - sigmoid(z))

def grad_tanh(z):
    return (1 - (np.tanh(z))**2)

def grad_relu(z):
    return (z>0)*(np.ones(np.shape(z))) + (z<0)*(0.01*np.ones(np.shape(z)))

# backprop with one example only
def back_prop(H, A, parameters, num_hidden, sizes, Y, Yhat, loss, activation, inputsize, outputsize):
    # gradient with respect to a_(numhidden+1)
    grad_one_eg = creategrads(num_hidden, sizes, inputsize = 784, outputsize = 10)
    # TODO: (0) handling a0!!!
    A["a0"] = np.zeros((inputsize,1))
    # Is this really necessary
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Yhat.ndim == 1:
        Yhat = Yhat[:, np.newaxis]

    if loss == "ce":
        grad_one_eg["da" + str(num_hidden + 1)] = Yhat - Y
    elif loss == "sq":
        grad_one_eg["da" + str(num_hidden + 1)] = (Yhat - Y)*Yhat - Yhat*(np.dot((Yhat-Y).T, Yhat))
    # TODO: (6) remove print
    # print(Yhat - Y, "diff", Yhat, "Yhat")

    for i in np.arange(num_hidden + 1, 0, -1):
        # TODO: (0) matmul?? Safe?
            grad_one_eg["dW" + str(i)] = np.dot(grad_one_eg["da" + str(i)], (H["h" + str(i-1)]).T)
            grad_one_eg["db" + str(i)] = grad_one_eg["da" + str(i)]

            grad_one_eg["dh" + str(i-1)] = np.dot((parameters["W" + str(i)]).T, grad_one_eg["da" + str(i)])

            if activation == "sigmoid":
                derv = grad_sigmoid(A["a" + str(i-1)])
            elif activation == "tanh":
                derv = grad_tanh(A["a" + str(i-1)])
            elif activation == "relu":
                derv = grad_relu(A["a" + str(i-1)])
            if derv.ndim == 1:
                derv = derv[:, np.newaxis]

            grad_one_eg["da" + str(i-1)] = (grad_one_eg["dh" + str(i-1)])*derv

    return grad_one_eg


def createmomenta(num_hidden, sizes, inputsize = 784, outputsize = 10):
    sizes = [inputsize] + sizes
    sizes = sizes + [outputsize]
    momenta = {}
    for i in range(1, num_hidden+2):
        momenta["vW" + str(i)] = np.zeros((sizes[i], sizes[i-1]))
        momenta["vb" + str(i)] = np.zeros((sizes[i],1))
        # TODO: (5) scale these by 0.01 like in andrew ng's course?

    return momenta

def createmomenta_squared(num_hidden, sizes, inputsize = 784, outputsize = 10):
    sizes = [inputsize] + sizes
    sizes = sizes + [outputsize]
    momenta = {}
    for i in range(1, num_hidden+2):
        momenta["mW" + str(i)] = np.zeros((sizes[i], sizes[i-1]))
        momenta["mb" + str(i)] = np.zeros((sizes[i],1))
        # TODO: (5) scale these by 0.01 like in andrew ng's course?

    return momenta

def find_accuracy(yhat,y):
    a = np.argmax(yhat, axis = 0)
    # print(a)
    b = np.argmax(y, axis = 0)
    # print(np.shape(a), np.shape(b))
    return 100*(np.sum(a == b)/len(a))

def read_data_test(path_to_csv):
    df = pd.read_csv(path_to_csv)
    data = df.to_numpy()
    X = data[:,1:]
    indices = data[:,0]
    return data, X.T, indices


def measure_performance(X, Y, X_val, Y_val, params, activation_func, num_hidden, loss):
    Yhat, _, _ = forward_pass(X, params, activation_func, num_hidden)
    train_acc = find_accuracy(Yhat, Y)
    train_err = 100 - train_acc
    if loss == "ce":
        train_loss = (cross_entropy_loss(Yhat,Y))
    elif loss == "sq":
        train_loss = (squared_loss(Yhat,Y))
    
    Yhat_val, _, _ = forward_pass(X_val, params, activation_func, num_hidden)
    val_acc = find_accuracy(Yhat_val, Y_val)
    val_err = 100 - val_acc
    if loss == "ce":
        valid_loss = cross_entropy_loss(Yhat_val, Y_val)
    elif loss == "sq":
        valid_loss = (squared_loss(Yhat_val, Y_val))

    return train_err, train_loss, val_err, valid_loss

def display_info(epoch, train_err, train_loss, val_err, valid_loss):
    print("epoch:" + str(epoch))       
    print("train error: ", "%.2f" % train_err, " train loss: ", "%.2f" % train_loss, 
        " validation error: ", "%.2f" % val_err, "valid loss: ", "%.2f" % valid_loss)
    print("\n")    

def create_submission(X_test, indices, params, activation_func, num_hidden, submission_path):
    Yhat_test, _, _ = forward_pass(X_test, params, activation_func, num_hidden)
    Yhat_test_classes = np.argmax(Yhat_test, axis = 0)
    output = np.array([indices, Yhat_test_classes])
    output = output.T
    sub = pd.DataFrame({"id": output[:,0], "label": output[:,1]})
    _ = sub.to_csv(submission_path, index = False)
    print("Created submission at " + submission_path)


def create_log_files(path_expt_dir, step_data):
    # TODO: (3) Handle nested directories
    try:
        os.mkdir(path_expt_dir)
        print("expt_dir created at " + path_expt_dir)
    except FileExistsError:
        print("expt_dir already exists at " + path_expt_dir)

    f = open(path_expt_dir + str("log_train.txt"), "w")
    for step in step_data:
        line = "Epoch " + str(step[0]) + ", Step " + str(step[1]) 
        line = line +  ", Loss: " + str(np.round(step_data[step][0], decimals = 2)) 
        line = line + ", Error: " + str(np.round(step_data[step][1], decimals = 2))
        line = line + ", lr: " + str(step_data[step][4])
        line = line + "\n" 
        f.write(line)
    f.close()
    
    f = open(path_expt_dir + str("log_val.txt"), "w")
    for step in step_data:
        line = "Epoch " + str(step[0]) + ", Step " + str(step[1]) 
        line = line +  ", Loss: " + str(np.round(step_data[step][2], decimals = 2)) 
        line = line + ", Error:" + str(np.round(step_data[step][3], decimals = 2))
        line = line + ", lr: " + str(step_data[step][4]) 
        line = line + "\n"
        f.write(line)
    f.close()
    print("log files created")


def create_readme(path_expt_dir, run_details):
    try:
        os.mkdir(path_expt_dir)
        print("expt_dir created at " + path_expt_dir)
    except FileExistsError:
        print("expt_dir already exists at " + path_expt_dir)
    f = open(path_expt_dir + str("readme.txt"), "w")
    f.write(run_details + "\n")
    f.write
    f.close()





def adam(X, Y, X_val, Y_val, activation_func, loss_func, eta, num_epochs, num_hidden, sizes, batch_size, path_save_dir,
 inputsize = 784, outputsize = 10, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, anneal = True, pretrain = False, state = 0):
    print("Adam")
    # data for information display and plots 
    step_data = {}
    epoch_data = []

    # TODO: (3) remove these comments
    # beta1 = 0.9
    # beta2 = 0.999
    # eps = 1e-8


    # initializing
    if pretrain == False:
        params = createnetwork(num_hidden, activation_func, sizes, inputsize, outputsize)
    elif pretrain == True:
        params = load_params(path_save_dir, state)

    prev_momenta = createmomenta(num_hidden, sizes, inputsize, outputsize)
    momenta = createmomenta(num_hidden, sizes, inputsize, outputsize)
    momenta_hat = createmomenta(num_hidden, sizes, inputsize, outputsize)

    prev_momenta_squared = createmomenta_squared(num_hidden, sizes, inputsize, outputsize)
    momenta_squared = createmomenta_squared(num_hidden, sizes, inputsize, outputsize)
    momenta_squared_hat = createmomenta_squared(num_hidden, sizes, inputsize, outputsize)
    pointsseen = 0
    epoch = 0
    while epoch < (num_epochs):
        grads = creategrads(num_hidden, sizes, inputsize, outputsize)
        step = 0
        for j in range(0, 55000):
            x = X[:,j]
            y = Y[:,j]
            yhat, A, H = forward_pass(x, params, activation_func, num_hidden)
            grad_current = back_prop(H, A, params, num_hidden, sizes, y, yhat, loss_func, activation_func, inputsize, outputsize)
            for key in grads:
                grads[key] = grads[key] + grad_current[key]

            pointsseen = pointsseen + 1

            # check if one step is done, update parameters, initialize new gradients
            if pointsseen% batch_size == 0:
                step = step + 1

                for newkey in params:
            # TODO: (0) add batch_size wala division
                    momenta["v" + newkey] = beta1*prev_momenta["v" + newkey] + (1 - beta1)*grads["d" + newkey]
                    momenta_squared["m" + newkey] = beta2*prev_momenta_squared["m" + newkey] + (1 - beta2)*((grads["d" + newkey])**2)

                    momenta_hat["v" + newkey] = momenta["v" + newkey]/(1 - np.power(beta1, step)) 
                    momenta_squared_hat["m" + newkey] = momenta_squared["m" + newkey]/(1 - np.power(beta2, step))

                    params[newkey] = params[newkey] - (eta/np.sqrt(momenta_squared_hat["m" + newkey] + eps))*momenta_hat["v" + newkey]

                    prev_momenta["v" + newkey] = momenta["v" + newkey]
                    prev_momenta_squared["m" + newkey] = momenta_squared["m" + newkey]
                     
                grads = creategrads(num_hidden, sizes, inputsize, outputsize)

                
                # store data for log files if 100 steps are done
                if step%100 == 0:
                    train_err, train_loss, val_err, valid_loss = measure_performance(X, Y, X_val, Y_val, params, activation_func, num_hidden, loss_func)
                    step_data[(epoch, step)] = [train_loss, train_err, valid_loss, val_err, eta]


        train_err, train_loss, val_err, valid_loss = measure_performance(X, Y, X_val, Y_val, params, activation_func, num_hidden, loss_func)

        if anneal and epoch >=1 and epoch_data[epoch - 1][2] <= valid_loss:
            eta = eta/2
            params = load_params(path_save_dir, epoch - 1)
            epoch = epoch - 1
            print("anneal")
        else: 
            display_info(epoch, train_err, train_loss, val_err, valid_loss)
            epoch_data.append([epoch, train_loss, valid_loss])
            pickle_params(params, epoch, path_save_dir)
       
        # train_err, train_loss, val_err, valid_loss = measure_performance(X, Y, X_val, Y_val, params, activation_func, num_hidden)
        # display_info(epoch, train_err, train_loss, val_err, valid_loss)
        # epoch_data.append([epoch, train_loss, valid_loss])

        epoch = epoch + 1
    return params, step_data, epoch_data



# TODO: (0) review updates for pickling made in a hurry
def mgd(X, Y, X_val, Y_val, activation_func, loss_func, eta, gamma, num_epochs, num_hidden, sizes, batch_size, path_save_dir,
 inputsize = 784, outputsize = 10, anneal = True, pretrain = False, state = 0):
    print("momentum gradient descent")
    # data for information display and plots 
    step_data = {}
    epoch_data = []

    # initializing
    if pretrain == False:
        params = createnetwork(num_hidden, activation_func, sizes, inputsize, outputsize)
    elif pretrain == True:
        params = load_params(path_save_dir, state)


    prev_momenta = createmomenta(num_hidden, sizes, inputsize, outputsize)
    momenta = createmomenta(num_hidden, sizes, inputsize, outputsize)
    pointsseen = 0

    epoch = 0
    while epoch < num_epochs:
        grads = creategrads(num_hidden, sizes, inputsize, outputsize)
        step = 0
        for j in range(0, 55000):
            x = X[:,j]
            y = Y[:,j]
            yhat, A, H = forward_pass(x, params, activation_func, num_hidden)
            grad_current = back_prop(H, A, params, num_hidden, sizes, y, yhat, loss_func, activation_func, inputsize, outputsize)
            for key in grads:
                grads[key] = grads[key] + grad_current[key]

            pointsseen = pointsseen + 1

            # check if one step is done, update parameters, initialize new gradients
            if pointsseen% batch_size == 0:
                for newkey in params:
            # TODO: (0) add batch_size wala division
                    momenta["v" + newkey] = gamma*prev_momenta["v" + newkey] + eta*grads["d" + newkey]
                    params[newkey] = params[newkey] - momenta["v" + newkey]
                    prev_momenta["v" + newkey] = momenta["v" + newkey] 
                grads = creategrads(num_hidden, sizes, inputsize, outputsize)

                step = step + 1
                # store data for log files if 100 steps are done
                if step%100 == 0:
                    train_err, train_loss, val_err, valid_loss = measure_performance(X, Y, X_val, Y_val, params, activation_func, num_hidden, loss_func)
                    step_data[(epoch, step)] = [train_loss, train_err, valid_loss, val_err, eta]
       
        train_err, train_loss, val_err, valid_loss = measure_performance(X, Y, X_val, Y_val, params, activation_func, num_hidden, loss_func)
        
        if anneal and epoch >=1 and epoch_data[epoch - 1][2] <= valid_loss:
            eta = eta/2
            params = load_params(path_save_dir, epoch - 1)
            epoch = epoch - 1
        else: 
            display_info(epoch, train_err, train_loss, val_err, valid_loss)
            epoch_data.append([epoch, train_loss, valid_loss])
            pickle_params(params, epoch, path_save_dir)

        epoch = epoch + 1

    return params, step_data, epoch_data


def sgd(X, Y, X_val, Y_val, activation_func, loss_func, eta, num_epochs, num_hidden, sizes, batch_size, path_save_dir,
 inputsize = 784, outputsize = 10, anneal = True, pretrain = False, state = 0):
    print("minibatch")
    step_data = {}
    epoch_data = []
    
    if pretrain == False:
        params = createnetwork(num_hidden, activation_func, sizes, inputsize, outputsize)
    elif pretrain == True:
        params = load_params(path_save_dir, state)


    pointsseen = 0
    epoch = 0
    while epoch < num_epochs:
        # creation of zero grad vectors at the start of every epoch
        grads = creategrads(num_hidden, sizes, inputsize, outputsize)
        step = 0

        # iterate through every data point
        for j in range(0, 55000):
            x = X[:,j]
            y = Y[:,j]
            
            # perform forward pass
            yhat, A, H = forward_pass(x, params, activation_func, num_hidden)

            # compute gradients and update them over examples
            grad_current = back_prop(H, A, params, num_hidden, sizes, y, yhat, loss_func, activation_func, inputsize, outputsize)
            for key in grads:
                grads[key] = grads[key] + grad_current[key]

            pointsseen = pointsseen + 1

            # check if one step is done, update parameters, initialize new gradients
            if pointsseen% batch_size == 0:
                for newkey in params:
            # TODO: (0) check this batch_size wala division
                    params[newkey] = params[newkey] - eta*(grads["d" + newkey]/batch_size) 
                grads = creategrads(num_hidden, sizes, inputsize, outputsize)

                # entering the if implies one step is done, so update step
                step = step + 1
                # store data for log files if 100 steps are done
                if step%100 == 0:
                    train_err, train_loss, val_err, valid_loss = measure_performance(X, Y, X_val, Y_val, params, activation_func, num_hidden, loss_func)
                    step_data[(epoch, step)] = [train_loss, train_err, valid_loss, val_err, eta]

        train_err, train_loss, val_err, valid_loss = measure_performance(X, Y, X_val, Y_val, params, activation_func, num_hidden, loss_func)

        if anneal and epoch >=1 and epoch_data[epoch - 1][2] <= valid_loss:
            eta = eta/2
            params = load_params(path_save_dir, epoch - 1)
            epoch = epoch - 1
        else: 
            display_info(epoch, train_err, train_loss, val_err, valid_loss)
            epoch_data.append([epoch, train_loss, valid_loss])
            pickle_params(params, epoch, path_save_dir)

        epoch = epoch + 1
    return params, step_data, epoch_data



def nag(X, Y, X_val, Y_val, activation_func, loss_func, eta, gamma, num_epochs, num_hidden, sizes, batch_size, path_save_dir,
 inputsize = 784, outputsize = 10, anneal = True, pretrain = False, state = 0):
    print("NAG")
    # data for information display and plots 
    step_data = {}
    epoch_data = []

    # initializing
    if pretrain == False:
        params = createnetwork(num_hidden, activation_func, sizes, inputsize, outputsize)
    elif pretrain == True:
        params = load_params(path_save_dir, state)

    prev_momenta = createmomenta(num_hidden, sizes, inputsize, outputsize)
    momenta = createmomenta(num_hidden, sizes, inputsize, outputsize)

    pointsseen = 0
    epoch = 0
    while epoch < (num_epochs):
        grads = creategrads(num_hidden, sizes, inputsize, outputsize)
        step = 0
        for j in range(0, 55000):
            x = X[:,j]
            y = Y[:,j]
            yhat, A, H = forward_pass(x, params, activation_func, num_hidden)
            grad_current = back_prop(H, A, params, num_hidden, sizes, y, yhat, loss_func, activation_func, inputsize, outputsize)
            for key in grads:
                grads[key] = grads[key] + grad_current[key]

            pointsseen = pointsseen + 1

            # check if one step is done, update parameters, initialize new gradients
            if pointsseen% batch_size == 0:
                step = step + 1

                for newkey in params:
            # TODO: (0) add batch_size wala division
                    momenta["v" + newkey] = gamma*prev_momenta["v" + newkey] + eta*grads["d" + newkey]
                    params[newkey] = params[newkey] - momenta["v" + newkey]
                    prev_momenta["v" + newkey] = momenta["v" + newkey] 

                    
                     
                grads = creategrads(num_hidden, sizes, inputsize, outputsize)

                
                # store data for log files if 100 steps are done
                if step%100 == 0:
                    train_err, train_loss, val_err, valid_loss = measure_performance(X, Y, X_val, Y_val, params, activation_func, num_hidden, loss_func)
                    step_data[(epoch, step)] = [train_loss, train_err, valid_loss, val_err, eta]

                # TODO: review what happens in the last step
                for next_key in params:
                    momenta["v" + next_key] = gamma*prev_momenta["v" + next_key]
                    params[next_key] = params[next_key] - momenta["v" + next_key]

        train_err, train_loss, val_err, valid_loss = measure_performance(X, Y, X_val, Y_val, params, activation_func, num_hidden, loss_func)

        if anneal and epoch >=1 and epoch_data[epoch - 1][2] <= valid_loss:
            eta = eta/2
            params = load_params(path_save_dir, epoch - 1)
            epoch = epoch - 1
            print("anneal")
        else: 
            display_info(epoch, train_err, train_loss, val_err, valid_loss)
            epoch_data.append([epoch, train_loss, valid_loss])
            pickle_params(params, epoch, path_save_dir)


        epoch = epoch + 1
    return params, step_data, epoch_data


# Read train, validation and test data
def init_data(path_train, path_val, path_test):
    data, X, y = read_data(path_train)
    
    Y = (convert_to_onehot(y, 10)).T
    X = (X.T/255)
    pca = PCA(n_components=50)
    pca.fit(X)
    X = pca.transform(X)
    X = X.T
    print(np.shape(X), np.shape(y), np.shape(data))

    data, X_val, y_val = read_data(path_val)
    
    Y_val = (convert_to_onehot(y_val, 10)).T
    X_val = (X_val.T/255)
    X_val = pca.transform(X_val)
    X_val = X_val.T
    print(np.shape(X_val), np.shape(y_val), np.shape(data))


    data, X_test, indices = read_data_test(path_test)

    X_test = X_test.T/255
    X_test = pca.transform(X_test)
    X_test = X_test.T
    #print(np.shape(X_test), "shape of test data")
    #np.savetxt("test_2.csv",X_test, delimiter = ",")

    return X, Y, X_val, Y_val, X_test, indices


def pickle_params(params, epoch, path_save_dir):
    try:
        os.mkdir(path_save_dir)
        print("save_dir created at " + path_save_dir)
    except FileExistsError:
        print("save_dir already exists at " + path_save_dir)
    filename = path_save_dir + "weights_" + str(epoch) + ".pickle"    
    with open(filename, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_params(path_save_dir, epoch):
    if os.path.isdir(path_save_dir):
        filename = path_save_dir + "weights_" + str(epoch) + ".pickle"
        with open(filename, 'rb') as handle:
            parameters = pickle.load(handle)
        return parameters 
    else:
        print("No directory at " + path_save_dir)

# writing the parser 

parser = argparse.ArgumentParser()
# TODO: (0) set types to each of these
parser.add_argument("--lr", help = "learning rate", type = float, default = 0.001)
parser.add_argument("--momentum", help = "value gamma for momentum", type = float, default = 0.5)
parser.add_argument("--num_hidden", help = "number of hidden layers between the input layer and output", type = int)
parser.add_argument("--sizes", help = "sizes of each of the hidden layers")
parser.add_argument("--activation", help = "non-linear activation for each neuron", default = "relu")
parser.add_argument("--loss", help = "loss function to be optimized", default = "ce")
parser.add_argument("--opt", help = "determines the type of optimizer: Adam, NAG, GD, momentum", default = "adam")
parser.add_argument("--batch_size", help = "size of each minibatch", type = int, default = 20 )
# TODO: (5) make sure that all are multiples of 5
parser.add_argument("--epochs", help = "number of epochs", type = int, default = 10)
# TODO: (5) checking for boolean etc
parser.add_argument("--anneal", help = "determines whether or not learning rate will be halved" )
parser.add_argument("--save_dir", help = "location for the storage of the final model")
parser.add_argument("--expt_dir", help = "location for the storage of the final logs")
parser.add_argument("--train", help = "path to the traing dataset")
parser.add_argument("--val", help = "path to the validation dataset")
parser.add_argument("--test", help = "path to the test dataset")
parser.add_argument("--pretrain", help = "determines whether or not to use pretrained model")
parser.add_argument("--state", help = "epoch from which the saved weights needed to be loaded", type = int)
parser.add_argument("--testing", help ="determines if there will be training or not")
args = parser.parse_args()


inter = args.testing
if inter == None:
	inter = "false"

if inter.lower() == "true":
    testing = True
else:
    testing = False

# setting variables globally 
if testing == False:
    eta = args.lr # learning rate
    gamma = args.momentum # momentum
    num_hidden = args.num_hidden
    # TODO: (5) assert that size of this array is the same as num_hidden
    sizes = [int(k) for k in args.sizes.split(",")]
    activation_func = args.activation
    loss_func = args.loss
    optim = args.opt
    batch_size = args.batch_size
    num_epochs = args.epochs

    if args.anneal.lower() == "true":
        anneal = True
    else:
        anneal = False

path_save_dir = args.save_dir
path_expt_dir = args.expt_dir
path_train = args.train
path_val = args.val
path_test = args.test

inter2 = args.pretrain
if inter2 == None:
	inter2 = "false"

if inter2.lower() == "true":
    pretrain = True
else:
    pretrain = False

state = args.state



# TODO: (10) the print
if testing == False:
    print(eta, gamma, num_hidden, sizes, activation_func, loss_func, optim, batch_size, num_epochs, path_save_dir,
      path_expt_dir, path_train, path_val, path_test)
    run_details = "eta: " + str(eta)  + " gamma: " + str(gamma) + " num_hidden: " + str(num_hidden)  + " sizes: " + str(sizes)  + " activation_func: " + str(activation_func)  + " loss_func: "  + str(loss_func) + " optim: " + str(optim) +  " batch_size: " + str(batch_size) + " num_epochs: " + str(num_epochs)
#################################   Play area #######################################

# Read data
if testing == False:
    X, Y, X_val, Y_val, X_test, indices = init_data(path_train, path_val, path_test)
elif testing == True:
    data = pd.read_csv(path_test, header = None)
    X_test = data.to_numpy()
    indices = np.arange(np.shape(data)[1])

# Train
if testing == False:
    if optim == "gd":
        params, step_data, epoch_data = sgd(X, Y, X_val, Y_val, activation_func, loss_func, eta, num_epochs, num_hidden, sizes, batch_size, path_save_dir, inputsize = np.shape(X)[0], anneal = anneal, pretrain = pretrain, state = state)
    elif optim == "momentum":
        # TODO: (0) add the anneal,save and stuff to momentum and all that
        params, step_data, epoch_data = mgd(X, Y, X_val, Y_val, activation_func, loss_func, eta, gamma, num_epochs, num_hidden, sizes, batch_size, path_save_dir, inputsize = np.shape(X)[0], anneal = anneal, pretrain = pretrain, state = state)
    elif optim == "adam":
        params, step_data, epoch_data = adam(X, Y, X_val, Y_val, activation_func, loss_func, eta, num_epochs, num_hidden, sizes, batch_size, path_save_dir, inputsize = np.shape(X)[0], anneal = anneal, pretrain = pretrain, state = state)
    elif optim == "nag":
        params, step_data, epoch_data = nag(X, Y, X_val, Y_val, activation_func, loss_func, eta, gamma, num_epochs, num_hidden, sizes, batch_size, path_save_dir, inputsize = np.shape(X)[0], anneal = anneal, pretrain = pretrain, state = state)
    # Test
    create_log_files(path_expt_dir, step_data)
    create_readme(path_expt_dir, run_details)
    submission_path = path_expt_dir + "test_submission.csv"
else:
    params = load_params(path_save_dir, state)
    num_hidden = int(len(params.keys())/2 - 1)
    print("num_hidden", num_hidden)
    activation_func = "relu"
    submission_path = path_expt_dir + "predictions_" + str(state) + ".csv"


create_submission(X_test, indices, params, activation_func, num_hidden, submission_path)


