# Directory: Assignment6

## Assignment 6

This is a programming assignment in which RBMs are trained using the Contrastive Divergence algorithm. Following this, hidden representations are computed for each available data point and visualized using t-SNE. It is expected that hidden representations of points associated with the same class will lie close to each other and this is indeed observed. Following this, the trained RBMs are used to generate new data and the similarity of the generated images is visually assessed. While most of the code is written in NumPy, scikit-learn is used to implement the same algorithm for MNIST handwritten digits data and the results are presented in the report.

- [Problem statement](https://drive.google.com/file/d/1ooP4zlA3Q-PB8djpD6HDUuD_d3Ex6zjk/view?usp=sharing)    
- [My report](https://drive.google.com/file/d/1MnY_QpCQimu4Or-VC1ojyRppfJvKvRjA/view?usp=sharing)
- [Training data](https://drive.google.com/file/d/1CtYzz3axS3603qfqfS2IC0MuahQzOodE/view?usp=sharing)    
- [Test data](https://drive.google.com/file/d/18KdjrFoKS5aQ3PB-590Yyo7od3f1EiAu/view?usp=sharing)    

## Contents 

### train.py 

This script contains the code to train a RBM, compute the hidden representations and visualise the representations using TSNE for the test data.

#### Usage 

Run as 
```
python train.py --n <n> --k <k> --eta <learning_rate> --num_epochs <num_epochs> --path_train <path_train> --path_test <path_test> 
```

__n__: number of hidden units in the RBM, defaults to 100        
__k__: number of steps of contrastive divergence to run per example, defaults to 1         
__learning_rate__: learning rate for contrastive divergence, defaults to 0.001         
__num_epochs__: number of iterations over the entire training data to be performed during the training process, defaults to 1    
__path_train__: path to training data, defaults to "./train.py"     
__path_test__: path to the test data, defaults to "./test.py"     

#### Outputs

The script creaates a folder with the name ```k<value_of_k> n<value_of_n> eta<eta> epochs<num_epochs>``` in the working directory which contains the following files:    

__clusters_test.png__: a plot of the TSNE embeddings of the hidden representations of the test data.   
__original.png__: one of the images from the dataset which is used to study the convergence of the CD algorithm. The image chosen can be changed by changing the variable image_id in the script.
__origafter.png__: image used to generate another image after the training of the RBM.
__origarecon.png__: image generated after sampling from the learnt distribution.
__changing\_image*n*.png__: an 8 x 8 plot of images each obtained after CD has been performed with 936 examples (there are 60000 training examples) in hte n_th epoch (indexing starts from 0). ```num_epochs``` number of plots are generated in the directory.


### train_loss.py 

This script contains the code to train a RBM and plot the resulting learning curves.

#### Usage 

Run as 
```
python train_loss.py --n <n> --k <k> --eta <learning_rate> --num_epochs <num_epochs> --path_train <path_train> --path_test <path_test> 
```

__n__: number of hidden units in the RBM, defaults to 100        
__k__: number of steps of contrastive divergence to run per example, defaults to 1         
__eta__: learning rate for contrastive divergence, defaults to 0.001         
__num_epochs__: number of iterations over the entire training data to be performed during the training process, defaults to 1    
__path_train__: path to training data, defaults to "./train.py"     
__path_test__: path to the test data, defaults to "./test.py"     

#### Outputs

The script creaates a folder with the name ```train_loss_k<value_of_k> n<value_of_n> eta<eta> epochs<num_epochs>``` in the working directory which contains the following files:    

__original.png__: one of the images from the dataset which is used to study the convergence of the CD algorithm. The image chosen can be changed by changing the variable image_id in the script.
__origafter.png__: image used to generate another image after the training of the RBM.
__origarecon.png__: image generated after sampling from the learnt distribution.
__changing\_image*n*.png__: an 8 x 8 plot of images each obtained after CD has been performed with 936 examples (there are 60000 training examples) in hte n_th epoch (indexing starts from 0). ```num_epochs``` number of plots are generated in the directory.    
__learning_curves.pdf__: a plot of the variation of the test and training loss with number of epochs.   


### run.sh

A script to run ```train.py``` with different settings of hyperparameters. Run as    

```
./run.sh
```
