# Directory: Assignment4     

## Assginment 4    

This is a programming assignment in which convolutional neural networks are implemented and are trained using TensorFlow without the use of higher level APIs such as Keras. The following links descibe in full, the problem statement and the results of experiments conducted:      

- [Problem statement](https://drive.google.com/file/d/1jWMBiAYhAzqpBd5um-hn-539QQxQw2vu/view?usp=sharing)      
- [Train data](https://drive.google.com/file/d/1nvse-2dDZ1fmFU1LXridPU2W1CLqfC-a/view?usp=sharing)    
- [Validation data](https://drive.google.com/file/d/1HSd6y7m8Fa4Zm0xD_0gPoZUOKNNZ1rRr/view?usp=sharing)     
- [Test data](https://drive.google.com/file/d/1-06hjxnA6D88u5HJXYNCMF0lXQ3UBz0d/view?usp=sharing)             
- [My report](https://drive.google.com/file/d/1oNV1kge-6_L2fUnDqFHy_sgmUDDsPUi7/view?usp=sharing)     
- [Best test predictions](https://drive.google.com/file/d/11K86rBp1UPUN26DMeX_3SHqtO22NB5_T/view?usp=sharing) (55.506% accuracy, baseline of 50%)     

## Contents   

### train.py

This script contains code for implementation and training of different convolutional neural networks for classification. The network to be trained must be changed within the code. Adam optimizer is used to train the network with cross entropy as the loss function. Data augmentation uses simple tricks such as flipping the images vertically, horizontally and rotating hte images.

#### Usage    

Run as 
```
python train.py --lr <learning_rate>  --batch_size <batch_size> --init <init> --save_dir <path_save_dir> --epochs <num_epochs> --dataAugment <augmentation> --train <path_to_train> --val <path_to_val> --test <path_to_test>
```


__learning_rate__: learning rate to be used for all updates, defaults to 0.001      
__batch_size__: size of minibatch, defaults to 256      
__init__: initialization, 1 corresponds to Xavier and 2 corresponds to He initialization, defaults to 1         
__path_save_dir__: path to the folder where the final model is stored           
__num_epochs__: number of epochs to run for, defaults to 10     
__augmentation__: set to 0 for no augmentation, 1 for augmentation   
__path_to_train__: path to the training data .csv file    
__path_to_val__: path to the validation dataset .csv file     
__path_to_test__: path to the test dataset .csv file     



#### Outputs

__./output/__: Folder containing the graph and other details of the model saved using tensorflow summary writer     
__./visualisation.pdf__: a plot of the 32 filters of the first layer      
__./submission_2lay.csv__: a csv file containing the predictions for the test data    
__./loss_vs_number_of_epochs.pdf__: a plot of the learning curves

__Note__: ```./``` indicates that the file is created in the working directory

### run.sh   

A shell file containing the best set of hyperparameters for the given task. Run as described below to train a network with the specified architecture and predict values for the test data.     

```
./run.sh
```    


