# Directory: Assignment5     

## Assginment 5    

This is a programming assignment in which sequence to sequence models based on LSTMs are implemented and are trained using TensorFlow without the use of higher level APIs such as Keras for the purpose of transliteration from English to [Hindi](https://en.wikipedia.org/wiki/Hindi). The following links descibe in full, the problem statement and the results of experiments conducted:      

- [Problem statement](https://drive.google.com/file/d/1nuDzVNkeUEAxcaJR8W5DSdAH5ALDYazz/view?usp=sharing)      
- [Train data](https://drive.google.com/file/d/1sNiPr-q04-ZtTZ0Rf3CXUTr4VD_j8z8l/view?usp=sharing)    
- [Validation data](https://drive.google.com/file/d/16VOm_unAvc0fiPSmYZGz2ZRTCarbJRNa/view?usp=sharing)     
- [Test data](https://drive.google.com/file/d/1UsUwJJO03-G2_WqYH7dTK0-hrbWRp6UV/view?usp=sharing)             
- [My report](https://drive.google.com/file/d/1XHRo_kodHXXGn6Kff412ashSFbOKu79a/view?usp=sharing)     
- [Best test predictions](https://drive.google.com/file/d/1iDk7MHwNEHQ-C1wLo2FyaL1Q4rkiJ0o7/view?usp=sharing) (41.714% accuracy, baseline of 65%, position 10/44 on [leaderboard](https://www.kaggle.com/c/programming-assignment-3/leaderboard))     

## Contents   

### train.py

This script contains code for implementation and training of a sequence to sequence model with a bidirectional LSTM encoder and a two layer decoder for the task of transliteration. Any changes to the architecture must be made by changing the code and no commandline arguements are supported. Adam optimizer is used to train the network with cross entropy as the loss function. 

#### Usage    

Run as 
```
python train.py --lr <learning_rate>  --batch_size <batch_size> --init <init> --dropout_prob <dropout_probab> --save_dir <path_save_dir> --epochs <num_epochs>  --train <path_to_train> --val <path_to_val> --test <path_to_test>
```


__learning_rate__: learning rate to be used for all updates, defaults to 0.001      
__batch_size__: size of minibatch, defaults to 256      
__init__: initialization, 1 corresponds to Xavier and 2 corresponds to He initialization, defaults to 1         
__dropout_probab__: dropout probability, defaults to 0.8      
__path_save_dir__: path to the folder where the final model is stored                 
__num_epochs__: number of epochs to run for, defaults to 10     
__path_to_train__: path to the training data .csv file    
__path_to_val__: path to the validation dataset .csv file     
__path_to_test__: path to the test dataset .csv file     



#### Outputs


__```path_save_dir```__: contains details of the model trained as the output of the tensorflow saver     
__./loss.pdf__: a plot of the learning curves for the RNN training    
__./accuracy.pdf__: a plot of the variation of accuracy with number of epochs     


__Note__: ```./``` indicates that the file is created in the working directory

### run.sh   

A shell file containing the best set of hyperparameters for the given task. Run as described below to train a network with the specified architecture and predict values for the test data.     

```
./run.sh
```    

## Useful links

- [Tensorflow NMT tutorial](https://github.com/tensorflow/nmt)      
- [Sequence to sequence models in tensoflow](https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f)      
- [Suggested reference](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py)      


