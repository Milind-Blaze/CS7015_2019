# Directory: Assignment     

## Assginment 3    

This is a programming assignment in which feedforward neural networks of varying architectures are implemented and are trained in NumPy without the use of frameworks such as TensorFlow. The following links descibe in full, the problem statement and the results of experiments conducted:      

- [Problem statement](https://drive.google.com/file/d/1PpCZIV4hWixnR3RQeWQVui11wfOmfnh_/view?usp=sharing)      
- [Train data](https://drive.google.com/file/d/1gnvCkLc7lfkMqJO8M4sndHdYvZZRSjzU/view?usp=sharing)    
- [Validation data](https://drive.google.com/file/d/1eMGC-zMhVp9wI8V8xuXWP6DxMOUd3-5r/view?usp=sharing)     
- [Test data](https://drive.google.com/file/d/1zFWbPvJ7dze5kICO6xtnd-ggORruY79G/view?usp=sharing)             
- [My report](https://drive.google.com/file/d/1oNV1kge-6_L2fUnDqFHy_sgmUDDsPUi7/view?usp=sharing)     
- [Best test predictions](https://drive.google.com/file/d/1MP-dqGZwUpDFY15z3uHlJgptmTD5mhDZ/view?usp=sharing) (95.457% accuracy)     

## Contents   

### train.py

This file contains implementations of the feed forward network, backpropagation, gradient descent, stochastic gradient descent, Adam and NAG. 

#### Usage    

Run as 
```
python train.py --lr <learning_rate> --momentum <gamma> --num_hidden <num_layers> --sizes <num_units> --activation <activation_func> --loss <loss_func> --opt <optimizer> --batch_size <batch_size> --epochs <num_epochs> --anneal <to_anneal>  --save_dir <path_save_dir> --expt_dir <expt_dir> --train <path_to_train> --val <path_to_val> --test <path_to_test> --pretrain <to_pretrain> --state <epoch_to_restore> --testing <to_test>
```


__lr__: learning rate to be used for all updates, defaults to 0.001      
__momentum__: momentum/gamma values for momentum based gradient descent, defaults to 0.5     
__num_hidden__: number of hidden layers       
__sizes__: comma separated values, number of units in each hidden layer      
__activation__: non-linearity to be used after each layer, defaults to relu      
__loss__: loss function to optimize, defaults to "ce" i.e cross entropy           
__opt__: optimizer to be used for learning, defaults to "adam". Could be any one of "gd", "momentum", "adam", "nag"      
__batch_size__: size of minibatch, default is 20      
__num_epochs__: number of epochs to run for, defaults to 10     
__anneal__: a boolean argument. If set to "true", validation loss of the current epoch is compared with the previous epoch and if found to be greater than the latter, the epoch is 	restarted and any weight updates during the epoch are discareded.      
__save_dir__: path to the folder where the final model is stored          
__expt_dir__: path to the folder where the logs are stored     
__train__: path to the training data .csv file    
__val__: path to the validation dataset .csv file     
__test__: path to the test dataset .csv file     
__pretrain__: boolean variable, if set to "true", weigths from the epoch determined by state from a previous training session are loaded   
__state__: epoch from which weights need to be loaded     
__testing__: boolean variable, if set to "true", weights determined by "state" are loaded and predictiosn produced for the test data    


__Note__: Currently, no limit is imposed on the number of times annealing is performed. Therefore, it is quite possible that the algorithm keeps annealing the learning rate until it is a ridiculously low value.     

#### Outputs

__weights\_*epoch_num*.pickle__: parameters/ weights after the ```epoch_num```th epoch. A total of ```num_epochs``` such pickle files are stored at ```save_dir```.    
__log_train.txt__: file containing details of the training loss, error on training data and learning rate for the epoch for each epoch. Created at ```expt_dir```.        
__log_val.txt__: file containing details of the validation loss, error on validation data and learning rate for the epoch for each epoch. Created at ```expt_dir```.        
__readme.txt__: file containing the values of the hyperparameters.     
__test_submission.csv__: created if ```testing``` is set to false. Contains predicted labels for the test data.     
__predicted_*state*.csv*__: created if ```testing``` is set to true. Contains predictiosn for the test data found using the parameters saved for the epoch given by ```state```.     



### run.sh   

A shell file containing the best set of hyperparameters for the given task. Run as described below to train a network with the specified architecture and predict values for the test data.     

```
./run.sh
```    


