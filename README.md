# CS7015 || Deep learning || IIT Madras

This repository contains the assignments completed as part of a brilliant introductory [course](http://www.cse.iitm.ac.in/~miteshk/CS7015.html) taught by Prof. Mitesh Khapra on Deep learning during the January - May 2019 semester.


## Contents

__Assignment1__: A theoretical assignment which serves as a refresher for Calculus.   
__Assignment2__: A theoretical assignment which serves as a refresher for Linear Algebra.    
__Assignment3__: A programming assignment to implement a feedforward neural network, backpropagation and gradient descent (and its variants) in NumPy without any deep learning frameworks.    
__environment.yml__: A .yml file to setup the environment I used to run all the Python code. Most networks were trained in this environment on my GTX 1060 GPU. The tensorflow-gpu and Nvidia driver versions were chosen accordingly. Details of how to use this file are given below. 


## Setting up the conda environment   

All code is written in Python (and functional in version 3.6.8) and uses the libraries listed in ```environment.yml```. This assumes that you have miniconda installed. If you do not, get it [here](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).     

Create a conda environment in which all the code can be run using the following command in the terminal:       
```
conda env create -f environment.yml
```     

This will create an environment named *deep_learning* (name can be changed by changing the value of "name" in the ```environment.yml``` file). Activate the environment using 
```
conda activate deep_learning
```     

and run the desired script as described.