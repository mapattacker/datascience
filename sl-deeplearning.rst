Deep Learning
===============
Deep Learning falls under the broad class of Articial Intelligence > Machine Learning.
It is a Machine Learning technique that uses multiple internal layers (**hidden layers**) of
non-linear processing units (**neurons**) to conduct supervised or unsupervised learning from data.

Some basic concepts of Deep Learning include:
  * Neutron, Perceptron
  * Neutral Network (NN)
  * Convolutional Neural Network (CNN)
  * Recurrent Neural Netowk (RNN)
  
  
Playground
-----------
Best applicaton to learn about NN without math. View it here_.

.. figure:: images/tensorflow_pg.png
    :width: 600px
    :align: center

    University of Michigan: Coursera Data Science in Python

.. _here: http://playground.tensorflow.org

  1. Learning Rate
    * Determines the learning speed (0.00001 to 10)
  2. Activation
    * Select type of activation function 
    * ReLU, Tanh, Sigmoid, Linear
  3. Regularizaton
    * Type of Regularization to reduce overfitting
    * ``L1``: can reduce coefficients to 0. Useful for few features.
    * ``L1``: useful for inputs that are correlated. 
  4. Regularization Rate
    * 0 to 10
  5. Problem type
    * Classificatoin or Regression

Other variables include adjusting the # neurons (max: 8), # hidden layers (max: 6), data type, noise, batch size.

Here's an example output of using **1 hidden layer** with **1, 2 & 3 neurons** in that layer. 
1 neuron can only split by one straight line.

.. figure:: images/tensorflow_pg1.png
    :width: 500px
    :align: center

For more complicated datasets, more hidden layers need to be added.

.. figure:: images/tensorflow_pg2.png
    :width: 600px
    :align: center
    
