Reinforcement Learning
=======================

This series of medium_ articles gave a good description of various types of reinforcement learning
with jupyter notebook descriptions for various games. This includes deep learning using tensorflow.

.. _medium: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0


Markov Decision Problem
------------------------

The core problem of MDPs is to find a "policy" for the decision maker: a function π that specifies the action 
π(s) that the decision maker will choose when in state s. The diagram illustrate the Markov Decision Problem.

.. figure:: images/reinforce1.png
    :width: 400px
    :align: center

    Udacity, Machine Learning for Trading

There are two ways to determine the policy.
    1. Model based
        * using value/policy iteration
    2. Model free

Q-Learning
-----------
Q-Learning is an example of model free reinforcement learning.