Reinforcement Learning
=======================

This series of medium_ articles gave a good description of various types of reinforcement learning
with jupyter notebook descriptions for various games. This includes deep learning using tensorflow.

.. _medium: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0


**Markov Decision Problem**

Reinforcement learning helps to solve Markov Decision Problems (MDP).
The core problem of MDPs is to find a "policy" for the decision maker: a function π that specifies the action 
π(s) that the decision maker will choose when in state s. The diagram illustrate the Markov Decision Problem.

.. figure:: images/reinforce1.png
    :width: 400px
    :align: center

    Udacity, Machine Learning for Trading



Q-Learning
-----------
Q-Learning is an example of model-free reinforcement learning to solve the Markov Decision Problem.
It derives the policy by directly looking at the data instead of developing a model.


Updating the function Q uses the following equation.

.. figure:: images/reinforce2.png
    :width: 400px
    :align: center

    from Medium