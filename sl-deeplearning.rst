Deep Learning
===============
Deep Learning falls under the broad class of Articial Intelligence > Machine Learning.
It is a Machine Learning technique that uses multiple internal layers (**hidden layers**) of
non-linear processing units (**neurons**) to conduct supervised or unsupervised learning from data.
  

ANN
-----------

Theory
*******

An **artifical neural network** consists of an input layer, hidden layers, and an output layer. 
This writeup by Berkeley_ gave an excellent introduction to the theory. 
Most of the diagrams are taken from the site.

.. _Berkeley https://ml.berkeley.edu/blog/2017/02/04/tutorial-3/

.. figure:: images/deeplearning1.png
    :width: 400px
    :align: center

    Structure of an artificial neutral network

Zooming in at a single perceptron, the input layer consists of every individual features, 
each with an assigned weight feeding to the hidden layer. 
An **activation function** tells the perception what outcome it is.


.. figure:: images/deeplearning2.png
    :width: 400px
    :align: center

    Structure of a single perceptron

Activation functions consists of *ReLU*, *Tanh*, *Linear*, *Sigmoid*, *Softmax* and many others.
Sigmoid is used for binary classifications, while softmax is used for multi-class classifications.

.. figure:: images/deeplearning3.png
    :width: 400px
    :align: center

    An activation function, using sigmoid function

The backward propagation algorithm works in such that the slopes of gradient descent is calculated by 
working backwards from the output layer back to the input layer. The weights are readjusted to reduce the 
loss and improve the accuracy of the model.

.. figure:: images/deeplearning4.png
    :width: 400px
    :align: center

    Backward propagation

**A summary is as follows**

  1. Randomly initialize the weights for all the nodes.
  2. For every training example, perform a forward pass using the current weights, and calculate the output of each node going from left to right. The final output is the value of the last node.
  3. Compare the final output with the actual target in the training data, and measure the error using a loss function.
  4. Perform a backwards pass from right to left and propagate the error to every individual node using backpropagation. Calculate each weight’s contribution to the error, and adjust the weights accordingly using gradient descent. Propagate the error gradients back starting from the last layer.

Keras Model
***************

Building an ANN model in Keras library requires
    * input & hidden layers
    * model compliation
    * model fitting
    * model evalution

Definition of layers are typically done using the typical Dense layer, or regularization layer called Dropout.
The latter prevents overfitting as it randomly selects neurons to be ignored during training.

.. code:: python

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))


Before training, the model needs to be compiled with the learning hyperparameters of optimizer, loss, and metric functions.

.. code:: python
    # from keras documentation
    # https://keras.io/getting-started/sequential-model-guide/

    # For a multi-class classification problem
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # For a binary classification problem
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # For a mean squared error regression problem
    model.compile(optimizer='rmsprop',
                  loss='mse')


The below gives a compiled code example code.

.. code:: python

    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import RMSprop

    (mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

    train_images = mnist_train_images.reshape(60000, 784)
    test_images = mnist_test_images.reshape(10000, 784)
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images /= 255
    test_images /= 255
    
    # convert the 0-9 labels into "one-hot" format, as we did for TensorFlow.
    train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
    test_labels = keras.utils.to_categorical(mnist_test_labels, 10)


    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.summary()


    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 512)               401920    
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 407,050
    Trainable params: 407,050
    Non-trainable params: 0
    _________________________________________________________________

    # set optimizer parameters
    keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(),
                    metrics=['accuracy'])


    history = model.fit(train_images, train_labels,
                        batch_size=100,
                        epochs=10,
                        verbose=2,
                        validation_data=(test_images, test_labels))

    # Train on 60000 samples, validate on 10000 samples
    # Epoch 1/10
    # - 4s - loss: 0.2459 - acc: 0.9276 - val_loss: 0.1298 - val_acc: 0.9606
    # Epoch 2/10
    # - 4s - loss: 0.0991 - acc: 0.9700 - val_loss: 0.0838 - val_acc: 0.9733
    # Epoch 3/10
    # - 4s - loss: 0.0656 - acc: 0.9804 - val_loss: 0.0738 - val_acc: 0.9784
    # Epoch 4/10
    # - 4s - loss: 0.0493 - acc: 0.9850 - val_loss: 0.0650 - val_acc: 0.9798
    # Epoch 5/10
    # - 4s - loss: 0.0367 - acc: 0.9890 - val_loss: 0.0617 - val_acc: 0.9817
    # Epoch 6/10
    # - 4s - loss: 0.0281 - acc: 0.9915 - val_loss: 0.0698 - val_acc: 0.9800
    # Epoch 7/10
    # - 4s - loss: 0.0221 - acc: 0.9936 - val_loss: 0.0665 - val_acc: 0.9814
    # Epoch 8/10
    # - 4s - loss: 0.0172 - acc: 0.9954 - val_loss: 0.0663 - val_acc: 0.9823
    # Epoch 9/10
    # - 4s - loss: 0.0128 - acc: 0.9964 - val_loss: 0.0747 - val_acc: 0.9825
    # Epoch 10/10
    # - 4s - loss: 0.0098 - acc: 0.9972 - val_loss: 0.0840 - val_acc: 0.9795


    
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])





CNN
----
**Convolutional Neural Network** (CNN)



RNN
----
**Recurrent Neural Network** (RNN)




