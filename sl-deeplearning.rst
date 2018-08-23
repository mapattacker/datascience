Deep Learning
===============
Deep Learning falls under the broad class of Articial Intelligence > Machine Learning.
It is a Machine Learning technique that uses multiple internal layers (**hidden layers**) of
non-linear processing units (**neurons**) to conduct supervised or unsupervised learning from data.

Some basic concepts of Deep Learning include:
  * Neuron, Perceptron
  * Neural Network (NN)
  * Convolutional Neural Network (CNN)
  * Recurrent Neural Netowk (RNN)
  
  
Playground
-----------
Tensorflow Playground is the best applicaton to learn about NN without math. 
View it here_. An excellent guide_ also gives a great background on the basics.

.. figure:: images/tensorflow_pg.png
    :width: 600px
    :align: center


.. _here: http://playground.tensorflow.org
.. _guide: https://cloud.google.com/blog/big-data/2016/07/understanding-neural-networks-with-tensorflow-playground


:Learning Rate: Determines the learning speed (0.00001 to 10)
:Activation: The activation function is what tells the perceptron to fire or not. eg. (ReLU, Tanh, Sigmoid, Linear)
:Regularizaton: Type of Regularization to reduce overfitting. ``L1``: can reduce coefficients to 0. Useful for few features. ``L1``: useful for inputs that are correlated. 
:Regularization Rate: 0 to 10
:Problem Type: Classification or Regression



Other variables include adjusting the # neurons (max: 8), # hidden layers (max: 6), data type, noise, batch size.

Here's an example output of using **1 hidden layer** with **1, 2 & 3 neurons** in that layer. 
1 neuron can only split by one straight line.

.. figure:: images/tensorflow_pg1.png
    :width: 400px
    :align: center

For more complicated datasets, more hidden layers need to be added.

.. figure:: images/tensorflow_pg2.png
    :width: 600px
    :align: center
    
From these examples, we can see that a **hidden layer** contains the intelligence
in a distributed fashion using many ``neurons``, ``interconnection``, ``weights``,
``activation functions``, etc. **Deep NN** have multiple neutral networks.


Backpropagation 
---------------
Backpropagation (BP) uses training iterations where **error size** is used
to determine the updated value of each weight in the NN.

.. figure:: images/backp1.png
    :width: 600px
    :align: center

NN
----

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
**Convolutional Neural Network** (CNN) is a Feedforward (FF) Neural Network (NN).
  * Designed based on animals' visual cortex. Where visual neurons progressively focus on overlapping tiles & sequentially shifts **convulation process** to cover the entire visual field.
  * Uses **Multi-Layer Perceptrons** (MLPs)
  * **ReLU** activation is often used
  * Image/video recognition, recommender systems, natural language processing

Subsampling
************
Median Value


RNN
----
**Recurrent Neural Network** (RNN)




