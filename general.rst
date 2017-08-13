General Notes
=============

A parsimonious model is a the model that accomplishes the desired level of prediction with as few predictor variables as possible.

Variables
---------
``x`` = independent variable = explanatory = predictor

``y`` = dependent variable = response = target


Data Types
----------
The type of data is essential as it determines what kind of tests can be applied to it.

``Continuous:`` Also known as quantitative. Unlimited number of values

``Categorical:`` Also known as discrete or qualitative. Fixed number of values or *categories*


Bias-Variance Tradeoff
-----------------------
The best predictive algorithm is one that has good *Generalization Ability*.
With that, it will be able to give accurate predictions to new and previously unseen data.

*High Bias* results from *Underfitting* the model. This usually results from erroneous assumptions, and cause the model to be too general.

*High Variance* results from *Overfitting* the model, and it will predict the training dataset very accurately, but not with unseen new datasets.
This is because it will fit even the slightless noise in the dataset.

The best model with the highest accuarcy is the middle ground between the two.

.. figure:: ./images/bias-variance.png
    :scale: 25 %
    :align: center

    from Andrew Ng's lecture

Steps to Build a Predictive Model
--------------------------------------------
Train Test Split
*****************
Split the dataset into *Train* and *Test* datasets.
By default, sklearn assigns 75% to train & 25% to test randomly.
A random state (seed) can be selected to fixed the randomisation

.. code:: Python
  
  from sklearn.cross_validation import train_test_split

  X_train, X_test, y_train, y_test
  = train_test_split(predictor, target, test_size=0.25, random_state=0)

Create Model
************
Choose model and set model parameters (if any).

.. code:: Python

  clf = DecisionTreeClassifier()


Fit Model
************
Fit the model using the training dataset.

.. code:: Python

  model = clf.fit(X_train, y_train)

>>> print model
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')

Test Model
**********
Test the model by predicting identity of unseen data using the testing dataset.

.. code:: Python

  y_predict = model.predict(X_test)


Score Model
***********
Use a confusion matrix and...

>>> print sklearn.metrics.confusion_matrix(y_test, predictions)
[[14  0  0]
 [ 0 13  0]
 [ 0  1 10]]

accuarcy percentage score to obtain the predictive accuarcy.

>>> print sklearn.metrics.accuracy_score(y_test, y_predict)*100, '%'
97.3684210526 %
