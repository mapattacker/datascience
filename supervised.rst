Supervised Learning
===================

Classification
--------------

K Nearest Neighbours (KNN)
**************************
1. ``Distance Metric:`` Eclidean Distance (default). In sklearn it is known as (Minkowski with p = 2)
2. ``How many nearest neighbour to look at:`` k=1 very specific, k=5 more general model. Use nearest k data points to determine classification
3. ``Weighting function on neighbours:`` (optional)
4. ``How to aggregate class of neighbour points:`` Simple majority (default)

**Train Test Split**

.. code:: python

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


**Create Model**

.. code:: python

  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5)

**Fit Model**

>>> knn.fit(X_train, y_train)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
         weights='uniform')

**Test Model**

>>> knn.score(X_test, y_test)
0.53333333333333333


Decision Tree
**************************
Uses gini index to split the data at binary level.

**Strengths:** Can select a large number of features that best determine the targets.
**Weakness:** Tends to overfit the data as it will split till the end.
Pruning can be done to remove the leaves to prevent overfitting but that is not available in sklearn.
Small changes in data can lead to different splits. Not very reproducible for future data (see random forest).


**Train Test Split**

.. code:: python

  train_predictor, test_predictor, train_target, test_target = train_test_split(predictor, target, test_size=0.25)

>>> print test_predictor.shape
>>> print train_predictor.shape
(38, 4)
(112, 4)

**Create Model**

.. code:: python

  from sklearn.tree import DecisionTreeClassifier
  clf = DecisionTreeClassifier()

**Fit Model**

.. code:: python

  model = clf.fit(train_predictor, train_target)

>>> print model
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')

**Test Model**

.. code:: python

  predictions = model.predict(test_predictor)

**Score Model**

>>> print sklearn.metrics.confusion_matrix(test_target,predictions)
>>> print sklearn.metrics.accuracy_score(test_target, predictions)*100, '%'
[[14  0  0]
 [ 0 13  0]
 [ 0  1 10]]
97.3684210526 %

.. code:: python

  # it is easier to use this package that does everything nicely for a perfect confusion matrix
  from pandas_confusion import ConfusionMatrix
>>> ConfusionMatrix(test_target, predictions)
Predicted   setosa  versicolor  virginica  __all__
Actual
setosa          14           0          0       14
versicolor       0          13          0       13
virginica        0           1         10       11
__all__         14          14         10       38


**Feature Importance**

.. code:: python

  df2= pd.DataFrame(model.feature_importances_, index=df.columns[:-2])

>>> df2.sort_values(by=0,ascending=False)
petal width (cm)	0.952542
petal length (cm)	0.029591
sepal length (cm)	0.017867
sepal width (cm)	0.000000


Random Forest
**************************
An ensemble of decision trees.


**Import Modules**

.. code:: python

  import pandas as pd
  import numpy as np
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.cross_validation import train_test_split
  import sklearn.metrics

**Train Test Split**

.. code:: python

  train_feature, test_feature, train_target, test_target = train_test_split(feature, target, test_size=.2)

>>> print train_feature.shape
>>> print test_feature.shape
(404, 13)
(102, 13)

**Create Model**

.. code:: python

  # use 100 decision trees
  clf = RandomForestClassifier(n_estimators=100)

**Fit Model**

.. code:: python

  model = clf.fit(train_feature, train_target)

>>> print model
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

**Test Model**

.. code:: python

  predictions = model.predict(test_feature)


**Score Model**

>>> print 'accuracy', '\n', sklearn.metrics.accuracy_score(test_target, predictions)*100, '%', '\n'
>>> print 'confusion matrix', '\n', sklearn.metrics.confusion_matrix(test_target,predictions)
accuracy
82.3529411765 %
confusion matrix
[[21  0  3]
 [ 0 21  4]
 [ 8  3 42]]

**Feature Importance**

.. code:: python

 # rank the importance of features
 df2= pd.DataFrame(model.feature_importances_, index=df.columns[:-2])
>>> df2.sort_values(by=0,ascending=False)
 RM	0.225612
 LSTAT	0.192478
 CRIM	0.108510
 DIS	0.088056
 AGE	0.074202
 NOX	0.067718
 B	0.057706
 PTRATIO	0.051702
 TAX	0.047568
 INDUS	0.037871
 RAD	0.026538
 ZN	0.012635
 CHAS	0.009405

**Optimum Ensemble of Trees**

.. code:: python

 # see how many decision trees are minimally required make the accuarcy consistent
 import numpy as np
 import matplotlib.pylab as plt
 import seaborn as sns
 %matplotlib inline

 trees=range(100)
 accuracy=np.zeros(100)

 for i in range(len(trees)):
    clf=RandomForestClassifier(n_estimators= i+1)
    model=clf.fit(train_feature, train_target)
    predictions=model.predict(test_feature)
    accuracy[i]=sklearn.metrics.accuracy_score(test_target, predictions)

 plt.plot(trees,accuracy)

 # well, seems like more than 10 trees will have a consistent accuracy of 0.82.
 # Guess there's no need to have an ensemble of 100 trees!


.. image:: images/randomforest.png


Logistic Regression
**************************

Support Vector Machine
***********************


Regression
----------

Ordinary Least Squares (OLS) Regression
***************************************
Best fit line ``ŷ = a + bx`` is drawn based on the ordrinary least squares method. i.e., least total area of squares with length from each x,y point to regresson line.


Ridge Regression
****************



Lasso Regression
****************
Least absolute shrinkage and selection operator regression, or LASSO regression, has a unique penalty parameter, lambda that *change unimportant features (their regression coefficients) into 0*.
This helps to prevent *overfitting*.

* Prevent overfitting.
* Uses regularisation.
* Uses a penalty parameter lambda to change unimportant features (their regression coefficients) into 0. When lambda = 0, then it is a normal OLS regression. (Note sklearn name it as alpha instead)

  a. Bias increase & variability decreases when lambda increases.
  b. Useful when there are many features (explanatory variables).
  c. Have to standardize all features so that they have mean 0 and std error 1.
  d. Have several algorithms: LAR (Least Angle Regression). Starts w 0 predictors & add each predictor that is most correlated at each step.

.. note::

  sklearn define lambda as alpha instead.
