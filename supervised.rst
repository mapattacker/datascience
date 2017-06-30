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

Create Model

.. code:: python

  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5)

Fit Model

.. code:: python

  knn.fit(X_train, y_train)
  
.. 

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
         weights='uniform')


Decision Tree
**************************

Random Forest
**************************

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