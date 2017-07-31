Cross Validation
-----------------

Takes more time and computation to use k-fold, but well worth the cost. 
By default, sklean uses stratified cross validation. Another type is leave one out cross-validation.

.. image:: \images\kfold.png

.. code:: python

  from sklearn.model_selection import cross_val_score

  clf = KNeighborsClassifier(n_neighbors = 5)
  X = X_fruits_2d.as_matrix()
  y = y_fruits_2d.as_matrix()
  cv_scores = cross_val_score(clf, X, y)

  print('Cross-validation scores (3-fold):', cv_scores)
  print('Mean cross-validation score (3-fold): {:.3f}'
       .format(np.mean(cv_scores)))