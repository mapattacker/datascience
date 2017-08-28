Feature Normalization
=======================
Normalisation is another important concept needed to change all features to the same scale.
This allows for faster convergence on learning, and more uniform influence for all weights.
More on sklearn website:

http://scikit-learn.org/stable/modules/preprocessing.html

Scale
-----
This changes the data to have means of 0 and standard error of 1.

.. code:: python

  import pandas pd
  from sklearn import preprocessing

  # standardise the means to 0 and standard error to 1
  for i in df.columns[:-1]: # df.columns[:-1] = dataframe for all features
    df[i] = preprocessing.scale(df[i].astype('float64'))


Min Max Scale
-------------
Another way to normalise is to use the Min Max Scaler, as defined below:

.. figure:: images/minmaxscaler.png

.. code:: python

  import pandas pd
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()

  from sklearn.linear_model import Ridge
  X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                     random_state = 0)

  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)

Pipeline
---------
Scaling have a chance of leaking the part of the test data in train-test split into the training data.
This is especially inevitable when using cross-validation.
A way to prevent data-leakage is to use the pipeline function in sklearn, which wraps the scaler and classifier together,
and scale them separately during cross validation.

.. code:: python

  from sklearn.pipeline import Pipeline
  pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])

  pipe.fit(X_train, y_train)
  Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svm', SVC(C=1.0, cac
            decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False))])

  pipe.score(X_test, y_test)
  0.95104895104895104