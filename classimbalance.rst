Class Imbalance
================

In domains like predictive maintenance, machine failures are usually rare occurrences in the lifetime of the assets compared to normal operation. 
This causes an imbalance in the label distribution which usually causes poor performance as algorithms tend to 
classify majority class examples better at the expense of minority class examples as the total misclassification error 
is much improved when majority class is labeled correctly. Techniques are available to correct for this.

The imbalance-learn package provides an excellent range of algorithms for adjusting for imbalanced data.
Install with ``pip install -U imbalanced-learn`` or ``conda install -c conda-forge imbalanced-learn``.

An important thing to note is that **resampling must be done AFTER the train-test split**, so as to prevent data leakage.


Over-Sampling
---------------

SMOTE (synthetic minority over-sampling technique) is a common and popular up-sampling technique.

.. code:: python

    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_sample(X_train, y_train)
    clf = LogisticRegression()
    clf.fit(X_resampled, y_resampled)


ADASYN is one of the more advanced over sampling algorithms.

.. code:: python

    from imblearn.over_sampling import ADASYN
    
    ada = ADASYN()
    X_resampled, y_resampled = ada.fit_sample(X_train, y_train)
    clf = LogisticRegression()
    clf.fit(X_resampled, y_resampled)

Under-Sampling
---------------

.. code:: python
    
    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler()
    X_resampled, y_resampled = rus.fit_sample(X_train, y_train)
    clf = LogisticRegression()
    clf.fit(X_resampled, y_resampled)


Under/Over-Sampling
--------------------

SMOTEENN combines SMOTE with Edited Nearest Neighbours, 
which is used to pare down and centralise the negative cases.

.. code:: python

    from imblearn.combine import SMOTEENN

    smo = SMOTEENN()
    X_resampled, y_resampled = smo.fit_sample(X_train, y_train)
    clf = LogisticRegression()
    clf.fit(X_resampled, y_resampled)


Cost Sensitive Classification
------------------------------

One can also make the classifier aware of the imbalanced data by incorporating the weights 
of the classes into a cost function. 
Intuitively, we want to give higher weight to minority class and lower weight to majority class.

http://albahnsen.github.io/CostSensitiveClassification/index.html
