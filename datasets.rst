In-Built Datasets
=================
There are in-built datasets provided in both statsmodels and sklearn packages.

**Statsmodels**

In statsmodels, many R datasets can be obtained from the function ``sm.datasets.get_rdataset()``. 
To view each dataset's description, use ``print(duncan_prestige.__doc__)``.

.. code:: python

  import statsmodels.api as sm
  prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data
  print prestige.head()


  
  type  income  education  prestige
  accountant  prof      62         86        82
  pilot       prof      72         76        83
  architect   prof      75         92        90
  author      prof      55         90        76
  chemist     prof      64         86        90

|
**Sklearn**

There are five common toy datasets here. For others, view http://scikit-learn.org/stable/datasets/index.html. 
To view each dataset's description, use ``print boston['DESCR']``.

+------------------------------------+-----------------------------------------------------------------+
| load_boston([return_X_y])          | Load and return the boston house-prices dataset (regression).   |
+------------------------------------+-----------------------------------------------------------------+
| load_iris([return_X_y])            | Load and return the iris dataset (classification).              |
+------------------------------------+-----------------------------------------------------------------+
| load_diabetes([return_X_y])        | Load and return the diabetes dataset (regression).              |
+------------------------------------+-----------------------------------------------------------------+
| load_digits([n_class, return_X_y]) | Load and return the digits dataset (classification).            |
+------------------------------------+-----------------------------------------------------------------+
| load_linnerud([return_X_y])        | Load and return the linnerud dataset (multivariate regression). |
+------------------------------------+-----------------------------------------------------------------+


.. code:: python

  from sklearn.datasets import load_iris
  
  # Load Iris data (https://en.wikipedia.org/wiki/Iris_flower_data_set)
  iris = load_iris()
  # Load iris into a dataframe and set the field names
  df = pd.DataFrame(iris.data, columns=iris.feature_names)
  df.head()
  
  sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
  0                5.1               3.5                1.4               0.2
  1                4.9               3.0                1.4               0.2
  2                4.7               3.2                1.3               0.2
  3                4.6               3.1                1.5               0.2
  4                5.0               3.6                1.4               0.2
  
  
  
  # Feature names are in .target & .target_names
  >>> print iris.target_names[:5]
  >>> ['setosa' 'versicolor' 'virginica']
  >>> print iris.target
  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
   2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
   2 2]
   
  
  
  # Change target to target_names & merge with main dataframe
  df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
  print df['species'].head()
  
  sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
  0                5.1               3.5                1.4               0.2
  1                4.9               3.0                1.4               0.2
  2                4.7               3.2                1.3               0.2
  3                4.6               3.1                1.5               0.2
  4                5.0               3.6                1.4               0.2
  0    setosa
  1    setosa
  2    setosa
  3    setosa
  4    setosa
  Name: species, dtype: category
  Categories (3, object): [setosa, versicolor, virginica]

   
   

  
