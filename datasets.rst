In-Built Datasets
=================
There are in-built datasets provided in both pandas and statsmodels packages.

**Statsmodels**

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


**Sklearn**

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

   
   

  
