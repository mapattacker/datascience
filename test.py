import pandas as pd
from sklearn.datasets import load_iris

# Load Iris data (https://en.wikipedia.org/wiki/Iris_flower_data_set)
iris = load_iris()
# Load iris into a dataframe and set the field names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

