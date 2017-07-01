Unsupervised Learning
=====================

Clustering
----------

K-Means
**************************

.. code:: python 

  #### IMPORT MODULES ####
  import pandas as pd
  from sklearn import preprocessing
  from sklearn.cross_validation import train_test_split
  from sklearn.cluster import KMeans
  from sklearn.datasets import load_iris
  


  #### NORMALIZATION ####
  # standardise the means to 0 and standard error to 1
  for i in df.columns[:-2]: # df.columns[:-1] = dataframe for all features, minus target
      df[i] = preprocessing.scale(df[i].astype('float64'))

  df.describe()
  
  
  
  #### TRAIN-TEST SPLIT ####
  train_feature, test_feature = train_test_split(feature, random_state=123, test_size=0.2)

  print train_feature.shape
  print test_feature.shape
  (120, 4)
  (30, 4)



  #### A LOOK AT THE MODEL ####
  >>> KMeans(n_clusters=2)
  KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10,
      n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
      verbose=0)
  
  
  
  #### ELBOW CHART TO DETERMINE OPTIMUM K ####
  from scipy.spatial.distance import cdist
  import numpy as np
  clusters=range(1,10)
  # to store average distance values for each cluster from 1-9
  meandist=[]

  # k-means cluster analysis for 9 clusters                                                           
  for k in clusters:
      # prepare the model
      model=KMeans(n_clusters=k)
      # fit the model
      model.fit(train_feature)
      # test the model
      clusassign=model.predict(train_feature)
      # gives average distance values for each cluster solution
          # cdist calculates distance of each two points from centriod
          # get the min distance (where point is placed in clsuter)
          # get average distance by summing & dividing by total number of samples
      meandist.append(sum(np.min(cdist(train_feature, model.cluster_centers_, 'euclidean'), axis=1)) 
      / train_feature.shape[0])
      
      
  import matplotlib.pylab as plt
  import seaborn as sns
  %matplotlib inline
  """Plot average distance from observations from the cluster centroid
  to use the Elbow Method to identify number of clusters to choose"""

  plt.plot(clusters, meandist)
  plt.xlabel('Number of clusters')
  plt.ylabel('Average distance')
  plt.title('Selecting k with the Elbow Method')

  # look a bend in the elbow that kind of shows where 
  # the average distance value might be leveling off such that adding more clusters 
  # doesn't decrease the average distance as much
  
.. image:: images/elbowchart.png


.. code:: python
  
  #### VIEW CLUSTER USING PCA ####
  # Interpret 3 cluster solution
  model3=KMeans(n_clusters=3)
  model3.fit(train_feature)
  clusassign=model3.predict(train_feature)
  # plot clusters

  # Use Canonical Discriminate Analysis to reduce the dimensions (into 2)
  # Creates a smaller no. of variables, with canonical variables ordered by proportion of variable accounted
  # i.e., 1st canonical variable is most importance & so on

  from sklearn.decomposition import PCA
  pca_2 = PCA(2) #return first two canonical variables
  plot_columns = pca_2.fit_transform(train_feature)
  # plot 1st canonical v in x axis, 2nd variable on y axis
  # color code variables based on cluster assignments (i.e., predicted targets)
  plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_)
  plt.xlabel('Canonical variable 1')
  plt.ylabel('Canonical variable 2')
  plt.title('Scatterplot of Canonical Variables for 3 Clusters')
  plt.show()
  
.. image:: images/kmeans.png