Unsupervised Learning
=====================
No labeled responses, the goal is to capture interesting structure or information.

Applications include:
  * Visualise structure of a complex dataset
  * Density estimations to predict probabilities of events
  * Compress and summarise the data
  * Extract features for supervised learning
  * Discover important clusters or outliers

Transformations
---------------
Processes that extract or compute information.

Kernel Density Estimation
*************************

.. figure:: images/kerneldensity.png
    :width: 600px
    :align: center

    University of Michigan: Coursera Data Science in Python

Dimensionality Reduction
************************
  * **Curse of Dimensionality**: Very hard to visualise with many dimensions
  * Finds an approximate version of your dataset using fewer features
  * Used for exploring and visualizing a dataset to understand grouping or relationships
  * Often visualized using a 2-dimensional scatterplot
  * Also used for compression, finding features for supervised learning
  * Can be classified into simple PCA, or manifold techniques

Principal Component Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PCA summarises multiple fields of data into principal components, 
usually just 2 so that it is easier to visualise in a 2-dimensional plot. 
The 1st component will show the most variance of the entire dataset in the hyperplane,
while the 2nd shows the 2nd shows the most variance at a right angle to the 1st.
Because of the strong variance between data points, 
patterns tend to be teased out from a high dimension to even when there's just two dimensions.
These 2 components can serve as new features for a supervised analysis.

.. figure:: images/pca3.png
    :width: 500px
    :align: center

In short, PCA finds the best possible characteristics, 
that summarises the classes of a feature. Two excellent sites elaborate more: setosa_,
quora_. The most challenging part of PCA is interpreting the components.

.. _setosa: http://setosa.io/ev/principal-component-analysis/
.. _quora: https://www.quora.com/What-is-an-intuitive-explanation-for-PCA

.. code:: python

  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  from sklearn.datasets import load_breast_cancer

  cancer = load_breast_cancer()
  df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

  # Before applying PCA, each feature should be centered (zero mean) and with unit variance
  scaled_data = StandardScaler().fit(df).transform(df)  

  pca = PCA(n_components = 2).fit(scaled_data)
  # PCA(copy=True, n_components=2, whiten=False)

  x_pca = pca.transform(scaled_data)
  print(df.shape, x_pca.shape)
  
  # RESULTS
  (569, 30) (569, 2)

To see how much variance is preserved for each dataset.

.. code:: python

   percent = pca.explained_variance_ratio_
   print(percent)
   print(sum(percent))

   # [0.9246348, 0.05238923] 1st component explained variance of 92%, 2nd explained 5%
   # 0.986 total variance explained from 2 components is 97%

Plotting the PCA-transformed version of the breast cancer dataset. 
We can see that malignant and benign cells cluster between two groups and can apply a linear classifier
to this two dimensional representation of the dataset.

.. code:: python

    plt.figure(figsize=(8,6))
    plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'], cmap='plasma', alpha=0.4, edgecolors='black', s=65);
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

  
.. figure:: images/pca1.png
    :width: 500px
    :align: center
      
      
Plotting the magnitude of each feature value for the first two principal components.
This gives the best explanation for the components for each field.

.. code:: python

  fig = plt.figure(figsize=(8, 4))
  plt.imshow(pca.components_, interpolation = 'none', cmap = 'plasma')
  feature_names = list(cancer.feature_names)

  plt.gca().set_xticks(np.arange(-.5, len(feature_names)));
  plt.gca().set_yticks(np.arange(0.5, 2));
  plt.gca().set_xticklabels(feature_names, rotation=90, ha='left', fontsize=12);
  plt.gca().set_yticklabels(['First PC', 'Second PC'], va='bottom', fontsize=12);

  plt.colorbar(orientation='horizontal', ticks=[pca.components_.min(), 0, 
                                                pca.components_.max()], pad=0.65);

                                                
.. figure:: images/pca2.png
    :width: 600px
    :align: center


We can also plot the feature magnitudes in the scatterplot like in R into two separate axes, also known as a biplot.
This shows the relationship of each feature's magnitude clearer in a 2D space.

.. code:: python

    # put feature values into dataframe
    components = pd.DataFrame(pca.components_.T, index=df.columns, columns=['PCA1','PCA2'])

    # plot size
    plt.figure(figsize=(10,8))

    # main scatterplot
    plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'], cmap='plasma', alpha=0.4, edgecolors='black', s=40);
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.ylim(15,-15);
    plt.xlim(20,-20);

    # individual feature values
    ax2 = plt.twinx().twiny();
    ax2.set_ylim(-0.5,0.5);
    ax2.set_xlim(-0.5,0.5);

    # reference lines
    ax2.hlines(0,-0.5,0.5, linestyles='dotted', colors='grey')
    ax2.vlines(0,-0.5,0.5, linestyles='dotted', colors='grey')

    # offset for labels
    offset = 1.07

    # arrow & text
    for a, i in enumerate(components.index):
        ax2.arrow(0, 0, components['PCA1'][a], -components['PCA2'][a], \
                alpha=0.5, facecolor='white', head_width=.01)
        ax2.annotate(i, (components['PCA1'][a]*offset, -components['PCA2'][a]*offset), color='orange')
        
.. figure:: images/pca4.png
    :width: 600px
    :align: center


Multi-Dimensional Scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Multi-Dimensional Scaling (MDS) is a type of manifold learning algorithm that to visualize 
a high dimensional dataset and project it onto a lower dimensional space - 
in most cases, a two-dimensional page. PCA is weak in this aspect.

sklearn gives a good overview of various manifold techniques. https://scikit-learn.org/stable/modules/manifold.html

.. code:: python

  from adspy_shared_utilities import plot_labelled_scatter
  from sklearn.preprocessing import StandardScaler
  from sklearn.manifold import MDS

  # each feature should be centered (zero mean) and with unit variance
  X_fruits_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)  

  mds = MDS(n_components = 2)

  X_fruits_mds = mds.fit_transform(X_fruits_normalized)

  plot_labelled_scatter(X_fruits_mds, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
  plt.xlabel('First MDS feature')
  plt.ylabel('Second MDS feature')
  plt.title('Fruit sample dataset MDS');

.. figure:: images/mds.png
    :width: 600px
    :align: center


t-SNE
^^^^^^
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a powerful manifold learning algorithm for visualizing clusters. It finds a two-dimensional representation of your data, 
such that the distances between points in the 2D scatterplot match as closely as possible the distances 
between the same points in the original high dimensional dataset. In particular, 
t-SNE gives much more weight to preserving information about distances between points that are neighbors. 

More information here_.

.. _here: https://distill.pub/2016/misread-tsne

.. code:: python

  from sklearn.manifold import TSNE

  tsne = TSNE(random_state = 0)

  X_tsne = tsne.fit_transform(X_fruits_normalized)

  plot_labelled_scatter(X_tsne, y_fruits, 
      ['apple', 'mandarin', 'orange', 'lemon'])
  plt.xlabel('First t-SNE feature')
  plt.ylabel('Second t-SNE feature')
  plt.title('Fruits dataset t-SNE');

.. figure:: images/tsne.png
    :width: 600px
    :align: center
    
    You can see how some dimensionality reduction methods may be less successful on some datasets. 
    Here, it doesn't work as well at finding structure in the small fruits dataset, compared to other methods like MDS.
    
Clustering
----------
Find groups in data & assign every point in the dataset to one of the groups.

K-Means
**************************
Need to specify K number of clusters. It is also important to scale the features before applying K-means,
unless the fields are not meant to be scaled, like distances.
Categorical data is not appropriate as clustering calculated using euclidean distance (means). 
For long distances over an lat/long coordinates, they need to be projected to a flat surface.

One aspect of k means is that different random starting points for the cluster centers often result in very different clustering solutions. 
So typically, the k-means algorithm is run in scikit-learn with ten different random initializations 
and the solution occurring the most number of times is chosen. 

.. figure:: images/kmeans4.png
    :width: 600px
    :align: center

    Introduction to Machine Learning with Python

**Methodology**
  1. Specify number of clusters (3)
  2. 3 random data points are randomly selected as cluster centers
  3. Each data point is assigned to the cluster center it is cloest to
  4. Cluster centers are updated to the mean of the assigned points
  5. Steps 3-4 are repeated, till cluster centers remain unchanged

.. figure:: images/kmeans2.png
    :width: 600px
    :align: center
    
    University of Michigan: Coursera Data Science in Python

**Example 1**

.. code:: python

  from sklearn.datasets import make_blobs
  from sklearn.cluster import KMeans
  from adspy_shared_utilities import plot_labelled_scatter
  from sklearn.preprocessing import MinMaxScaler

  fruits = pd.read_table('fruit_data_with_colors.txt')
  X_fruits = fruits[['mass','width','height', 'color_score']].as_matrix()
  y_fruits = fruits[['fruit_label']] - 1

  X_fruits_normalized = MinMaxScaler().fit(X_fruits).transform(X_fruits)  

  kmeans = KMeans(n_clusters = 4, random_state = 0)
  kmeans.fit(X_fruits)

  plot_labelled_scatter(X_fruits_normalized, kmeans.labels_, 
                        ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])

.. figure:: images/kmeans3.png
    :width: 600px
    :align: center


**Example 2**

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
  :scale: 50 %
  :align: center


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
  :scale: 50 %
  :align: center
  

Gaussian Mixture Model
************************

GMM is, in essence a density estimation model but can function like clustering. It has a probabilistic model under the hood so it 
returns a matrix of probabilities belonging to each cluster for each data point. More: https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html


We can input the `covariance_type` argument such that it can choose between `diag` (the default, ellipse constrained to the axes), 
`spherical` (like k-means), or `full` (ellipse without a specific orientation).

.. code:: python

  from sklearn.mixture import GaussianMixture

  # gmm accepts input as array, so have to convert dataframe to numpy
  input_gmm = normal.values

  gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
  gmm.fit(input_gmm)



.. figure:: images/gmm1.PNG
  :width: 500px
  :align: center

  from Python Data Science Handbook by Jake VanderPlas


`BIC` or `AIC` are used to determine the optimal number of clusters, the former usually recommends a simpler model. 
Note that number of clusters or components measures how well GMM works as a density estimator, not as a clustering algorithm.

.. code:: python

  from sklearn.mixture import GaussianMixture
  import matplotlib.pyplot as plt
  %matplotlib inline 
  %config InlineBackend.figure_format = 'retina'

  input_gmm = normal.values

  bic_list = []
  aic_list = []
  ranges = range(1,30)

  for i in ranges:
      gmm = GaussianMixture(n_components=i).fit(input_gmm)
      # BIC
      bic = gmm.bic(input_gmm)
      bic_list.append(bic)
      # AIC
      aic = gmm.aic(input_gmm)
      aic_list.append(aic)

  plt.figure(figsize=(10, 5))
  plt.plot(ranges, bic_list, label='BIC');
  plt.plot(ranges, aic_list, label='AIC');
  plt.legend(loc='best');



.. figure:: images/gmm2.PNG
  :width: 450px
  :align: center

  from Python Data Science Handbook by Jake VanderPlas





Agglomerative Clustering
************************

Agglomerative Clustering is a method of clustering technique used to build clusters from bottom up.

.. figure:: images/aggocluster.png
    :width: 600px
    :align: center
    
    University of Michigan: Coursera Data Science in Python

Methods of linking clusters together.
    
.. figure:: images/aggocluster2.png
    :width: 600px
    :align: center
    
    University of Michigan: Coursera Data Science in Python
        
        
.. code:: python  
  
  from sklearn.datasets import make_blobs
  from sklearn.cluster import AgglomerativeClustering
  from adspy_shared_utilities import plot_labelled_scatter

  X, y = make_blobs(random_state = 10)

  cls = AgglomerativeClustering(n_clusters = 3)
  cls_assignment = cls.fit_predict(X)

  plot_labelled_scatter(X, cls_assignment, 
          ['Cluster 1', 'Cluster 2', 'Cluster 3'])
          
.. figure:: images/aggocluster3.png
    :width: 600px
    :align: center

One of the benfits of this clustering is that a hierarchy can be built.

.. code:: python

  X, y = make_blobs(random_state = 10, n_samples = 10)
  plot_labelled_scatter(X, y, 
          ['Cluster 1', 'Cluster 2', 'Cluster 3'])
  print(X)

  [[  5.69192445  -9.47641249]
   [  1.70789903   6.00435173]
   [  0.23621041  -3.11909976]
   [  2.90159483   5.42121526]
   [  5.85943906  -8.38192364]
   [  6.04774884 -10.30504657]
   [ -2.00758803  -7.24743939]
   [  1.45467725  -6.58387198]
   [  1.53636249   5.11121453]
   [  5.4307043   -9.75956122]]
   
   # BUILD DENDROGRAM
   from scipy.cluster.hierarchy import ward, dendrogram
   plt.figure()
   dendrogram(ward(X))
   plt.show()


.. figure:: images/aggocluster4.png
    :width: 600px
    :align: center
        
More in this link: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/`

DBSCAN
*******
Density-Based Spatial Clustering of Applications with Noise (DBSCAN). 
Need to scale/normalise data. DBSCAN works by identifying crowded regions
referred to as dense regions.

Key parameters are ``eps`` and ``min_samples``. 
If there are at least min_samples many data points within a distance of eps
to a given data point, that point will be classified as a core sample.
Core samples that are closer to each other than the distance eps are put into
the same cluster by DBSCAN.

.. figure:: images/dbscan1.png
    :width: 650px
    :align: center

    Introduction to Machine Learning with Python

**Methodology**
  1. Pick an arbitrary point to start
  2. Find all points with distance *eps* or less from that point
  3. If points are more than *min_samples* within distance of *esp*, point is labelled as a core sample, and assigned a new cluster label
  4. Then all neighbours within *eps* of the point are visited
  5. If they are core samples their neighbours are visited in turn and so on
  6. The cluster thus grows till there are no more core samples within distance *eps* of the cluster
  7. Then, another point that has not been visited is picked, and step 1-6 is repeated
  8. 3 kinds of points are generated in the end, core points, boundary points, and noise
  9. Boundary points are core clusters but not within distance of *esp*



.. figure:: images/dbscan.png
    :width: 600px
    :align: center
    
    University of Michigan: Coursera Data Science in Python

.. code:: python

  from sklearn.cluster import DBSCAN
  from sklearn.datasets import make_blobs

  X, y = make_blobs(random_state = 9, n_samples = 25)

  dbscan = DBSCAN(eps = 2, min_samples = 2)

  cls = dbscan.fit_predict(X)
  print("Cluster membership values:\n{}".format(cls))
  >>> Cluster membership values:
      [ 0  1  0  2  0  0  0  2  2 -1  1  2  0  0 -1  0  0  1 -1  1  1  2  2  2  1]
      # -1 indicates noise or outliers

  plot_labelled_scatter(X, cls + 1, 
          ['Noise', 'Cluster 0', 'Cluster 1', 'Cluster 2'])
          
          
.. figure:: images/dbscan2.png
    :width: 600px
    :align: center