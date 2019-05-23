Feature Engineering
=====================
Feature Engineering is one of the most important part of model building.
Collecting and creating of relevant features are most often the determinant of 
a high prediction value.

Manual 
--------

Fourier Transformation
***********************
Converts amplitudes into frequencies

Wavelet Package Analysis
***************************

Auto
-----
Automatic generation of new features from existing ones are starting to gain popularity,
as it can save a lot of time. 

tsfresh
********
tsfresh is a feature extraction package for time-series. It can extract more than 1200 different features,
and filter out features that are deemed relevant. In essence, it is a univariate feature extractor.

https://tsfresh.readthedocs.io/en/latest/

.. code:: python

    from tsfresh import extract_relevant_features

    features_filtered_direct = extract_relevant_features(timeseries, y,
                                                        column_id='id', column_sort='time')

FeatureTools
*************
FeatureTools is extremely useful if you have datasets with a base data, with other tables
that have relationships to it.

We first create an **EntitySet**, which is like a database. Then we create **entities**, i.e., individual
tables with a unique id for each table, and showing their **relationships** between each other.

https://github.com/Featuretools/featuretools

.. code:: python

    import featuretools as ft

    def make_entityset(data):
    es = ft.EntitySet('Dataset')
    es.entity_from_dataframe(dataframe=data,
                             entity_id='recordings',
                             index='index',
                             time_index='time')

    es.normalize_entity(base_entity_id='recordings', 
                        new_entity_id='engines',
                        index='engine_no')

    es.normalize_entity(base_entity_id='recordings', 
                        new_entity_id='cycles',
                        index='time_in_cycles')
    return es
    es = make_entityset(data)
    es

We then use something called **Deep Feature Synthesis (dfs)** to generate features automatically.

**Primitives** are the type of new features to be extracted from the datasets. They can be 
**aggregations** (data is combined) or **transformation** (data is changed via a function) type of extractors.
The list can be found via ``ft.primitives.list_primitives()``.
External primitives like tsfresh, or custom calculations can also be input into FeatureTools.

.. code:: python

    feature_matrix, feature_names = ft.dfs(entityset=es, 
                                            target_entity = 'normal',
                                            agg_primitives=['last', 'max', 'min'],
                                            trans_primitives=[], 
                                            max_depth = 2, 
                                            verbose = 1, 
                                            n_jobs = 3)
    # see all old & new features created
    feature_matrix.columns

FeatureTools appears to be a very powerful auto-feature extractor. Some resources to 
read further are as follows:

 * https://brendanhasz.github.io/2018/11/11/featuretools
 * https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219
 * https://medium.com/@rrfd/simple-automatic-feature-engineering-using-featuretools-in-python-for-classification-b1308040e183
