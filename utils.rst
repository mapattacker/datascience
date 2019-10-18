Utilities
==========

This page lists some useful functions and tips to make your datascience journey smoother.

Persistance
-------------
Data, models and scalers are examples of objects that can benefit greatly from pickling. 
For the former, it allows multiples faster loading
compared to other sources since it is saved in a python format.
For others, there are no other ways of saving as they are natively python objects.

Saving dataframes.

.. code:: python

    import pandas as pd

    df.to_pickle('df.pkl')
    df = pd.read_pickle('df.pkl')

Saving models or scalers. More: https://scikit-learn.org/stable/modules/model_persistence.html

.. code:: python

    import pickle

    pickle.dump(model, open('model_rf.pkl', 'wb'))

From sklearn's documentation, it is said that in the specific case of scikit-learn, 
it may be better to use joblib’s replacement of pickle (dump & load), 
which is more efficient on objects that carry large numpy arrays internally 
as is often the case for fitted scikit-learn estimators, 
but can only pickle to the disk and not to a string.

More: https://scikit-learn.org/stable/modules/model_persistence.html

.. code:: python

    import joblib

    joblib.dump(clf, 'model.joblib')
    joblib.load('model.joblib')


Memory Reduction
-----------------
If the dataset is huge, it can be a problem storing the dataframe in memory.
However, we can reduce the dataset size significantly by analysing the data values for each column,
and change the datatype to the smallest that can fit the range of values.
Below is a function created by arjanso in Kaggle that can be plug and play.

.. code:: python

    import pandas as pd
    import numpy as np

    def reduce_mem_usage(df):
        '''
        Source: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
        Reduce size of dataframe significantly using the following process
        1. Iterate over every column
        2. Determine if the column is numeric
        3. Determine if the column can be represented by an integer
        4. Find the min and the max value
        5. Determine and apply the smallest datatype that can fit the range of values
        '''
        start_mem_usg = df.memory_usage().sum() / 1024**2 
        print("Memory usage of properties dataframe is :",start_mem_usg," MB")
        NAlist = [] # Keeps track of columns that have missing values filled in. 
        for col in df.columns:
            if df[col].dtype != object:  # Exclude strings            
                # Print current column type
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",df[col].dtype)            
                # make variables for Int, max and min
                IsInt = False
                mx = df[col].max()
                mn = df[col].min()
                print("min for this col: ",mn)
                print("max for this col: ",mx)
                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(df[col]).all(): 
                    NAlist.append(col)
                    df[col].fillna(mn-1,inplace=True)  
                    
                # test if column can be converted to an integer
                asint = df[col].fillna(0).astype(np.int64)
                result = (df[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True            
                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            df[col] = df[col].astype(np.uint8)
                        elif mx < 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif mx < 4294967295:
                            df[col] = df[col].astype(np.uint32)
                        else:
                            df[col] = df[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)    
                # Make float datatypes 32 bit
                else:
                    df[col] = df[col].astype(np.float32)
                
                # Print new column type
                print("dtype after: ",df[col].dtype)
                print("******************************")
        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = df.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
        return df, NAlist


Jupyter Extension
------------------

Jupyter Notebook is the go-to IDE for data science. 
However, it can be further enhanced using jupyter extensions.
``pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install``

Some of my favourite extensions are:
 * Table of Contents*: Sidebar showing TOC based on 
 * *ExecuteTime*: Time to execute script for each cell
 * *Variable Inspector*: Overview of all variables saved in memory. Allow deletion of variables to save memory.

More: https://towardsdatascience.com/jupyter-notebook-extensions-517fa69d2231