Tests for Assumptions
=====================

Normality
---------

.. code:: python

  import scipy.stats as stats
  stats.normaltest(df3['depth'])
  
  >>> NormaltestResult(statistic=33363.134206705407, pvalue=0.0)



Homogeneity of Variances
------------------------